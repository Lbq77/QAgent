"""GraphRAG 知识库客户端：本地调用 graphrag.api 查询。

索引产物 parquet 放在本目录 out/ 下，settings.yaml 配置查询模型（SiliconFlow）。
global_search 无需嵌入；local_search 需要实体描述嵌入，首次调用时按需生成到
output/lancedb（之后复用）。
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent


class GraphRAGError(Exception):
    """GraphRAG 查询失败（配置 / 加载 / 调用 / 结果解析）。"""


class GraphRAGClient:
    """本地 GraphRAG 查询客户端。

    读取 ``root/out/`` 下的 parquet 索引，通过 ``graphrag.api`` 执行
    global / local 检索，返回答案文本。
    """

    def __init__(self, root: str | Path | None = None) -> None:
        from graphrag.config.load_config import load_config  # 惰性导入，未启用 GraphRAG 时无需安装该包

        self.root = Path(root).resolve() if root else DATA_DIR
        self._config = load_config(self.root)
        # load_config 会把 db_uri 按进程 CWD 解析；这里固定到 root 下，避免受运行目录影响
        self._config.vector_store.db_uri = str(
            (self.root / "output" / "lancedb").resolve()
        )
        # SiliconFlow 异步请求无默认超时会卡在 socket 等待，这里给所有模型补上超时
        for model in (
            *self._config.completion_models.values(),
            *self._config.embedding_models.values(),
        ):
            model.call_args.setdefault("timeout", 120)
        out = self.root / "out"
        self._entities = pd.read_parquet(out / "entities.parquet")
        self._communities = pd.read_parquet(out / "communities.parquet")
        self._community_reports = pd.read_parquet(out / "community_reports.parquet")
        self._text_units = pd.read_parquet(out / "text_units.parquet")
        self._relationships = pd.read_parquet(out / "relationships.parquet")

    def search(self, query: str, method: str = "global") -> str:
        method = method if method in ("local", "global") else "global"
        try:
            if method == "global":
                return asyncio.run(self._global_search(query))
            self._ensure_entity_embeddings()
            return asyncio.run(self._local_search(query))
        except Exception as exc:  # noqa: BLE001 — 统一包装成 GraphRAGError
            raise GraphRAGError(f"graphrag {method} search failed: {exc}") from exc

    async def _global_search(self, query: str) -> str:
        import graphrag.api as api

        resp, _ = await api.global_search(
            config=self._config,
            entities=self._entities,
            communities=self._communities,
            community_reports=self._community_reports,
            community_level=2,
            dynamic_community_selection=False,
            response_type="Multiple Paragraphs",
            query=query,
        )
        return str(resp)

    async def _local_search(self, query: str) -> str:
        import graphrag.api as api

        resp, _ = await api.local_search(
            config=self._config,
            entities=self._entities,
            communities=self._communities,
            community_reports=self._community_reports,
            text_units=self._text_units,
            relationships=self._relationships,
            covariates=None,
            community_level=2,
            response_type="Multiple Paragraphs",
            query=query,
        )
        return str(resp)

    def _ensure_entity_embeddings(self) -> None:
        """为 local_search 生成实体描述嵌入（幂等：向量库已有则跳过）。"""
        from graphrag.config.embeddings import entity_description_embedding
        from graphrag_llm.embedding import create_embedding
        from graphrag_vectors import VectorStoreDocument, create_vector_store

        embedder = create_embedding(
            self._config.get_embedding_model_config(
                self._config.local_search.embedding_model_id
            )
        )
        schema = self._config.vector_store.index_schema[entity_description_embedding]
        # 默认 schema vector_size=3072，但 bge-m3 实际输出 1024 维；以真实维度为准，否则建表会不匹配
        dim = len(embedder.embedding(input=["sample"]).first_embedding)
        schema.vector_size = dim
        self._config.vector_store.vector_size = dim

        store = create_vector_store(self._config.vector_store, schema)
        store.connect()
        # 表尚不存在时需要先建表（connect 只在表存在时挂上 collection）
        if getattr(store, "document_collection", None) is None:
            store.create_index()
        if store.count() >= len(self._entities):
            return
        for _, row in self._entities.iterrows():
            text = str(row.get("description") or row.get("title") or "").strip()
            if not text:
                continue
            try:
                vec = embedder.embedding(input=[text]).first_embedding
            except Exception:  # noqa: BLE001 — 单个实体嵌入失败不阻断整体
                continue
            if vec:
                store.insert(
                    VectorStoreDocument(
                        id=str(row["id"]), vector=vec, data={"text": text}
                    )
                )


def build_default_graphrag_client() -> GraphRAGClient | None:
    """按配置构建本地 GraphRAG 客户端；未启用时返回 None（工具回退到桩）。"""
    from ..config import get_settings

    s = get_settings()
    if not s.graphrag_enabled:
        return None
    # settings.yaml 用 ${SILICONFLOW_API_KEY} 占位，这里从 QAgent 配置注入环境变量
    os.environ.setdefault("SILICONFLOW_API_KEY", s.siliconflow_api_key or "")
    root = s.graphrag_root or None  # 空则用包内 src/qagent/data
    return GraphRAGClient(root=root)
