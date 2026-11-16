"""GraphRAG 本地客户端与工具接入的单元测试（不发起真实 LLM 请求）。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from qagent.data import GraphRAGClient, build_default_graphrag_client
from qagent.tools.research import SearchFinancialKnowledge

DATA_DIR = Path(__file__).resolve().parents[1] / "src" / "qagent" / "data"


def test_client_loads_index_parquet(monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")
    client = GraphRAGClient(root=DATA_DIR)
    assert isinstance(client._entities, pd.DataFrame)
    assert "title" in client._entities.columns
    assert len(client._entities) == 35
    assert len(client._relationships) == 45
    assert len(client._communities) == 9


def test_build_default_client_disabled(monkeypatch):
    import qagent.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "get_settings",
        lambda: config_mod.Settings(graphrag_enabled=False),
    )
    assert build_default_graphrag_client() is None


class _FakeClient:
    def search(self, query: str, method: str = "global") -> str:
        return f"Fake({method}): {query}"


def test_search_financial_knowledge_uses_client():
    tool = SearchFinancialKnowledge(client=_FakeClient())  # type: ignore[arg-type]
    out = tool.call({"query": "SpaceX 主营", "mode": "local"})
    assert out.answer == "Fake(local): SpaceX 主营"
    assert out.matches[0].url == "graphrag://local"


def test_search_financial_knowledge_stub_without_client(monkeypatch):
    # 未启用 GraphRAG 时 build_default_graphrag_client 返回 None → 工具回退为桩
    from qagent.tools import research

    monkeypatch.setattr(research, "build_default_graphrag_client", lambda: None)
    tool = SearchFinancialKnowledge(client=None)  # type: ignore[arg-type]
    out = tool.call({"query": "SpaceX 主营", "mode": "global"})
    assert out.answer.startswith("[stub]")
