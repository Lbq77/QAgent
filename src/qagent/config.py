"""全局配置（pydantic-settings）。

只承载模型与存储配置；Harness 的迭代/超时/预算会在 正式接入。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="QAGENT_", env_file=".env", extra="ignore"
    )

    # 模型调用：stub 离线桩 / openai / siliconflow（工厂切换）
    model_provider: str = "stub"
    model_name: str = "gpt-4o-mini"  # openai provider 使用的模型名

    # OpenAI（及任意 OpenAI 兼容接口）
    openai_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("QAGENT_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    openai_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("QAGENT_OPENAI_BASE_URL", "OPENAI_BASE_URL"),
    )

    # 硅基流动（SiliconFlow，OpenAI 兼容接口）
    siliconflow_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("QAGENT_SILICONFLOW_API_KEY", "SILICONFLOW_API_KEY"),
    )
    siliconflow_base_url: str = Field(
        default="https://api.siliconflow.cn/v1",
        validation_alias=AliasChoices("QAGENT_SILICONFLOW_BASE_URL", "SILICONFLOW_BASE_URL"),
    )
    siliconflow_model: str = Field(
        default="deepseek-ai/DeepSeek-V3",
        validation_alias=AliasChoices("QAGENT_SILICONFLOW_MODEL", "SILICONFLOW_MODEL"),
    )

    # Harness 预算 / 超时 / 重试（落地）
    max_iterations: int = 10
    max_tool_calls: int = 30
    tool_timeout_seconds: float = 30.0
    tool_retries: int = 1
    tool_retry_backoff_seconds: float = 0.5

    # Checkpoint 后端：memory（单进程/测试）或 sqlite（持久化断点恢复）
    checkpoint_backend: str = "sqlite"

    # 存储
    data_dir: Path = Path("./data")
    artifact_dir: Path = Path("./artifacts")
    checkpoint_db: Path = Path("./data/checkpoints.sqlite")
    results_db: Path = Path("./data/results.sqlite")  # 回测结果/实验记录（外置）

    # GraphRAG 知识库（本地 graphrag.api 查询；不启用时 search_financial_knowledge 用桩）
    graphrag_enabled: bool = False
    graphrag_root: str = ""  # 本地 graphrag 根目录（settings.yaml + out/ 所在）；空则用包内 src/qagent/data

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
