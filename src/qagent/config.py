"""全局配置（pydantic-settings）。

Day 1 只承载模型与存储配置；Harness 的迭代/超时/预算会在 Day 8 正式接入。
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

    # Harness 预算（占位，Day 8 落地）
    max_iterations: int = 10
    max_tool_calls: int = 30
    tool_timeout_seconds: float = 30.0

    # 存储
    data_dir: Path = Path("./data")
    artifact_dir: Path = Path("./artifacts")
    checkpoint_db: Path = Path("./data/checkpoints.sqlite")

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
