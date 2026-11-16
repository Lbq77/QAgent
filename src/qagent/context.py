"""运行上下文：把配置与模型绑定在一起，注入到每个节点。

节点签名统一为 `(state: AgentState, ctx: RuntimeContext) -> dict`，
由 build_graph 在 add_node 时用 lambda 绑定，避免全局单例。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import Settings, get_settings
from .harness import build_harness
from .llm import LLMProvider, get_llm
from .store import configure_result_store
from .tools import ToolRegistry, build_default_registry


@dataclass
class RuntimeContext:
    settings: Settings
    llm: LLMProvider
    tool_registry: ToolRegistry = field(default_factory=build_default_registry)


def build_context(settings: Settings | None = None) -> RuntimeContext:
    settings = settings or get_settings()
    settings.ensure_dirs()
    # 生产路径：回测结果外置到 SQLite（跨进程可复现）+ 工具调用套 Harness
    configure_result_store(settings.results_db)
    registry = build_default_registry(harness=build_harness(settings))
    return RuntimeContext(settings=settings, llm=get_llm(settings), tool_registry=registry)
