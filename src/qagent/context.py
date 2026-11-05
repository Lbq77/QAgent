"""运行上下文：把配置与模型绑定在一起，注入到每个节点。

节点签名统一为 `(state: AgentState, ctx: RuntimeContext) -> dict`，
由 build_graph 在 add_node 时用 lambda 绑定，避免全局单例。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import Settings, get_settings
from .llm import LLMProvider, get_llm
from .tools import ToolRegistry, build_default_registry


@dataclass
class RuntimeContext:
    settings: Settings
    llm: LLMProvider
    tool_registry: ToolRegistry = field(default_factory=build_default_registry)


def build_context(settings: Settings | None = None) -> RuntimeContext:
    settings = settings or get_settings()
    settings.ensure_dirs()
    return RuntimeContext(settings=settings, llm=get_llm(settings))
