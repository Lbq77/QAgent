"""统一工具系统（Day 3）：工具抽象、注册表、Schema 与 9 个工具定义。

- Tool：name/description/input_model/output_model/permission + call()（边界校验）
- ToolRegistry：register/resolve/invoke（含权限白名单）
- schemas：各工具输入/输出 Pydantic 模型 + StrategyConfig
- research/quant/risk：9 个工具（Day 3 确定性桩，Day 4–6 换真实现）
"""
from .base import Tool, ToolError, ToolInputError, ToolOutputError
from .registry import (
    ToolNotFoundError,
    ToolPermissionError,
    ToolRegistry,
    build_default_registry,
    ensure_agent_allowed,
)

__all__ = [
    "Tool",
    "ToolError",
    "ToolInputError",
    "ToolOutputError",
    "ToolNotFoundError",
    "ToolPermissionError",
    "ToolRegistry",
    "build_default_registry",
    "ensure_agent_allowed",
]
