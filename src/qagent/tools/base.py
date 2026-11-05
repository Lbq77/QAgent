"""工具抽象与校验错误类型（Day 3：统一工具系统）。

每个工具用 Pydantic 定义输入/输出 Schema，`call()` 在边界做校验：
- 输入非法 → ToolInputError（不进入 execute）
- 输出非预期形状 → ToolOutputError
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError


class ToolError(Exception):
    """所有工具错误的基类。"""


class ToolInputError(ToolError):
    """工具输入未通过 Schema 校验。"""

    def __init__(self, tool: str, detail: str) -> None:
        self.tool = tool
        self.detail = detail
        super().__init__(f"[{tool}] invalid input: {detail}")


class ToolOutputError(ToolError):
    """工具输出未通过 Schema 校验。"""

    def __init__(self, tool: str, detail: str) -> None:
        self.tool = tool
        self.detail = detail
        super().__init__(f"[{tool}] invalid output: {detail}")


class Tool(ABC):
    """统一工具基类：name/description/Schema/权限 + execute 钩子。

    子类只需实现 `execute(args)` 返回 output_model 实例（或可校验的 dict），
    输入/输出校验由 `call()` 统一完成。
    """

    name: ClassVar[str]
    description: ClassVar[str]
    input_model: ClassVar[type[BaseModel]]
    output_model: ClassVar[type[BaseModel]]
    permission: ClassVar[str] = "read-only"

    def call(self, raw: dict[str, Any]) -> BaseModel:
        """校验输入 → 执行 → 校验输出，返回 output_model 实例。"""
        try:
            args = self.input_model.model_validate(raw)
        except ValidationError as exc:
            raise ToolInputError(self.name, str(exc)) from exc

        out = self.execute(args)
        if isinstance(out, self.output_model):
            return out
        try:
            return self.output_model.model_validate(out)
        except ValidationError as exc:
            raise ToolOutputError(self.name, str(exc)) from exc

    @abstractmethod
    def execute(self, args: BaseModel) -> Any:
        """具体工具逻辑；返回 output_model 实例或可校验的 dict。"""
