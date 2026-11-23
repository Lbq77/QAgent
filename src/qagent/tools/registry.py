"""工具注册表 + 权限白名单强制。

角色工具白名单来自 agents.AGENT_DEFINITIONS；这里提供：
- ToolRegistry：register / resolve / 重复检测
- ensure_agent_allowed：调用工具前校验角色权限
- invoke：白名单校验 + 解析 + 执行 一步到位（节点用这个入口）
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ..agents import AGENT_DEFINITIONS
from .base import Tool

if TYPE_CHECKING:
    from ..data import MarketDataService


class ToolNotFoundError(KeyError):
    """resolve 到未注册的工具。"""


class ToolPermissionError(PermissionError):
    """角色调用了白名单之外的工具。"""


class ToolRegistry:
    def __init__(self, harness=None) -> None:
        self._tools: dict[str, Tool] = {}
        self.harness = harness  # 可选：Harness（预算/超时/重试）

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def resolve(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFoundError(f"unknown tool '{name}'") from None

    def invoke(self, agent: str, tool_name: str, raw: dict[str, Any]) -> BaseModel:
        """节点统一入口：先查白名单，再解析工具并执行（含 Schema 校验）。

        有 Harness 时经它执行（预算/超时/重试），否则直接调用（测试/无超时场景）。
        """
        ensure_agent_allowed(agent, tool_name)
        tool = self.resolve(tool_name)
        if self.harness is not None:
            return self.harness.call(tool.call, raw)
        return tool.call(raw)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


def ensure_agent_allowed(agent: str, tool_name: str) -> None:
    """校验 agent 是否在 AGENT_DEFINITIONS 白名单里被允许调用 tool_name。"""
    allowed = AGENT_DEFINITIONS.get(agent, {}).get("tools", [])
    if tool_name not in allowed:
        raise ToolPermissionError(
            f"agent '{agent}' is not allowed to call tool '{tool_name}'"
        )


def build_default_registry(
    market_data: "MarketDataService | None" = None, harness=None
) -> ToolRegistry:
    """把 定义的全部 10 个工具注册进一个新 registry。

    market_data：注入行情数据服务（测试/离线传 Stub 组成的 service）；
    为 None 时 GetMarketData 内部用默认真实源（Yahoo）。
    harness：可选 Harness（预算/超时/重试），注入后所有 invoke 都经它执行。
    """
    from . import quant, research, risk

    reg = ToolRegistry(harness=harness)
    for tool in [
        research.GetMarketData(service=market_data),
        research.SearchFinancialKnowledge(),
        research.GetCompanyProfile(),
        research.SearchNews(),
        quant.CalculateFactor(service=market_data),
        quant.ScreenStocks(service=market_data),
        quant.RunBacktest(service=market_data),
        quant.GetBacktestResult(),
        risk.AuditStrategy(),
        risk.ValidateReportClaims(),
    ]:
        reg.register(tool)
    return reg
