"""条件路由（PRD §6.2）。

每个路由函数读 State 返回下一个节点的键名；图构建时把键名映射到具体节点。
Day 2 已支持：缺参数 → await_input、按需跳过 research/quant、拒绝 → 结束、失败 → error。
"""
from __future__ import annotations

from ..state import AgentState, TaskStatus


def route_after_planner(state: AgentState) -> str:
    if state.status == TaskStatus.FAILED:
        return "error"
    if state.task_plan.missing_params:
        return "await_input"
    if state.task_plan.needs_research:
        return "research"
    if state.task_plan.needs_backtest:
        return "quant"
    return "report"


def route_after_research(state: AgentState) -> str:
    if state.status == TaskStatus.FAILED:
        return "error"
    if state.task_plan.needs_backtest:
        return "quant"
    return "report"


def route_after_confirm(state: AgentState) -> str:
    if state.confirmation.get("action") == "reject":
        return "end"  # 用户拒绝 → 任务取消，直接结束
    return "risk"


def route_after_risk(state: AgentState) -> str:
    if state.status == TaskStatus.FAILED:
        return "error"
    return "fact_check"
