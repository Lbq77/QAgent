"""条件路由。

每个路由函数读 State 返回下一个节点的键名；图构建时把键名映射到具体节点。
已支持：缺参数 → await_input、按需跳过 research/quant、拒绝 → 结束、失败 → error。
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
    if state.status == TaskStatus.FAILED:
        return "error"
    action = state.confirmation.get("action")
    if action == "reject":
        return "end"  # 用户拒绝 → 任务取消，直接结束
    if action == "modify":
        return "quant"  # 用户修改配置 → 回 quant 重新回测，再回到 human_confirm 确认
    return "risk"


def route_after_risk(state: AgentState) -> str:
    if state.status == TaskStatus.FAILED:
        return "error"
    return "fact_check"


def route_after_quant(state: AgentState) -> str:
    """quant 之后本走 human_confirm；若 quant 抛错（status=FAILED）则去 error。"""
    return "error" if state.status == TaskStatus.FAILED else "human_confirm"


def route_after_fact_check(state: AgentState) -> str:
    """fact_check 之后本走 report；失败则去 error。"""
    return "error" if state.status == TaskStatus.FAILED else "report"


def route_after_report(state: AgentState) -> str:
    """report 之后本结束；失败则去 error。"""
    return "error" if state.status == TaskStatus.FAILED else "end"
