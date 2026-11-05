"""工作流节点（PRD §6.1）。

Day 2：planner 真正解析 LLM 输出并落 plan + constraints；await_input 缺参时生成友好提问；
quant 使用 Planner 提取的因子（经 factors.resolve_factor 归一化）；其余节点为确定性桩，
产物用结构化 Pydantic 模型（ResearchOutput / QuantOutput / RiskReport / FactCheckResult）。

节点统一签名 `(state, ctx) -> dict`，返回要更新的字段子集。
"""
from __future__ import annotations

from typing import Any

from ..agents.planner import build_task_plan
from ..context import RuntimeContext
from ..factors import resolve_factor
from ..state import (
    AgentState,
    FactCheckResult,
    NodeStatus,
    QuantOutput,
    ResearchOutput,
    RiskReport,
    TaskStatus,
    TraceRecord,
    _now,
)

_PARAM_LABELS = {
    "date_range": "回测时间范围",
    "start_date": "开始日期",
    "end_date": "结束日期",
    "universe": "股票池",
    "market": "市场",
    "factor": "因子",
}


def _build_question(missing: list[str], index: str = "", factor: str = "") -> str:
    # 未知因子是「不支持」而非「缺少」，单独成句
    if "factor" in missing and factor:
        return (
            f"因子「{factor}」暂不支持，可选：momentum_5d、momentum_20d、momentum_60d、"
            "volatility_20d、avg_volume_20d、market_cap"
        )
    labels = [_PARAM_LABELS.get(p, p) for p in missing]
    if "universe" in missing and index:
        hint = f"需要解析指数「{index}」的成分股（Day 4 接入），或请直接提供股票代码"
    elif "date_range" in missing:
        hint = "请问你希望比较哪一段时间？"
    else:
        hint = "请补充后继续。"
    return f"当前任务缺少：{'、'.join(labels)}。{hint}"


def _finish(state: AgentState, node: str, node_status: NodeStatus, **updates: Any) -> dict:
    """收尾：追加一条 trace，更新 current_step，并合并其余字段更新。"""
    note = updates.pop("_note", None)
    error = updates.pop("_error", None)
    rec = TraceRecord(node=node, status=node_status, ended_at=_now(), note=note, error=error)
    return {"trace": [*state.trace, rec], "current_step": node, **updates}


def planner_node(state: AgentState, ctx: RuntimeContext) -> dict:
    plan, constraints, parse_err = build_task_plan(ctx, state.user_query)

    errors = list(state.errors)
    note = None
    if parse_err:
        errors.append({"node": "planner_node", "msg": parse_err})
        note = parse_err

    return _finish(
        state,
        "planner_node",
        NodeStatus.SUCCEEDED,
        task_plan=plan,
        task_constraints=constraints,
        status=TaskStatus.RUNNING,
        errors=errors,
        _note=note,
    )


def await_input_node(state: AgentState, ctx: RuntimeContext) -> dict:
    """缺少必要参数时进入：标记 blocked，生成一句友好提问，等待用户补充后 resume。"""
    missing = state.task_plan.missing_params
    question = _build_question(missing, state.task_constraints.index, state.task_plan.factor)
    return _finish(
        state,
        "await_input_node",
        NodeStatus.SUCCEEDED,
        status=TaskStatus.BLOCKED,
        pending_question=question,
        errors=[*state.errors, {"node": "await_input_node", "msg": f"missing params: {missing}"}],
        _note=question,
    )


def research_node(state: AgentState, ctx: RuntimeContext) -> dict:
    # Day 3：经统一工具系统调 get_market_data（确定性桩）；Day 4 换真实行情/GraphRAG。
    market_data = ctx.tool_registry.invoke(
        "research",
        "get_market_data",
        {
            "symbols": state.task_constraints.universe,
            "start": state.task_constraints.date_range.get("start", ""),
            "end": state.task_constraints.date_range.get("end", ""),
            "market": state.task_constraints.market,
        },
    )
    out = ResearchOutput(
        summary=f"已检索行情与金融资料：{state.user_query[:60]}（{len(market_data.records)} 条行情）",
        evidence_refs=[market_data.source] if market_data.source else [],
        snapshot_id=market_data.snapshot_id,
    )
    return _finish(
        state,
        "research_node",
        NodeStatus.SUCCEEDED,
        research_context=out.summary,
        evidence_refs=out.evidence_refs,
        snapshot_id=out.snapshot_id,
    )


def quant_node(state: AgentState, ctx: RuntimeContext) -> dict:
    # Day 5–6 替换为 JSON 配置 → Schema 校验 → 因子/回测确定性执行。
    # 注意：不再 `or [...]` 静默替换默认股票——空股票池应在上游 await_input 被拦截。
    universe = state.task_constraints.universe
    ranking = resolve_factor(state.task_plan.factor, state.task_plan.factor_direction)
    out = QuantOutput(
        strategy_config={
            "market": state.task_constraints.market,
            "universe": universe,
            "ranking": ranking,
            "top_n": state.task_plan.top_n,
            "cost": 0.001,  # 交易成本不为 0（PRD §9.8）
            "t_plus_1": True,  # t 日信号，t+1 日执行
        },
        backtest_result_ref="backtest_stub",
    )
    return _finish(
        state,
        "quant_node",
        NodeStatus.SUCCEEDED,
        strategy_config=out.strategy_config,
        backtest_result_ref=out.backtest_result_ref,
    )


def human_confirm_node(state: AgentState, ctx: RuntimeContext) -> dict:
    # Day 1 自动通过桩；Day 7 换成 LangGraph interrupt 挂起等待用户确认/修改/拒绝。
    return _finish(
        state,
        "human_confirm_node",
        NodeStatus.SUCCEEDED,
        confirmation={"action": "approve", "config": state.strategy_config},
        awaiting_confirmation=False,
        _note="auto-approved stub (Day 7: interrupt)",
    )


def risk_node(state: AgentState, ctx: RuntimeContext) -> dict:
    # Day 6/8 接入 audit_strategy / validate_report_claims。
    report = RiskReport(
        level="low",
        findings=[],
        survivorship_bias=True,  # 固定股票池存在幸存者偏差，必须标记（PRD §9.5）
        cost_check="t+1 confirmed, cost != 0",
    )
    return _finish(state, "risk_node", NodeStatus.SUCCEEDED, risk_report=report)


def fact_check_node(state: AgentState, ctx: RuntimeContext) -> dict:
    # Day 11 接入数字核查：报告数字 vs 工具产物逐一比对。
    result = FactCheckResult(claims_checked=0, mismatches=[])
    return _finish(state, "fact_check_node", NodeStatus.SUCCEEDED, fact_check_result=result)


def report_node(state: AgentState, ctx: RuntimeContext) -> dict:
    # Day 2+ 汇总各节点产物生成报告，这里只落一个产物引用。
    return _finish(
        state,
        "report_node",
        NodeStatus.SUCCEEDED,
        status=TaskStatus.SUCCEEDED,
        artifact_refs={"report": "artifact://report_stub"},
    )


def error_handler_node(state: AgentState, ctx: RuntimeContext) -> dict:
    # 上游失败 → 下游 block；错误显式记录，不把失败伪装成成功（PRD §2/§6.4）。
    return _finish(
        state,
        "error_handler_node",
        NodeStatus.SUCCEEDED,
        status=TaskStatus.FAILED,
        errors=[*state.errors, {"node": state.current_step, "msg": "routed to error handler"}],
        _note=f"error from {state.current_step}",
    )
