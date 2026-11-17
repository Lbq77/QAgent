"""工作流节点。

planner 真正解析 LLM 输出并落 plan + constraints；await_input 缺参时生成友好提问；
quant 使用 Planner 提取的因子（经 factors.resolve_factor 归一化）；其余节点为确定性桩，
产物用结构化 Pydantic 模型（ResearchOutput / QuantOutput / RiskReport / FactCheckResult）。

节点统一签名 `(state, ctx) -> dict`，返回要更新的字段子集。
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from pydantic import BaseModel

from ..agents.planner import build_task_plan
from ..agents.prompts import SYSTEM_PROMPTS
from ..backtest import load_result
from ..context import RuntimeContext
from ..factors import resolve_factor
from ..state import (
    AgentState,
    FactCheckResult,
    NodeStatus,
    QuantOutput,
    ResearchOutput,
    TaskStatus,
    ToolCallRecord,
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
        hint = f"需要解析指数「{index}」的成分股（接入），或请直接提供股票代码"
    elif "date_range" in missing:
        hint = "请问你希望比较哪一段时间？"
    else:
        hint = "请补充后继续。"
    return f"当前任务缺少：{'、'.join(labels)}。{hint}"


def _finish(state: AgentState, node: str, node_status: NodeStatus, **updates: Any) -> dict:
    """收尾：追加一条 trace（含该节点的工具调用明细），更新 current_step，合并其余字段。"""
    note = updates.pop("_note", None)
    error = updates.pop("_error", None)
    tool_calls = updates.pop("_tool_calls", [])
    rec = TraceRecord(
        node=node,
        status=node_status,
        ended_at=_now(),
        note=note,
        error=error,
        tool_calls=tool_calls,
    )
    return {"trace": [*state.trace, rec], "current_step": node, **updates}


def _summarize(value: Any, limit: int = 200) -> str:
    """把工具输入/输出压成一条短摘要，避免 State 膨胀（上下文外置原则）。"""
    if value is None:
        return ""
    if isinstance(value, BaseModel):
        value = value.model_dump()
    try:
        s = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        s = str(value)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _call_tool(
    ctx: RuntimeContext, agent: str, tool_name: str, raw: dict[str, Any]
) -> tuple[Any, ToolCallRecord]:
    """统一工具调用 + 计时 + 记录一条 tool trace。"""
    started = time.monotonic()
    out = ctx.tool_registry.invoke(agent, tool_name, raw)
    record = ToolCallRecord(
        tool=tool_name,
        agent=agent,
        status="succeeded",
        duration_ms=round((time.monotonic() - started) * 1000.0, 2),
        input_summary=_summarize(raw),
        output_summary=_summarize(out),
    )
    return out, record


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


def _extract_json(text: str) -> str:
    """从 LLM 输出抠出 JSON（去代码围栏 + 取第一个 { 到最后一个 }）。"""
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object")
    return text[start : end + 1]


def _plan_research_tools(ctx: RuntimeContext, state: AgentState) -> list[str]:
    """工具路由：让 LLM 只决定「调用哪几个研究工具」，参数一律由程序填充。

    注意：LLM 只输出工具名，不产出参数——symbols/日期/query 全部取自 task_constraints，
    绝不信任 LLM 编造的符号或日期。stub / 解析失败 / 空结果 → 回退「只拉行情」。
    """
    default = ["get_market_data"]
    if getattr(ctx.llm, "is_stub", False):
        return default
    constraints = {
        "universe": state.task_constraints.universe,
        "date_range": state.task_constraints.date_range,
        "market": state.task_constraints.market,
    }
    user = json.dumps({"query": state.user_query, "constraints": constraints}, ensure_ascii=False)
    try:
        raw = ctx.llm.invoke(SYSTEM_PROMPTS["research_route"], user)
        data = json.loads(_extract_json(raw))
        tools = data.get("tools") or []
    except Exception:
        return default
    allowed = {"get_market_data", "search_financial_knowledge", "get_company_profile"}
    names = [t for t in tools if isinstance(t, str) and t in allowed]
    return names or default


def _build_research_summary(state: AgentState, market_data, knowledge, profile) -> str:
    """汇总研究结果；知识库答案与公司概况详情一并保留，让报告能看到完整内容。"""
    lines: list[str] = []
    if market_data is not None:
        src = f"（来源 {market_data.source}）" if market_data.source else ""
        lines.append(f"行情：{len(market_data.records)} 条{src}")
    if knowledge is not None:
        answer = (knowledge.answer or "").strip()
        if len(answer) > 500:
            answer = answer[:500] + "…"
        lines.append(f"知识库检索：{answer}")
    if profile is not None:
        desc = " / ".join(x for x in (profile.name, profile.sector, profile.exchange) if x)
        lines.append(f"公司概况：{profile.symbol} {desc}".strip())
    body = "\n".join(lines) if lines else "无检索结果"
    return f"研究 {state.user_query[:40]}：\n{body}"


def research_node(state: AgentState, ctx: RuntimeContext) -> dict:
    # 工具路由：LLM 只选工具，参数全部由 task_constraints 填充（不信任 LLM 编造）。
    names = _plan_research_tools(ctx, state)
    tool_calls: list[ToolCallRecord] = []
    market_data = knowledge = profile = None

    for name in names:
        if name == "get_market_data":
            args = {
                "symbols": state.task_constraints.universe,
                "start": state.task_constraints.date_range.get("start", ""),
                "end": state.task_constraints.date_range.get("end", ""),
                "market": state.task_constraints.market,
            }
        elif name == "search_financial_knowledge":
            args = {"query": state.user_query, "mode": "global"}
        elif name == "get_company_profile":
            args = {"symbol": (state.task_constraints.universe or [""])[0]}

        out, tc = _call_tool(ctx, "research", name, args)
        tool_calls.append(tc)
        if name == "get_market_data":
            market_data = out
        elif name == "search_financial_knowledge":
            knowledge = out
        elif name == "get_company_profile":
            profile = out

    summary = _build_research_summary(state, market_data, knowledge, profile)
    evidence: list[str] = []
    if market_data is not None and market_data.source:
        evidence.append(market_data.source)
    if knowledge is not None:
        evidence.extend(m.url for m in knowledge.matches if m.url)
    snapshot_id = market_data.snapshot_id if market_data is not None else ""

    out = ResearchOutput(summary=summary, evidence_refs=evidence, snapshot_id=snapshot_id)
    return _finish(
        state,
        "research_node",
        NodeStatus.SUCCEEDED,
        _tool_calls=tool_calls,
        research_context=out.summary,
        evidence_refs=out.evidence_refs,
        snapshot_id=out.snapshot_id,
    )


def quant_node(state: AgentState, ctx: RuntimeContext) -> dict:
    # JSON 配置 → Schema 校验 → 确定性因子/回测执行（run_backtest 内部完成选股）。
    # 注意：不再 `or [...]` 静默替换默认股票——空股票池应在上游 await_input 被拦截。
    # 若 state.strategy_config 已存在（人类修改后回路重跑），直接用修改后的配置重新回测。
    from ..tools.schemas import StrategyConfig

    existing = state.strategy_config
    if existing:
        config = StrategyConfig.model_validate(existing)
    else:
        universe = state.task_constraints.universe
        ranking = resolve_factor(state.task_plan.factor, state.task_plan.factor_direction)
        config = StrategyConfig(
            market=state.task_constraints.market,
            universe=universe,
            ranking=ranking,
            top_n=state.task_plan.top_n,
            cost=0.001,  # 交易成本不为 0
            t_plus_1=True,  # t 日信号，t+1 日执行
        )
    start = state.task_constraints.date_range.get("start", "")
    end = state.task_constraints.date_range.get("end", "")
    bt_out, tc = _call_tool(
        ctx,
        "quant",
        "run_backtest",
        {
            "strategy_config": config,
            "start": start,
            "end": end,
            "snapshot_id": state.snapshot_id,  # 复用 research 的行情快照，避免重复拉取
        },
    )
    out = QuantOutput(
        strategy_config=config.model_dump(),
        backtest_result_ref=bt_out.ref,
    )
    return _finish(
        state,
        "quant_node",
        NodeStatus.SUCCEEDED,
        _tool_calls=[tc],
        strategy_config=out.strategy_config,
        backtest_result_ref=out.backtest_result_ref,
    )


def human_confirm_node(state: AgentState, ctx: RuntimeContext) -> dict:
    # LangGraph interrupt 挂起，等待用户 approve / modify / reject。
    # 注意：resume 时该节点会从头重跑，interrupt() 此时返回恢复值（不再挂起）。
    from langgraph.types import interrupt
    from pydantic import ValidationError

    from ..tools.schemas import StrategyConfig

    decision = interrupt(
        {
            "type": "human_confirm",
            "question": "请确认策略配置，或修改后重跑（approve / modify / reject）",
            "strategy_config": state.strategy_config,
            "backtest_result_ref": state.backtest_result_ref,
            "allowed_actions": ["approve", "modify", "reject"],
        }
    )

    action = (decision or {}).get("action", "approve")
    reason = (decision or {}).get("reason", "")

    if action == "reject":
        return _finish(
            state,
            "human_confirm_node",
            NodeStatus.SUCCEEDED,
            confirmation={"action": "reject", "reason": reason},
            awaiting_confirmation=False,
            status=TaskStatus.CANCELLED,
            _note="human rejected the strategy",
        )

    if action == "modify":
        # 在现有配置上叠加修改字段，再经 Schema 校验；非法修改视为无效并退回 approve 原配置。
        merged = dict(state.strategy_config)
        merged.update((decision or {}).get("config") or {})
        try:
            config = StrategyConfig.model_validate(merged)
        except ValidationError as exc:
            return _finish(
                state,
                "human_confirm_node",
                NodeStatus.SUCCEEDED,
                confirmation={"action": "approve", "config": state.strategy_config},
                awaiting_confirmation=False,
                errors=[*state.errors, {"node": "human_confirm_node", "msg": f"invalid modify: {exc}"}],
                _note="invalid modify, kept original config",
            )
        return _finish(
            state,
            "human_confirm_node",
            NodeStatus.SUCCEEDED,
            confirmation={"action": "modify", "config": config.model_dump()},
            awaiting_confirmation=False,
            strategy_config=config.model_dump(),
            _note="human modified the strategy",
        )

    # approve
    return _finish(
        state,
        "human_confirm_node",
        NodeStatus.SUCCEEDED,
        confirmation={"action": "approve", "config": state.strategy_config},
        awaiting_confirmation=False,
        _note="human approved the strategy",
    )


def risk_node(state: AgentState, ctx: RuntimeContext) -> dict:
    # 经统一工具系统调 audit_strategy，读取策略配置与回测执行结果做真实审计。
    report, tc = _call_tool(
        ctx,
        "risk_auditor",
        "audit_strategy",
        {
            "strategy_config": state.strategy_config,
            "backtest_result_ref": state.backtest_result_ref,
        },
    )
    return _finish(state, "risk_node", NodeStatus.SUCCEEDED, _tool_calls=[tc], risk_report=report)


def fact_check_node(state: AgentState, ctx: RuntimeContext) -> dict:
    # 数字核查 + 证据链：报告数字必须能追溯到工具产物，逐一比对。
    from ..tools.schemas import Claim

    mismatches: list[str] = []
    tool_calls: list[ToolCallRecord] = []
    claims: list[Claim] = []

    # 1) 证据链：backtest 任务必须有可解析的回测引用，research 任务必须有行情证据
    if state.task_plan.needs_backtest:
        if not state.backtest_result_ref:
            mismatches.append("缺少回测结果引用（backtest_result_ref 为空）")
        elif load_result(state.backtest_result_ref) is None:
            mismatches.append(f"回测结果引用无法解析：{state.backtest_result_ref}")
    if state.task_plan.needs_research and not state.evidence_refs:
        mismatches.append("研究任务缺少证据引用（evidence_refs 为空）")

    # 2) 数字核查：回测三大指标 vs 结果存储（经 validate_report_claims 独立再读）
    if state.backtest_result_ref:
        bt, tc_get = _call_tool(
            ctx, "quant", "get_backtest_result", {"ref": state.backtest_result_ref}
        )
        tool_calls.append(tc_get)
        m = bt.metrics
        claims = [
            Claim(field="annual_return", value=m.annual_return, source_ref=state.backtest_result_ref),
            Claim(field="max_drawdown", value=m.max_drawdown, source_ref=state.backtest_result_ref),
            Claim(field="sharpe", value=m.sharpe, source_ref=state.backtest_result_ref),
        ]
        verified, tc_verify = _call_tool(
            ctx,
            "risk_auditor",
            "validate_report_claims",
            {
                "claims": [c.model_dump() for c in claims],
                "backtest_result_ref": state.backtest_result_ref,
            },
        )
        tool_calls.append(tc_verify)
        mismatches.extend(verified.mismatches)

    # 3) 一致性：风险等级与策略配置交叉核对（low 必须是「无违规」的配置）
    cfg = state.strategy_config
    if cfg and state.risk_report.level == "low":
        if cfg.get("lookback_days", 0) <= 0:
            mismatches.append("风险等级为 low 但 lookback_days<=0")
        if cfg.get("cost", 0) == 0:
            mismatches.append("风险等级为 low 但 cost=0")
        if not cfg.get("t_plus_1", True):
            mismatches.append("风险等级为 low 但未启用 t+1")

    result = FactCheckResult(claims_checked=len(claims), mismatches=mismatches)
    return _finish(
        state,
        "fact_check_node",
        NodeStatus.SUCCEEDED,
        _tool_calls=tool_calls,
        fact_check_result=result,
    )


def _deterministic_summary(facts: dict) -> str:
    """stub / LLM 失败时的确定性总结兜底（数字仍来自工具产物）。"""
    m = facts.get("metrics") or {}
    level = facts.get("risk_level", "unknown")
    findings = facts.get("risk_findings") or []
    if m:
        perf = (
            f"年化收益 {m.get('annual_return', 0):.2%}、"
            f"最大回撤 {m.get('max_drawdown', 0):.2%}、"
            f"夏普 {m.get('sharpe', 0):.2f}"
        )
    else:
        perf = "无回测结果"
    risk = f"风险等级 {level}" + (f"（{len(findings)} 项发现）" if findings else "")
    return (
        f"策略在回测区间内 {perf}，{risk}。"
        "请结合风险提示与自身目标审慎决策。本内容仅供研究学习，不构成投资建议。"
    )


def _generate_summary(ctx: RuntimeContext, facts: dict) -> str:
    """调用 LLM 生成总结与投资建议；stub / 失败 / 返回 JSON 时回退确定性模板。"""
    if getattr(ctx.llm, "is_stub", False):
        return _deterministic_summary(facts)
    try:
        out = (ctx.llm.invoke(SYSTEM_PROMPTS["report"], json.dumps(facts, ensure_ascii=False)) or "").strip()
    except Exception:
        return _deterministic_summary(facts)
    if not out or out.startswith("{"):
        return _deterministic_summary(facts)
    return out


def _verify_summary_claims(summary: str, facts: dict) -> tuple[list[str], int]:
    """从总结里抽出所有百分比 + 夏普数字，逐个与真实指标比对。

    比固定正则更稳：无论「年化收益为 45%」「45% 的年化收益」「年化收益率达到45%」都能抓到。
    任何百分比只要对不上任意一个真实值（三指标 + 成本 + 三档成本场景），就算不符。
    返回 (不符列表, 实际核查条数)。
    """
    metrics = facts.get("metrics") or {}
    if not metrics:
        return [], 0

    # 可接受的真实值集合：三个指标 + 交易成本 + 三档成本场景里的百分比指标
    acceptable: list[float] = []
    for k in ("annual_return", "total_return", "max_drawdown"):
        v = metrics.get(k)
        if v is not None:
            acceptable.append(v)
    cfg = facts.get("strategy") or {}
    if cfg.get("cost") is not None:
        acceptable.append(cfg["cost"])
    for scen in (facts.get("cost_scenarios") or {}).values():
        for k in ("annual_return", "total_return", "max_drawdown"):
            v = scen.get(k)
            if v is not None:
                acceptable.append(v)

    mismatches: list[str] = []
    pct_claims = [float(m.group(1)) / 100.0 for m in re.finditer(r"(-?\d+(?:\.\d+)?)\s*%", summary)]
    checked = len(pct_claims)
    for pct in pct_claims:
        if not any(abs(pct - v) <= 0.01 for v in acceptable):
            mismatches.append(f"百分比 {pct:.2%} 与真实指标不符")

    # 夏普（比率，非百分比）：只认「夏普」附近的数字
    m = re.search(r"夏普[^\d-]*(-?\d+(?:\.\d+)?)", summary)
    if m and metrics.get("sharpe") is not None:
        checked += 1
        claimed = float(m.group(1))
        if abs(claimed - metrics["sharpe"]) > 0.5:
            mismatches.append(f"夏普：总结声称 {claimed:.2f}，实际 {metrics['sharpe']:.2f}")
    return mismatches, checked


def report_node(state: AgentState, ctx: RuntimeContext) -> dict:
    # 四角色串联——把 planner/research/quant/risk 的产物汇总成一份可读报告。
    # 数字一律来自工具产物（回测结果、风险等级），LLM 只负责用自然语言总结，不编数字。
    tool_calls: list[ToolCallRecord] = []
    metrics_text = "(无回测)"
    facts: dict[str, Any] = {
        "goal": state.task_plan.goal or "(未指定)",
        "strategy": state.strategy_config,
        "risk_level": state.risk_report.level,
        "risk_findings": state.risk_report.findings,
        "fact_check": state.fact_check_result.model_dump(),
    }
    if state.backtest_result_ref:
        bt, tc = _call_tool(
            ctx, "quant", "get_backtest_result", {"ref": state.backtest_result_ref}
        )
        tool_calls.append(tc)
        m = bt.metrics
        metrics_text = (
            f"年化 {m.annual_return:.2%} / 最大回撤 {m.max_drawdown:.2%} / 夏普 {m.sharpe:.2f}"
        )
        facts["metrics"] = m.model_dump()
        payload = load_result(state.backtest_result_ref)
        if payload:
            facts["selected"] = payload.get("selected", [])
            facts["cost_scenarios"] = payload.get("cost_scenarios", {})

    summary = _generate_summary(ctx, facts)
    summary_mismatches, summary_checked = _verify_summary_claims(summary, facts)
    if summary_mismatches:
        # 总结里出现了对不上的数字 → 弃用该总结，改用确定性模板（保证报告数字正确）
        summary = _deterministic_summary(facts)
    fact_check_result = FactCheckResult(
        claims_checked=state.fact_check_result.claims_checked + summary_checked,
        mismatches=state.fact_check_result.mismatches + summary_mismatches,
    )

    lines = [
        f"# 研究结论",
        f"目标：{state.task_plan.goal or '(未指定)'}",
        f"研究摘要：{state.research_context or '(无)'}",
        f"策略配置：{json.dumps(state.strategy_config, ensure_ascii=False)}",
        f"回测指标：{metrics_text}",
        f"风险等级：{state.risk_report.level}",
        f"证据链：{state.backtest_result_ref or '-'} | {state.snapshot_id or '-'} | {', '.join(state.evidence_refs) or '-'}",
    ]
    if state.risk_report.findings:
        lines.append("风险发现：" + "；".join(state.risk_report.findings))
    if fact_check_result.mismatches:
        lines.append("事实核查不符：" + "；".join(fact_check_result.mismatches))
    lines.append("")
    lines.append("## 分析师总结与建议")
    lines.append(summary)

    report = "\n".join(lines)
    return _finish(
        state,
        "report_node",
        NodeStatus.SUCCEEDED,
        _tool_calls=tool_calls,
        status=TaskStatus.SUCCEEDED,
        final_report=report,
        fact_check_result=fact_check_result,
        artifact_refs={**state.artifact_refs, "report": "artifact://report_final"},
    )


def error_handler_node(state: AgentState, ctx: RuntimeContext) -> dict:
    # 上游失败 → 下游 block；错误显式记录，不把失败伪装成成功（/）。
    return _finish(
        state,
        "error_handler_node",
        NodeStatus.SUCCEEDED,
        status=TaskStatus.FAILED,
        errors=[*state.errors, {"node": state.current_step, "msg": "routed to error handler"}],
        _note=f"error from {state.current_step}",
    )
