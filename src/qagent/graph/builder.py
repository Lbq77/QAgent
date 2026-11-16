"""LangGraph StateGraph 构建。

流程（支持按需跳过；支持人工确认回路；节点抛错兜底到 error_handler）：
    START → planner →(条件)→ research →(条件)→ quant →(条件)→ human_confirm
         →(条件)→ risk → fact_check →(条件)→ report →(条件)→ END
    planner 缺参数 → await_input → END（blocked）
    失败/拒绝 → error_handler → END

每个节点用 `_guarded` 包裹：节点内部抛出的异常（工具超时/输入非法等）不再崩掉整条
流程，而是落 status=FAILED + errors，由各条件路由统一导向 error_handler。

checkpointer 由配置决定（memory 或 SQLite 持久化），见 checkpoint.build_checkpointer。
"""
from __future__ import annotations

from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph

from ..checkpoint import build_checkpointer
from ..context import RuntimeContext
from ..state import AgentState, TaskStatus
from . import nodes
from . import routing


def _guarded(node_name: str, fn):
    """包裹节点：异常 → 返回 FAILED 状态（含错误记录），交由路由导向 error_handler。

    注意：`interrupt()` 内部抛 GraphInterrupt，这是 LangGraph 的挂起机制，必须原样
    向上抛，不能当成普通异常吞掉，否则人工确认会失效。
    """

    def wrapper(state):
        try:
            return fn(state)
        except GraphInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 — 统一转 FAILED，不伪装成功也不崩流程
            return {
                "status": TaskStatus.FAILED,
                "errors": [
                    *state.errors,
                    {"node": node_name, "msg": f"{type(exc).__name__}: {exc}"},
                ],
                "current_step": node_name,
            }

    return wrapper


def build_graph(ctx: RuntimeContext):
    g = StateGraph(AgentState)

    # 绑定节点（用 lambda 注入 ctx，并用 _guarded 兜住异常）
    g.add_node("planner", _guarded("planner_node", lambda s: nodes.planner_node(s, ctx)))
    g.add_node("await_input", _guarded("await_input_node", lambda s: nodes.await_input_node(s, ctx)))
    g.add_node("research", _guarded("research_node", lambda s: nodes.research_node(s, ctx)))
    g.add_node("quant", _guarded("quant_node", lambda s: nodes.quant_node(s, ctx)))
    g.add_node("human_confirm", _guarded("human_confirm_node", lambda s: nodes.human_confirm_node(s, ctx)))
    g.add_node("risk", _guarded("risk_node", lambda s: nodes.risk_node(s, ctx)))
    g.add_node("fact_check", _guarded("fact_check_node", lambda s: nodes.fact_check_node(s, ctx)))
    g.add_node("report", _guarded("report_node", lambda s: nodes.report_node(s, ctx)))
    g.add_node("error_handler", _guarded("error_handler_node", lambda s: nodes.error_handler_node(s, ctx)))

    # 边与条件路由
    g.add_edge(START, "planner")
    g.add_conditional_edges(
        "planner",
        routing.route_after_planner,
        {
            "research": "research",
            "quant": "quant",
            "report": "report",
            "await_input": "await_input",
            "error": "error_handler",
        },
    )
    g.add_conditional_edges(
        "research",
        routing.route_after_research,
        {"quant": "quant", "report": "report", "error": "error_handler"},
    )
    g.add_edge("await_input", END)
    g.add_conditional_edges(
        "quant",
        routing.route_after_quant,
        {"human_confirm": "human_confirm", "error": "error_handler"},
    )
    g.add_conditional_edges(
        "human_confirm",
        routing.route_after_confirm,
        {"risk": "risk", "quant": "quant", "end": END, "error": "error_handler"},
    )
    g.add_conditional_edges(
        "risk",
        routing.route_after_risk,
        {"fact_check": "fact_check", "error": "error_handler"},
    )
    g.add_conditional_edges(
        "fact_check",
        routing.route_after_fact_check,
        {"report": "report", "error": "error_handler"},
    )
    g.add_conditional_edges(
        "report",
        routing.route_after_report,
        {"end": END, "error": "error_handler"},
    )
    g.add_edge("error_handler", END)

    # 按配置选择 checkpointer（memory 或 SQLite 持久化）。
    return g.compile(checkpointer=build_checkpointer(ctx.settings))
