"""LangGraph StateGraph 构建（PRD §6.1）。

流程（Day 2 支持按需跳过）：
    START → planner →(条件)→ research →(条件)→ quant → human_confirm
         →(条件)→ risk → fact_check → report → END
    planner 缺参数 → await_input → END（blocked）
    失败/拒绝 → error_handler → END

Day 1 用 MemorySaver；Day 8 换成 SQLite Checkpointer 实现断点恢复。
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from ..context import RuntimeContext
from ..state import AgentState
from . import nodes
from . import routing


def build_graph(ctx: RuntimeContext):
    g = StateGraph(AgentState)

    # 绑定节点（用 lambda 注入 ctx，保持节点签名统一）
    g.add_node("planner", lambda s: nodes.planner_node(s, ctx))
    g.add_node("await_input", lambda s: nodes.await_input_node(s, ctx))
    g.add_node("research", lambda s: nodes.research_node(s, ctx))
    g.add_node("quant", lambda s: nodes.quant_node(s, ctx))
    g.add_node("human_confirm", lambda s: nodes.human_confirm_node(s, ctx))
    g.add_node("risk", lambda s: nodes.risk_node(s, ctx))
    g.add_node("fact_check", lambda s: nodes.fact_check_node(s, ctx))
    g.add_node("report", lambda s: nodes.report_node(s, ctx))
    g.add_node("error_handler", lambda s: nodes.error_handler_node(s, ctx))

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
    g.add_edge("quant", "human_confirm")
    g.add_conditional_edges(
        "human_confirm",
        routing.route_after_confirm,
        {"risk": "risk", "end": END},
    )
    g.add_conditional_edges(
        "risk",
        routing.route_after_risk,
        {"fact_check": "fact_check", "error": "error_handler"},
    )
    g.add_edge("fact_check", "report")
    g.add_edge("report", END)
    g.add_edge("error_handler", END)

    # Day 1：内存 checkpointer；Day 8 换 SqliteSaver。
    return g.compile(checkpointer=MemorySaver())
