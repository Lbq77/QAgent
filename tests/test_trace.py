"""工具调用轨迹（tool_calls 记录进 TraceRecord）。"""
from __future__ import annotations

from qagent.config import Settings
from qagent.context import RuntimeContext
from qagent.data import MarketDataService, StubMarketDataProvider
from qagent.llm import StubLLM
from qagent.runtime import AgentRuntime
from qagent.tools import build_default_registry


def _runtime() -> AgentRuntime:
    ctx = RuntimeContext(
        settings=Settings(model_provider="stub"),
        llm=StubLLM(),
        tool_registry=build_default_registry(
            market_data=MarketDataService([StubMarketDataProvider()])
        ),
    )
    return AgentRuntime(ctx)


QUERY = "研究 AAPL 在 2024-01-01 到 2024-12-31 的 20 日动量并回测"


def _trace(state, node: str):
    return next(t for t in state.trace if t.node == node)


def test_tool_calls_recorded():
    rt = _runtime()
    s = rt.run(QUERY)  # 停在 human_confirm；research/quant 已跑完
    assert any(tc.tool == "get_market_data" for tc in _trace(s, "research_node").tool_calls)
    assert any(tc.tool == "run_backtest" for tc in _trace(s, "quant_node").tool_calls)

    final = rt.confirm(s.task_id, {"action": "approve"})
    assert any(tc.tool == "audit_strategy" for tc in _trace(final, "risk_node").tool_calls)
    assert any(tc.tool == "validate_report_claims" for tc in _trace(final, "fact_check_node").tool_calls)
    assert any(tc.tool == "get_backtest_result" for tc in _trace(final, "report_node").tool_calls)


def test_tool_call_record_has_metadata():
    rt = _runtime()
    final = rt.confirm(rt.run(QUERY).task_id, {"action": "approve"})
    tc = _trace(final, "quant_node").tool_calls[0]
    assert tc.tool == "run_backtest"
    assert tc.agent == "quant"
    assert tc.status == "succeeded"
    assert tc.duration_ms >= 0
    assert tc.input_summary  # 输入摘要非空（上下文外置：只存摘要，不存全量）
