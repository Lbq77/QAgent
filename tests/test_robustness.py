"""健壮性回归测试：工具抛错走 error_handler、工具调用预算按任务重置。"""
from __future__ import annotations

from qagent.config import Settings
from qagent.context import RuntimeContext
from qagent.data import MarketDataService, StubMarketDataProvider
from qagent.harness import Harness
from qagent.llm import StubLLM
from qagent.runtime import AgentRuntime
from qagent.tools import build_default_registry

QUERY = "研究 AAPL 在 2024-01-01 到 2024-12-31 的 20 日动量并回测"


def _runtime(harness: Harness) -> AgentRuntime:
    ctx = RuntimeContext(
        settings=Settings(model_provider="stub"),
        llm=StubLLM(),
        tool_registry=build_default_registry(
            market_data=MarketDataService([StubMarketDataProvider()]), harness=harness
        ),
    )
    return AgentRuntime(ctx)


def test_tool_error_routes_to_error_handler():
    """工具调用超预算 → 不应崩掉整个流程，而应路由到 error_handler 落 FAILED。"""
    harness = Harness(max_tool_calls=0, timeout_seconds=30.0, retries=0)  # 预算为 0，首次调用即失败
    rt = _runtime(harness)

    state = rt.run(QUERY)  # research 第一步就超预算

    assert state.status.value == "failed"
    assert "error_handler_node" in [t.node for t in state.trace]
    assert any(e.get("node") == "research_node" for e in state.errors)


def test_harness_budget_resets_between_tasks():
    """同一个进程里连续跑两个任务，第二个任务不该因为预算累加而失败。"""
    # 一个完整任务约 6 次工具调用；给 8 的余量，若不重置第二个任务必超预算
    harness = Harness(max_tool_calls=8, timeout_seconds=30.0, retries=0)
    rt = _runtime(harness)

    s1 = rt.run(QUERY)
    f1 = rt.confirm(s1.task_id, {"action": "approve"})
    assert f1.status.value == "succeeded"

    s2 = rt.run(QUERY)  # 预算应已被 _reset_harness 清零
    f2 = rt.confirm(s2.task_id, {"action": "approve"})
    assert f2.status.value == "succeeded"
