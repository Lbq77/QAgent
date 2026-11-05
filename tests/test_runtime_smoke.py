"""Day 1/2 冒烟测试：离线（stub 模型）跑通整条骨架并校验轨迹。

显式使用 StubLLM，避免测试依赖 .env 或网络。
"""
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


def test_smoke_happy_path():
    state = _runtime().run("研究 AAPL、MSFT、NVDA 在 2024-01-01 到 2024-12-31 的 20 日动量并回测")

    assert state.status.value == "succeeded"
    assert state.strategy_config.get("t_plus_1") is True
    assert state.strategy_config.get("cost", 0) != 0
    assert state.risk_report.survivorship_bias is True

    nodes = [t.node for t in state.trace]
    for expected in [
        "planner_node",
        "research_node",
        "quant_node",
        "human_confirm_node",
        "risk_node",
        "fact_check_node",
        "report_node",
    ]:
        assert expected in nodes, f"missing node {expected} in {nodes}"
