"""Day 3 工具系统单元测试：注册表、权限白名单、Schema 校验、节点接入。"""
from __future__ import annotations

import pytest

from qagent.config import Settings
from qagent.context import RuntimeContext
from qagent.data import MarketDataService, StubMarketDataProvider
from qagent.llm import StubLLM
from qagent.runtime import AgentRuntime
from qagent.tools import (
    Tool,
    ToolInputError,
    ToolNotFoundError,
    ToolOutputError,
    ToolPermissionError,
    ToolRegistry,
    build_default_registry,
    ensure_agent_allowed,
)
from qagent.tools.research import GetMarketData
from qagent.tools.schemas import ScreenStocksInput, ScreenStocksOutput, StrategyConfig


def test_registry_registers_all_nine_tools():
    reg = build_default_registry()
    assert reg.names() == [
        "audit_strategy",
        "calculate_factor",
        "get_backtest_result",
        "get_company_profile",
        "get_market_data",
        "run_backtest",
        "screen_stocks",
        "search_financial_knowledge",
        "validate_report_claims",
    ]
    assert "get_market_data" in reg
    assert reg.resolve("run_backtest").name == "run_backtest"


def test_duplicate_register_raises():
    reg = ToolRegistry()
    reg.register(GetMarketData())
    with pytest.raises(ValueError):
        reg.register(GetMarketData())


def test_resolve_unknown_raises():
    with pytest.raises(ToolNotFoundError):
        ToolRegistry().resolve("nope")


def test_whitelist_allows_own_tools():
    ensure_agent_allowed("research", "get_market_data")  # 不抛异常


def test_whitelist_blocks_planner():
    with pytest.raises(ToolPermissionError):
        ensure_agent_allowed("planner", "get_market_data")  # planner 是 orchestrate-only


def test_whitelist_blocks_wrong_agent():
    with pytest.raises(ToolPermissionError):
        ensure_agent_allowed("research", "run_backtest")


def test_invoke_checks_whitelist():
    reg = build_default_registry()
    with pytest.raises(ToolPermissionError):
        reg.invoke("research", "run_backtest", {})


def test_input_validation_missing_required():
    reg = build_default_registry()
    with pytest.raises(ToolInputError):
        reg.resolve("get_company_profile").call({})  # symbol 必填缺失


def test_input_validation_wrong_type():
    reg = build_default_registry()
    with pytest.raises(ToolInputError):
        reg.resolve("screen_stocks").call({"top_n": "three"})


class _BrokenTool(Tool):
    name = "broken"
    description = "d"
    input_model = ScreenStocksInput
    output_model = ScreenStocksOutput
    permission = "read-only"

    def execute(self, args: ScreenStocksInput) -> dict:
        return {"selected": "not-a-list"}  # 类型错误，无法通过 output_model 校验


def test_output_validation_raises():
    reg = ToolRegistry()
    reg.register(_BrokenTool())
    with pytest.raises(ToolOutputError):
        reg.resolve("broken").call({"top_n": 3})


def test_strategy_config_defaults():
    cfg = StrategyConfig(
        market="US",
        universe=["AAPL"],
        ranking={"factor": "momentum_20d", "direction": "desc"},
    )
    assert cfg.cost == 0.001
    assert cfg.t_plus_1 is True


def test_run_backtest_rejects_invalid_config():
    reg = build_default_registry()
    with pytest.raises(ToolInputError):
        reg.resolve("run_backtest").call({"strategy_config": {"top_n": "nope"}})


def test_context_has_default_registry():
    ctx = RuntimeContext(settings=Settings(model_provider="stub"), llm=StubLLM())
    assert len(ctx.tool_registry.names()) == 9


def test_research_node_uses_tool_registry():
    svc = MarketDataService([StubMarketDataProvider()])
    ctx = RuntimeContext(
        settings=Settings(model_provider="stub"),
        llm=StubLLM(),
        tool_registry=build_default_registry(market_data=svc),
    )
    state = AgentRuntime(ctx).run(
        "研究 AAPL、MSFT、NVDA 在 2024-01-01 到 2024-12-31 的 20 日动量并回测"
    )
    assert state.snapshot_id.startswith("snap_")  # 来自 get_market_data
    assert state.evidence_refs == ["stub"]  # 来自 StubMarketDataProvider
