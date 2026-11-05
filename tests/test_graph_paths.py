"""Day 2 图级路径测试：同一张图按 state 走出不同分支。"""
import json
from datetime import date

from qagent.config import Settings
from qagent.context import RuntimeContext
from qagent.data import MarketDataService, StubMarketDataProvider
from qagent.llm import StubLLM
from qagent.runtime import AgentRuntime
from qagent.tools import build_default_registry


def _runtime(raw: str | None = None) -> AgentRuntime:
    ctx = RuntimeContext(
        settings=Settings(model_provider="stub"),
        llm=StubLLM(raw=raw),
        tool_registry=build_default_registry(
            market_data=MarketDataService([StubMarketDataProvider()])
        ),
    )
    return AgentRuntime(ctx)


def _plan_json(**overrides) -> str:
    base = {
        "goal": "g",
        "needs_research": True,
        "needs_backtest": True,
        "needs_risk_audit": True,
        "missing_params": [],
    }
    base.update(overrides)
    return json.dumps(base)


def test_normal_full_path():
    state = _runtime().run("研究 AAPL 在 2024-01-01 到 2024-12-31 的 20 日动量并回测")
    assert state.status.value == "succeeded"
    assert [t.node for t in state.trace] == [
        "planner_node",
        "research_node",
        "quant_node",
        "human_confirm_node",
        "risk_node",
        "fact_check_node",
        "report_node",
    ]


def test_research_only_skips_quant():
    raw = _plan_json(needs_backtest=False, needs_risk_audit=False)
    state = _runtime(raw).run("只查 AAPL 的资料")
    nodes = [t.node for t in state.trace]
    assert "quant_node" not in nodes
    assert "report_node" in nodes
    assert state.status.value == "succeeded"


def test_no_backtest_skips_research_and_quant():
    raw = _plan_json(needs_research=False, needs_backtest=False, needs_risk_audit=False)
    state = _runtime(raw).run("随便聊聊")
    assert [t.node for t in state.trace] == ["planner_node", "report_node"]


def test_missing_params_blocks():
    raw = _plan_json(missing_params=["date_range"])
    state = _runtime(raw).run("帮我回测一下动量策略")
    assert state.status.value == "blocked"
    assert [t.node for t in state.trace] == ["planner_node", "await_input_node"]
    assert "回测时间范围" in state.pending_question


def test_invalid_json_falls_back_and_succeeds():
    state = _runtime("这不是 JSON").run("研究 AAPL 在 2024-01-01 到 2024-12-31 的动量")
    assert state.status.value == "succeeded"
    assert any(e.get("node") == "planner_node" for e in state.errors)


def test_factor_propagation_to_strategy():
    raw = json.dumps(
        {
            "goal": "g",
            "needs_research": True,
            "needs_backtest": True,
            "needs_risk_audit": True,
            "missing_params": [],
            "market": "US",
            "universe": ["AAPL"],
            "date_range": {"start": "2024-01-01", "end": "2024-12-31"},
            "factor": "market_cap",
            "factor_direction": "asc",
        }
    )
    state = _runtime(raw).run("用小市值因子选股并回测")
    assert state.status.value == "succeeded"
    assert state.strategy_config["ranking"] == {"factor": "market_cap", "direction": "asc"}


def test_index_universe_blocks_not_substituted():
    # 用户说标普500 → universe 空 + 代码识别指数 → block，且绝不静默替换成默认股票
    raw = json.dumps(
        {
            "goal": "g",
            "needs_research": True,
            "needs_backtest": True,
            "needs_risk_audit": True,
            "missing_params": [],
            "market": "US",
            "universe": [],
            "date_range": {"start": "2023-01-01", "end": "2026-01-01"},
        }
    )
    state = _runtime(raw).run("在标普500中使用小市值因子回测")
    assert state.status.value == "blocked"
    assert "标普500" in state.pending_question
    assert state.strategy_config == {}  # 没走到 quant，未静默替换


def test_empty_universe_blocks():
    raw = json.dumps(
        {
            "goal": "g",
            "needs_research": True,
            "needs_backtest": True,
            "needs_risk_audit": True,
            "missing_params": [],
            "market": "US",
            "universe": [],
            "date_range": {"start": "2023-01-01", "end": "2026-01-01"},
        }
    )
    state = _runtime(raw).run("回测小市值因子")  # 无代码也无指数
    assert state.status.value == "blocked"
    assert "股票池" in state.pending_question


def test_top_n_extracted():
    raw = json.dumps(
        {
            "goal": "g",
            "needs_research": True,
            "needs_backtest": True,
            "needs_risk_audit": True,
            "missing_params": [],
            "market": "US",
            "universe": ["AAPL", "MSFT"],
            "date_range": {"start": "2023-01-01", "end": "2026-01-01"},
            "top_n": 10,
        }
    )
    state = _runtime(raw).run("选出市值最小的10只")
    assert state.status.value == "succeeded"
    assert state.strategy_config["top_n"] == 10


def test_unknown_factor_blocks_not_substituted():
    # 用户明确要一个不存在的因子 → 阻塞，绝不静默替换成动量
    raw = json.dumps(
        {
            "goal": "g",
            "needs_research": True,
            "needs_backtest": True,
            "needs_risk_audit": True,
            "missing_params": [],
            "market": "US",
            "universe": ["AAPL"],
            "date_range": {"start": "2023-01-01", "end": "2026-01-01"},
            "factor": "估值因子",
            "factor_direction": "asc",
        }
    )
    state = _runtime(raw).run("用一个不存在的估值因子回测")
    assert state.status.value == "blocked"
    assert "估值因子" in state.pending_question
    assert state.strategy_config == {}  # 没走到 quant，未替换


def test_unsupported_factor_in_query_blocks_even_if_llm_substitutes():
    # LLM 把「估值因子」擅自替换成 momentum_20d（合法因子），代码仍按用户原话阻塞
    raw = json.dumps(
        {
            "goal": "g",
            "needs_research": True,
            "needs_backtest": True,
            "needs_risk_audit": True,
            "missing_params": [],
            "market": "US",
            "universe": ["AAPL"],
            "date_range": {"start": "2023-01-01", "end": "2026-01-01"},
            "factor": "momentum_20d",
            "factor_direction": "desc",
        }
    )
    state = _runtime(raw).run("使用估值因子回测")
    assert state.status.value == "blocked"
    assert "估值因子" in state.pending_question
    assert state.strategy_config == {}


def test_relative_date_overrides_llm_hallucinated_date():
    # LLM 返回了它瞎猜的日期（模拟它不知道今天），代码应覆盖为正确相对区间
    raw = json.dumps(
        {
            "goal": "g",
            "needs_research": True,
            "needs_backtest": True,
            "needs_risk_audit": True,
            "missing_params": [],
            "market": "US",
            "universe": ["AAPL"],
            "date_range": {"start": "2018-01-01", "end": "2023-01-01"},
        }
    )
    state = _runtime(raw).run("回测过去五年 AAPL 的表现")
    assert state.status.value == "succeeded"
    start = state.task_constraints.date_range["start"]
    assert start.startswith(str(date.today().year - 5))  # 今天往前推 5 年，而非 LLM 的 2018


def test_resume_continues_same_task():
    rt = _runtime()  # stub
    s1 = rt.run("回测 AAPL 的动量策略")  # 无日期 → blocked
    assert s1.status.value == "blocked"
    assert s1.pending_question

    tid = s1.task_id
    s2 = rt.resume(tid, "比较 2020-01-01 到 2024-12-31 这段时间")
    assert s2.task_id == tid  # 同一任务，不新建
    assert s2.status.value == "succeeded"
    assert s2.task_constraints.date_range == {"start": "2020-01-01", "end": "2024-12-31"}
