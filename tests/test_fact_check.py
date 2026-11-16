"""数字核查（validate_report_claims 真实比对）+ 证据链。"""
from __future__ import annotations

from qagent.backtest import clear_results, save_result
from qagent.config import Settings
from qagent.context import RuntimeContext
from qagent.data import MarketDataService, StubMarketDataProvider
from qagent.llm import StubLLM
from qagent.runtime import AgentRuntime
from qagent.tools import build_default_registry
from qagent.tools.risk import ValidateReportClaims
from qagent.tools.schemas import Claim


def test_validate_claims_all_match():
    clear_results()
    save_result(
        "bt_ok",
        {"metrics": {"annual_return": 0.5, "max_drawdown": 0.1, "sharpe": 1.2}},
    )
    out = ValidateReportClaims().call(
        {
            "claims": [
                Claim(field="annual_return", value=0.5),
                Claim(field="max_drawdown", value=0.1),
                Claim(field="sharpe", value=1.2),
            ],
            "backtest_result_ref": "bt_ok",
        }
    )
    assert out.claims_checked == 3
    assert out.mismatches == []


def test_validate_claims_detects_mismatch():
    clear_results()
    save_result("bt_x", {"metrics": {"annual_return": 0.5}})
    out = ValidateReportClaims().call(
        {
            "claims": [Claim(field="annual_return", value=0.99)],  # 声称值 ≠ 实际值
            "backtest_result_ref": "bt_x",
        }
    )
    assert out.claims_checked == 1
    assert len(out.mismatches) == 1
    assert "annual_return" in out.mismatches[0]


def test_validate_claims_unresolvable_ref():
    clear_results()
    out = ValidateReportClaims().call(
        {
            "claims": [Claim(field="annual_return", value=0.5)],
            "backtest_result_ref": "bt_missing",
        }
    )
    assert any("不可用" in m for m in out.mismatches)


def _runtime() -> AgentRuntime:
    ctx = RuntimeContext(
        settings=Settings(model_provider="stub"),
        llm=StubLLM(),
        tool_registry=build_default_registry(
            market_data=MarketDataService([StubMarketDataProvider()])
        ),
    )
    return AgentRuntime(ctx)


def test_fact_check_happy_path_no_mismatches():
    rt = _runtime()
    s = rt.run("研究 AAPL 在 2024-01-01 到 2024-12-31 的 20 日动量并回测")
    final = rt.confirm(s.task_id, {"action": "approve"})
    # 3（确定性指标）+ 3（总结文本里的三个指标）= 6
    assert final.fact_check_result.claims_checked == 6
    assert final.fact_check_result.mismatches == []
    assert "证据链" in final.final_report  # 报告带证据链
