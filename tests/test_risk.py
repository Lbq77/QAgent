"""风险审计测试（离线）。"""
from __future__ import annotations

from qagent.backtest import clear_results, save_result
from qagent.tools.risk import AuditStrategy
from qagent.tools.schemas import StrategyConfig


def _config(**overrides) -> StrategyConfig:
    base = dict(
        market="US",
        universe=["A", "B", "C"],
        ranking={"factor": "momentum_20d", "direction": "desc"},
        top_n=3,
        cost=0.001,
        t_plus_1=True,
        lookback_days=60,
    )
    base.update(overrides)
    return StrategyConfig(**base)


def test_audit_good_config_low_risk():
    out = AuditStrategy().call({"strategy_config": _config(), "backtest_result_ref": ""})
    assert out.level == "low"
    assert out.findings == []
    assert out.survivorship_bias is True
    assert "已分离" in out.lookahead_check
    assert "t+1" in out.execution_check
    assert "cost=0.001" in out.cost_check


def test_audit_lookahead_fails():
    out = AuditStrategy().call(
        {"strategy_config": _config(lookback_days=0), "backtest_result_ref": ""}
    )
    assert out.level == "high"
    assert any("未来函数" in f for f in out.findings)


def test_audit_cost_zero():
    out = AuditStrategy().call(
        {"strategy_config": _config(cost=0.0), "backtest_result_ref": ""}
    )
    assert out.level == "medium"
    assert any("成本" in f for f in out.findings)


def test_audit_t_plus_1_false():
    out = AuditStrategy().call(
        {"strategy_config": _config(t_plus_1=False), "backtest_result_ref": ""}
    )
    assert out.level == "medium"
    assert any("t+1" in f for f in out.findings)


def test_audit_reads_failed_backtest():
    clear_results()
    save_result("bt_boom", {"error": "network down", "config": _config().model_dump()})
    out = AuditStrategy().call(
        {"strategy_config": _config(), "backtest_result_ref": "bt_boom"}
    )
    assert out.level == "medium"
    assert any("回测执行失败" in f for f in out.findings)
