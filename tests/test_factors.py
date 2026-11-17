"""因子归一化测试。"""
import pytest

from qagent.factors import (
    UnknownFactorError,
    detect_unsupported_factor,
    factor_value,
    resolve_factor,
)


def test_market_cap_unsupported():
    # 市值因子 PIT（时点）未对齐、存在未来函数，已从白名单移除：resolve 报错、文本检测命中
    with pytest.raises(UnknownFactorError):
        resolve_factor("market_cap", None)
    with pytest.raises(UnknownFactorError):
        resolve_factor("小市值", None)
    assert detect_unsupported_factor("用小市值因子选股") == "市值因子"


def test_momentum_default_desc():
    assert resolve_factor("momentum_20d", None) == {"factor": "momentum_20d", "direction": "desc"}


def test_momentum_window_aliases():
    assert resolve_factor("5日动量", None)["factor"] == "momentum_5d"
    assert resolve_factor("60日动量", None)["factor"] == "momentum_60d"


def test_empty_factor_defaults_to_momentum():
    # 用户「没指定」因子 → 默认动量；这不是替换
    assert resolve_factor("", None) == {"factor": "momentum_20d", "direction": "desc"}


def test_unknown_factor_raises():
    # 用户「明确指定」但不在白名单 → 报错，不静默替换
    with pytest.raises(UnknownFactorError):
        resolve_factor("估值因子", "asc")


def test_respects_explicit_direction():
    # 显式方向优先于默认方向（波动率默认 asc，显式 desc 应被尊重）
    assert resolve_factor("volatility_20d", "desc") == {"factor": "volatility_20d", "direction": "desc"}


def test_detect_unsupported_factor():
    assert detect_unsupported_factor("使用估值因子回测") == "估值因子"
    assert detect_unsupported_factor("按市盈率选股") == "市盈率"
    assert detect_unsupported_factor("用 ROE 因子") == ""  # 单独 roe 未收录，避免误报


def test_detect_unsupported_factor_no_match():
    assert detect_unsupported_factor("用 20 日动量回测") == ""


def test_factor_value_momentum():
    # 5 日涨 5%：105/100 - 1 = 5%
    closes = [100, 101, 102, 103, 104, 105]
    assert factor_value("momentum_5d", closes) == pytest.approx(5.0)


def test_factor_value_momentum_insufficient_data():
    # 窗口不足返回 None，而不是编造数值
    assert factor_value("momentum_20d", [100, 101]) is None


def test_factor_value_volatility_flat():
    # 恒等收益 → 波动率为 0
    assert factor_value("volatility_20d", [100, 110, 121]) == pytest.approx(0.0)


def test_factor_value_avg_volume():
    assert factor_value("avg_volume_20d", [100, 101], [10, 20, 30]) == pytest.approx(20.0)


def test_factor_value_unknown_returns_none():
    assert factor_value("bogus", [100, 101]) is None
