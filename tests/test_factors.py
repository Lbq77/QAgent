"""因子归一化测试。"""
import pytest

from qagent.factors import UnknownFactorError, detect_unsupported_factor, resolve_factor


def test_market_cap_default_asc():
    assert resolve_factor("market_cap", None) == {"factor": "market_cap", "direction": "asc"}


def test_market_cap_alias():
    assert resolve_factor("小市值", None) == {"factor": "market_cap", "direction": "asc"}


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
    assert resolve_factor("market_cap", "desc") == {"factor": "market_cap", "direction": "desc"}


def test_detect_unsupported_factor():
    assert detect_unsupported_factor("使用估值因子回测") == "估值因子"
    assert detect_unsupported_factor("按市盈率选股") == "市盈率"
    assert detect_unsupported_factor("用 ROE 因子") == ""  # 单独 roe 未收录，避免误报


def test_detect_unsupported_factor_no_match():
    assert detect_unsupported_factor("用 20 日动量回测") == ""
    assert detect_unsupported_factor("用小市值因子") == ""
