"""相对时间 / 区间 → 具体日期测试（固定 today，保证确定性）。"""
from datetime import date

from qagent.dates import resolve_date_range

TODAY = date(2026, 11, 4)


def test_recent_one_year():
    assert resolve_date_range("最近一年", today=TODAY) == {
        "start": "2025-11-04",
        "end": "2026-11-04",
    }


def test_recent_three_months():
    assert resolve_date_range("最近三个月", today=TODAY) == {
        "start": "2026-08-04",
        "end": "2026-11-04",
    }


def test_past_five_years():
    assert resolve_date_range("过去五年", today=TODAY) == {
        "start": "2021-11-04",
        "end": "2026-11-04",
    }


def test_explicit_year_range():
    assert resolve_date_range("2020年至2024年") == {
        "start": "2020-01-01",
        "end": "2024-12-31",
    }


def test_explicit_date_range():
    assert resolve_date_range("2024-01-01 到 2024-06-30") == {
        "start": "2024-01-01",
        "end": "2024-06-30",
    }


def test_no_date_returns_none():
    assert resolve_date_range("研究 AAPL 的动量") is None
