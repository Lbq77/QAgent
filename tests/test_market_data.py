"""Day 4 行情数据源单元测试（离线，不碰网络）。"""
from __future__ import annotations

import pandas as pd
import pytest

from qagent.data import (
    MarketDataError,
    MarketDataProvider,
    MarketDataService,
    StubMarketDataProvider,
    build_default_service,
)
from qagent.data.yahoo import _df_to_records
from qagent.tools.research import GetMarketData


class _Raising(MarketDataProvider):
    name = "raising"

    def fetch(self, symbol, start, end, market="US"):
        raise MarketDataError("boom")


class _Empty(MarketDataProvider):
    name = "empty"

    def fetch(self, symbol, start, end, market="US"):
        return []


class _One(MarketDataProvider):
    name = "one"

    def fetch(self, symbol, start, end, market="US"):
        from qagent.tools.schemas import MarketRecord

        return [MarketRecord(symbol=symbol, date=end, close=7.0)]


def test_stub_provider_deterministic():
    recs = StubMarketDataProvider().fetch("AAPL", "2024-01-01", "2024-12-31")
    assert len(recs) == 1
    assert recs[0].symbol == "AAPL"
    assert recs[0].close == 100.0


def test_service_falls_back_on_exception():
    svc = MarketDataService([_Raising(), _One()])
    res = svc.fetch("AAPL", "2024-01-01", "2024-12-31")
    assert res.source == "one"
    assert res.records[0].close == 7.0


def test_service_falls_back_on_empty():
    svc = MarketDataService([_Empty(), _One()])
    assert svc.fetch("AAPL", "2024-01-01", "2024-12-31").source == "one"


def test_service_all_fail_raises():
    svc = MarketDataService([_Raising(), _Empty()])
    with pytest.raises(MarketDataError):
        svc.fetch("AAPL", "2024-01-01", "2024-12-31")


def test_get_market_data_tool_marks_missing():
    tool = GetMarketData(service=MarketDataService([_Raising()]))
    out = tool.call({"symbols": ["AAPL", "MSFT"], "start": "2024-01-01", "end": "2024-12-31"})
    assert out.records == []
    assert out.missing == ["AAPL", "MSFT"]


def test_df_to_records_skips_nan():
    df = pd.DataFrame(
        {"Close": [100.0, 101.0, float("nan"), 103.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
    )
    recs = _df_to_records(df, "AAPL")
    assert [r.close for r in recs] == [100.0, 101.0, 103.0]  # NaN 行被跳过
    assert recs[0].symbol == "AAPL"
    assert recs[0].date == "2024-01-02"


def test_df_to_records_empty():
    assert _df_to_records(pd.DataFrame(), "AAPL") == []


def test_default_service_uses_yfinance():
    svc = build_default_service()
    assert svc.providers[0].name == "yfinance"
