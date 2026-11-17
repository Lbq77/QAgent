"""回测引擎与 quant 工具闭环测试（离线，合成数据）。"""
from __future__ import annotations

import pytest

from qagent.backtest import clear_results, load_result, run_backtest_metrics
from qagent.data import MarketDataService
from qagent.data.base import MarketDataProvider
from qagent.tools.quant import (
    CalculateFactor,
    GetBacktestResult,
    RunBacktest,
    ScreenStocks,
)
from qagent.tools.schemas import BacktestMetrics, MarketRecord, StrategyConfig


def _series(start: float, n: int, step: float) -> list[float]:
    return [round(start * (1 + step) ** i, 4) for i in range(n)]


def _records(symbol: str, closes: list[float], volumes: list[float] | None = None) -> list[MarketRecord]:
    return [
        MarketRecord(
            symbol=symbol,
            date=f"2024-01-{i + 1:02d}",
            close=c,
            open=c,  # 合成数据 open≈close，首日 open-to-close 收益为 0，便于断言
            volume=(volumes[i] if volumes else 0.0),
        )
        for i, c in enumerate(closes)
    ]


class _Synthetic(MarketDataProvider):
    name = "synthetic"

    def __init__(self, data: dict[str, list[MarketRecord]], caps: dict[str, float] | None = None):
        self._data = data
        self._caps = caps or {}

    def fetch(self, symbol: str, start: str, end: str, market: str = "US") -> list[MarketRecord]:
        return self._data.get(symbol, [])

    def fetch_market_cap(self, symbol: str) -> float | None:
        return self._caps.get(symbol)


def _service() -> MarketDataService:
    # A 持续上涨、B 持续下跌、C 温和上涨 → momentum desc 应 A > C > B
    data = {
        "A": _records("A", _series(100, 25, 0.02)),
        "B": _records("B", _series(100, 25, -0.02)),
        "C": _records("C", _series(100, 25, 0.01)),
    }
    return MarketDataService([_Synthetic(data)])


# ---- 纯回测指标 ----


def test_backtest_empty():
    assert run_backtest_metrics({}) == BacktestMetrics()


def test_backtest_constant_gain_no_opens():
    closes = _series(100, 11, 0.03)
    m = run_backtest_metrics({"X": closes}, cost=0.0)
    # 无开盘价：入口近似 t+1 收盘，其后 close-to-close，共 10 个 +3% 收益
    total = 1.03**10 - 1
    assert m.annual_return == pytest.approx(total * 252 / 10, rel=1e-3)
    assert m.max_drawdown == pytest.approx(0.0, abs=1e-6)


def test_backtest_open_execution():
    # 持有窗 2 根：t+1 开盘 100 买入收盘 110；次日收盘 121 → 两日各 +10%
    closes = [110.0, 121.0]
    opens = [100.0, 110.0]
    m = run_backtest_metrics({"X": closes}, {"X": opens}, cost=0.0)
    total = 1.1 * 1.1 - 1
    assert m.annual_return == pytest.approx(total * 252 / 2, rel=1e-3)


def test_backtest_sharpe_positive():
    m = run_backtest_metrics({"X": [100, 103, 101, 105, 104, 108]}, cost=0.0)
    assert m.sharpe > 0


def test_backtest_drawdown():
    m = run_backtest_metrics({"X": [100, 110, 99, 100]}, cost=0.0)
    assert m.max_drawdown == pytest.approx(0.1, abs=0.01)


def test_backtest_cost_reduces_return():
    closes = _series(100, 11, 0.03)
    a = run_backtest_metrics({"X": closes}, cost=0.0)
    b = run_backtest_metrics({"X": closes}, cost=0.01)
    assert b.annual_return < a.annual_return


# ---- quant 工具闭环 ----


def test_calculate_factor_ranks():
    svc = _service()
    out = CalculateFactor(service=svc).call(
        {
            "factor": "momentum_20d",
            "direction": "desc",
            "universe": ["A", "B", "C"],
            "start": "2024-01-01",
            "end": "2024-12-31",
        }
    )
    symbols = [s.symbol for s in out.ranking]
    assert symbols[0] == "A"
    assert symbols[-1] == "B"


def test_screen_stocks_top_n():
    svc = _service()
    out = ScreenStocks(service=svc).call(
        {
            "universe": ["A", "B", "C"],
            "top_n": 2,
            "factor": "momentum_20d",
            "direction": "desc",
            "start": "2024-01-01",
            "end": "2024-12-31",
        }
    )
    assert out.selected == ["A", "C"]


def test_run_backtest_and_retrieve():
    clear_results()
    svc = _service()
    cfg = StrategyConfig(
        market="US",
        universe=["A", "B", "C"],
        ranking={"factor": "momentum_5d", "direction": "desc"},
        top_n=1,
        cost=0.001,
        t_plus_1=True,
        lookback_days=10,
    )
    out = RunBacktest(service=svc).call(
        {"strategy_config": cfg, "start": "2024-01-01", "end": "2024-12-31"}
    )
    assert out.ref.startswith("bt_")
    assert out.metrics.annual_return != 0.0
    assert set(out.cost_scenarios) == {"0x", "default", "2x"}
    assert out.cost_scenarios["0x"].annual_return >= out.cost_scenarios["default"].annual_return

    got = GetBacktestResult().call({"ref": out.ref})
    assert got.ref == out.ref
    assert got.metrics.sharpe == out.metrics.sharpe
    assert got.cost_scenarios["default"].sharpe == out.metrics.sharpe


def test_get_backtest_result_missing():
    clear_results()
    got = GetBacktestResult().call({"ref": "bt_nope"})
    assert got.metrics == BacktestMetrics()


def test_run_backtest_no_lookahead():
    clear_results()
    # UP：信号窗涨、持有窗跌；DOWN：信号窗跌、持有窗涨。
    # 正确做法应选 UP（只用信号窗动量）；未来函数做法会选 DOWN（用了持有窗的涨幅）。
    up = [100 * 1.01**i for i in range(10)] + [100 * 1.01**9 * 0.99**i for i in range(1, 16)]
    down = [100 * 0.99**i for i in range(10)] + [100 * 0.99**9 * 1.01**i for i in range(1, 16)]
    svc = MarketDataService(
        [_Synthetic({"UP": _records("UP", up), "DOWN": _records("DOWN", down)})]
    )
    cfg = StrategyConfig(
        market="US",
        universe=["UP", "DOWN"],
        ranking={"factor": "momentum_5d", "direction": "desc"},
        top_n=1,
        cost=0.0,
        lookback_days=10,
    )
    out = RunBacktest(service=svc).call(
        {"strategy_config": cfg, "start": "2024-01-01", "end": "2024-12-31"}
    )
    assert load_result(out.ref)["selected"] == ["UP"]


def test_run_backtest_reuses_market_snapshot():
    """research 存了行情快照后，quant 应按 snapshot_id 复用，不再重复 fetch。"""
    from qagent.tools.research import GetMarketData

    clear_results()

    class _CountingProvider(MarketDataProvider):
        name = "counting"

        def __init__(self) -> None:
            self.count = 0

        def fetch(self, symbol: str, start: str, end: str, market: str = "US"):
            self.count += 1
            return _records(symbol, _series(100, 80, 0.02))

    provider = _CountingProvider()
    svc = MarketDataService([provider])

    # 1) get_market_data 拉数据并落快照
    md = GetMarketData(service=svc).call(
        {"symbols": ["A", "B"], "start": "2024-01-01", "end": "2024-12-31", "market": "US"}
    )
    assert md.snapshot_id.startswith("snap_")
    assert provider.count == 2  # 拉了 A、B 两只

    # 2) run_backtest 带 snapshot_id → 复用快照，不再 fetch
    cfg = StrategyConfig(
        market="US",
        universe=["A", "B"],
        ranking={"factor": "momentum_20d", "direction": "desc"},
        top_n=2,
        lookback_days=60,
    )
    out = RunBacktest(service=svc).call(
        {
            "strategy_config": cfg,
            "start": "2024-01-01",
            "end": "2024-12-31",
            "snapshot_id": md.snapshot_id,
        }
    )
    assert out.ref.startswith("bt_")
    assert provider.count == 2  # 没有新增 fetch，说明复用了快照
