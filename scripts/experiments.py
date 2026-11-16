"""2 个对照实验（离线合成数据，结果记进 ResultStore）。

实验 1：动量因子 vs 波动率因子 —— 同一股票池，比较选股差异与年化收益
实验 2：成本敏感度 —— 同一动量策略，cost 0x vs 2x，比较年化收益

每次 RunBacktest 都把结果（config + metrics + engine_version + created_at）写进
SQLite 结果存储，`list_experiments()` 可回看全部实验记录。

用法（项目根目录）：
    .venv/bin/python scripts/experiments.py
"""
from __future__ import annotations

from qagent.backtest import clear_results
from qagent.data import MarketDataService
from qagent.data.base import MarketDataProvider, MarketRecord
from qagent.store import list_experiments
from qagent.tools.quant import RunBacktest
from qagent.tools.schemas import StrategyConfig


def _records(symbol: str, closes: list[float]) -> list[MarketRecord]:
    return [
        MarketRecord(symbol=symbol, date=f"2024-01-{i + 1:02d}", close=c, open=c)
        for i, c in enumerate(closes)
    ]


class _Synthetic(MarketDataProvider):
    name = "synthetic"

    def __init__(self, data: dict[str, list[MarketRecord]]) -> None:
        self._data = data

    def fetch(self, symbol: str, start: str, end: str, market: str = "US") -> list[MarketRecord]:
        return self._data.get(symbol, [])


def _service() -> MarketDataService:
    # 上行/下行/震荡三只，足够拉开因子差异
    up = [100 * 1.02**i for i in range(40)]
    down = [100 * 0.98**i for i in range(40)]
    chop = [100 + 5 * ((-1) ** i) for i in range(40)]
    data = {
        "UP": _records("UP", up),
        "DOWN": _records("DOWN", down),
        "CHOP": _records("CHOP", chop),
    }
    return MarketDataService([_Synthetic(data)])


def _run(svc: MarketDataService, ranking: dict, cost: float, lookback: int = 25) -> dict:
    cfg = StrategyConfig(
        market="US",
        universe=["UP", "DOWN", "CHOP"],
        ranking=ranking,
        top_n=2,
        cost=cost,
        t_plus_1=True,
        lookback_days=lookback,
    )
    out = RunBacktest(service=svc).call(
        {"strategy_config": cfg, "start": "2024-01-01", "end": "2024-12-31"}
    )
    payload = {"config": cfg.model_dump(), "ref": out.ref, "metrics": out.metrics.model_dump()}
    return payload


def main() -> None:
    clear_results()
    svc = _service()

    # 实验 1：动量 desc vs 波动率 asc（同一股票池，看选股差异）
    exp_mom = _run(svc, {"factor": "momentum_20d", "direction": "desc"}, cost=0.001)
    exp_vol = _run(svc, {"factor": "volatility_20d", "direction": "asc"}, cost=0.001)

    # 实验 2：成本敏感度（同一动量策略，cost 0x vs 2x）
    exp_0x = _run(svc, {"factor": "momentum_20d", "direction": "desc"}, cost=0.0)
    exp_2x = _run(svc, {"factor": "momentum_20d", "direction": "desc"}, cost=0.002)

    print("== 实验 1：动量 desc vs 波动率 asc（top_n=2）==")
    print(f"  动量   annual_return={exp_mom['metrics']['annual_return']:.4f}  ref={exp_mom['ref']}")
    print(f"  波动率 annual_return={exp_vol['metrics']['annual_return']:.4f}  ref={exp_vol['ref']}")

    print("\n== 实验 2：成本敏感度（动量策略）==")
    print(f"  0x  annual_return={exp_0x['metrics']['annual_return']:.4f}")
    print(f"  2x  annual_return={exp_2x['metrics']['annual_return']:.4f}")
    print(f"  成本损耗 = {exp_0x['metrics']['annual_return'] - exp_2x['metrics']['annual_return']:.4f}")

    print(f"\n共记录 {len(list_experiments())} 条实验（list_experiments() 可回看）")


if __name__ == "__main__":
    main()
