"""Quant 工具（因子 / 选股 / 回测接真实确定性计算）。

回测遵守 信号窗与持有窗分离（防未来函数）、t+1 开盘执行、三档成本对比。
数据一律经 MarketDataService 获取；因子值用 factors.factor_value 确定性计算；
结果按 ref 存入进程内存储，可复现、不编造。
"""
from __future__ import annotations

import uuid

from ..backtest import load_result, run_backtest_metrics, run_backtest_series, save_result
from ..data import MarketDataService, build_default_service
from ..data.base import MarketRecord
from ..factors import factor_value
from .base import Tool
from .schemas import (
    BacktestMetrics,
    CalculateFactorInput,
    CalculateFactorOutput,
    FactorScore,
    GetBacktestResultInput,
    RunBacktestInput,
    RunBacktestOutput,
    ScreenStocksInput,
    ScreenStocksOutput,
)


def _fetch_records(
    service: MarketDataService,
    symbols: list[str],
    start: str,
    end: str,
    market: str,
) -> dict[str, list[MarketRecord]]:
    """拉取股票池日线；单只失败/空结果跳过（不阻断整体）。"""
    out: dict[str, list[MarketRecord]] = {}
    for sym in symbols:
        try:
            res = service.fetch(sym, start, end, market)
            if res.records:
                out[sym] = res.records
        except Exception:  # noqa: BLE001 — 单只数据缺失跳过
            continue
    return out


def _extract_prices(
    records_map: dict[str, list[MarketRecord]],
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """records_map → (closes_map, volumes_map)。"""
    closes = {s: [r.close for r in recs] for s, recs in records_map.items()}
    volumes = {s: [r.volume for r in recs] for s, recs in records_map.items()}
    return closes, volumes


def _rank_universe(
    ranking: dict[str, str],
    closes_map: dict[str, list[float]],
    volumes_map: dict[str, list[float]],
    service: MarketDataService,
) -> list[tuple[str, float]]:
    """按因子给股票池打分并排序，返回 [(symbol, value)]。"""
    factor = ranking.get("factor", "")
    direction = ranking.get("direction", "desc")
    scored: list[tuple[str, float]] = []
    for sym, closes in closes_map.items():
        if factor == "market_cap":
            value = service.fetch_market_cap(sym)
        else:
            value = factor_value(factor, closes, volumes_map.get(sym))
        if value is not None:
            scored.append((sym, float(value)))
    scored.sort(key=lambda x: x[1], reverse=(direction == "desc"))
    return scored


class CalculateFactor(Tool):
    name = "calculate_factor"
    description = "计算因子值并按方向排序（真实计算，确定性）。"
    input_model = CalculateFactorInput
    output_model = CalculateFactorOutput
    permission = "restricted-compute"

    def __init__(self, service: MarketDataService | None = None) -> None:
        self._service = service or build_default_service()

    def execute(self, args: CalculateFactorInput) -> CalculateFactorOutput:
        records_map = _fetch_records(
            self._service, args.universe, args.start, args.end, args.market
        )
        closes, volumes = _extract_prices(records_map)
        ranking = _rank_universe(
            {"factor": args.factor, "direction": args.direction},
            closes,
            volumes,
            self._service,
        )
        return CalculateFactorOutput(
            factor=args.factor,
            direction=args.direction,
            ranking=[FactorScore(symbol=s, value=v) for s, v in ranking],
        )


class ScreenStocks(Tool):
    name = "screen_stocks"
    description = "按因子排序选 top_n 只股票（真实选股）。"
    input_model = ScreenStocksInput
    output_model = ScreenStocksOutput
    permission = "restricted-compute"

    def __init__(self, service: MarketDataService | None = None) -> None:
        self._service = service or build_default_service()

    def execute(self, args: ScreenStocksInput) -> ScreenStocksOutput:
        factor = args.factor or "momentum_20d"
        records_map = _fetch_records(
            self._service, args.universe, args.start, args.end, args.market
        )
        closes, volumes = _extract_prices(records_map)
        ranking = _rank_universe(
            {"factor": factor, "direction": args.direction},
            closes,
            volumes,
            self._service,
        )
        return ScreenStocksOutput(selected=[s for s, _ in ranking[: args.top_n]])


class RunBacktest(Tool):
    name = "run_backtest"
    description = "按策略配置跑历史回测（真实计算，确定性，防未来函数）。"
    input_model = RunBacktestInput
    output_model = RunBacktestOutput
    permission = "restricted-compute"

    def __init__(self, service: MarketDataService | None = None) -> None:
        self._service = service or build_default_service()

    def _records_from_snapshot(self, snapshot_id: str) -> dict[str, list[MarketRecord]]:
        """按 snapshot_id 取回 research 存下的行情快照，还原成 {symbol: [MarketRecord]}；无则返回空。"""
        if not snapshot_id:
            return {}
        from ..store import load_market_snapshot

        payload = load_market_snapshot(snapshot_id)
        if not payload:
            return {}
        out: dict[str, list[MarketRecord]] = {}
        for r in payload.get("records", []):
            rec = MarketRecord(**r)
            out.setdefault(rec.symbol, []).append(rec)
        return out

    def execute(self, args: RunBacktestInput) -> RunBacktestOutput:
        config = args.strategy_config
        ref = f"bt_{uuid.uuid4().hex[:8]}"
        lookback = max(int(config.lookback_days), 1)
        try:
            # 优先复用 research 已经拉好的行情快照（按 snapshot_id 取），取不到再重新拉
            records_map = self._records_from_snapshot(args.snapshot_id)
            if not records_map:
                records_map = _fetch_records(
                    self._service, config.universe, args.start, args.end, config.market
                )
            # 信号窗 / 持有窗分离：因子只用前 lookback 个交易日，回测只用其后（t+1 起）
            signal_closes: dict[str, list[float]] = {}
            signal_volumes: dict[str, list[float]] = {}
            holding_closes: dict[str, list[float]] = {}
            holding_opens: dict[str, list[float]] = {}
            holding_dates: list[str] = []
            signal_end_date = ""
            for sym, records in records_map.items():
                if len(records) <= lookback:
                    continue  # 不足一个信号窗 + 至少一个持有日
                signal = records[:lookback]
                holding = records[lookback:]
                if not holding_dates:
                    holding_dates = [r.date for r in holding]
                    signal_end_date = signal[-1].date  # 信号末日（t0，曲线起点）
                signal_closes[sym] = [r.close for r in signal]
                signal_volumes[sym] = [r.volume for r in signal]
                holding_closes[sym] = [r.close for r in holding]
                holding_opens[sym] = [r.open for r in holding]

            ranking = _rank_universe(config.ranking, signal_closes, signal_volumes, self._service)
            selected = [s for s, _ in ranking[: config.top_n]]
            c_closes = {s: holding_closes[s] for s in selected if s in holding_closes}
            c_opens = {s: holding_opens[s] for s in selected if s in holding_opens}

            m_default = run_backtest_metrics(c_closes, c_opens, cost=config.cost)
            m_0x = run_backtest_metrics(c_closes, c_opens, cost=0.0)
            m_2x = run_backtest_metrics(c_closes, c_opens, cost=config.cost * 2)
            scenarios = {"0x": m_0x, "default": m_default, "2x": m_2x}

            # 收益曲线（真实数据）：策略=选中股票等权；基准=全股票池等权买入持有
            strategy = run_backtest_series(c_closes, c_opens, cost=config.cost)
            benchmark = run_backtest_series(holding_closes, holding_opens, cost=0.0)
            curve_dates = ([signal_end_date] + holding_dates)[: len(strategy["cumulative"])]
            # 个股曲线：每只选中股票单独的累计收益（同一套 t+1 开盘首日口径）
            stocks = []
            for sym in selected:
                if sym in c_closes and sym in c_opens and len(c_closes[sym]) >= 2:
                    one = run_backtest_series({sym: c_closes[sym]}, {sym: c_opens[sym]}, cost=0.0)
                    stocks.append({"symbol": sym, "cumulative": one["cumulative"]})
            series = {
                "dates": curve_dates,
                "cumulative": strategy["cumulative"],
                "drawdown": strategy["drawdown"],
                "benchmark": benchmark["cumulative"][: len(strategy["cumulative"])],
                "stocks": stocks,
            }

            save_result(
                ref,
                {
                    "metrics": m_default.model_dump(),
                    "cost_scenarios": {k: v.model_dump() for k, v in scenarios.items()},
                    "config": config.model_dump(),
                    "selected": selected,
                    "lookback_days": lookback,
                    "start": args.start,
                    "end": args.end,
                    "series": series,
                    # 数据快照：持有窗实际用到的收盘序列（可复现）
                    "data_snapshot": {"holding_closes": c_closes},
                },
            )
            return RunBacktestOutput(ref=ref, metrics=m_default, cost_scenarios=scenarios)
        except Exception as exc:  # noqa: BLE001 — 回测失败保存错误与配置，不伪造指标
            save_result(
                ref,
                {
                    "error": str(exc),
                    "config": config.model_dump(),
                    "start": args.start,
                    "end": args.end,
                },
            )
            return RunBacktestOutput(ref=ref, metrics=BacktestMetrics())


class GetBacktestResult(Tool):
    name = "get_backtest_result"
    description = "按引用 ID 取回测结果（真实存储）。"
    input_model = GetBacktestResultInput
    output_model = RunBacktestOutput
    permission = "restricted-compute"

    def execute(self, args: GetBacktestResultInput) -> RunBacktestOutput:
        payload = load_result(args.ref)
        if payload is None:
            return RunBacktestOutput(ref=args.ref, metrics=BacktestMetrics())
        metrics = BacktestMetrics(**payload.get("metrics", {}))
        scenarios = {
            k: BacktestMetrics(**v) for k, v in payload.get("cost_scenarios", {}).items()
        }
        return RunBacktestOutput(ref=args.ref, metrics=metrics, cost_scenarios=scenarios)
