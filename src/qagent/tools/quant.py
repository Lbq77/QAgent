"""Quant 工具（Day 3：确定性桩，Day 5–6 接因子/回测真实计算）。"""
from __future__ import annotations

import uuid

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


class CalculateFactor(Tool):
    name = "calculate_factor"
    description = "计算因子值并按方向排序（Day 5 接真实计算，当前确定性桩）。"
    input_model = CalculateFactorInput
    output_model = CalculateFactorOutput
    permission = "restricted-compute"

    def execute(self, args: CalculateFactorInput) -> CalculateFactorOutput:
        ranking = [
            FactorScore(symbol=s, value=float(len(args.universe) - i))
            for i, s in enumerate(args.universe)
        ]
        return CalculateFactorOutput(
            factor=args.factor, direction=args.direction, ranking=ranking
        )


class ScreenStocks(Tool):
    name = "screen_stocks"
    description = "按 top_n 筛选股票（Day 5 接真实选股，当前确定性桩）。"
    input_model = ScreenStocksInput
    output_model = ScreenStocksOutput
    permission = "restricted-compute"

    def execute(self, args: ScreenStocksInput) -> ScreenStocksOutput:
        return ScreenStocksOutput(selected=args.universe[: args.top_n])


class RunBacktest(Tool):
    name = "run_backtest"
    description = "按策略配置跑历史回测（Day 6 接真实回测，当前确定性桩）。"
    input_model = RunBacktestInput
    output_model = RunBacktestOutput
    permission = "restricted-compute"

    def execute(self, args: RunBacktestInput) -> RunBacktestOutput:
        return RunBacktestOutput(ref=f"bt_{uuid.uuid4().hex[:8]}", metrics=BacktestMetrics())


class GetBacktestResult(Tool):
    name = "get_backtest_result"
    description = "按引用 ID 取回测结果（Day 6 接真实存储，当前确定性桩）。"
    input_model = GetBacktestResultInput
    output_model = RunBacktestOutput
    permission = "restricted-compute"

    def execute(self, args: GetBacktestResultInput) -> RunBacktestOutput:
        return RunBacktestOutput(ref=args.ref, metrics=BacktestMetrics())
