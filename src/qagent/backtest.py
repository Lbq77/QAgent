"""确定性回测引擎。

用历史收盘价做等权买入持有的向量化回测，输出年化收益 / 最大回撤 / 夏普。
结果按 ref 存入持久化存储（起 SQLite，见 store.py），GetBacktestResult 按 ref 取回；
State 只存 ref，完整结果外置，跨进程可复现。
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .store import get_result_store
from .tools.schemas import BacktestMetrics

ENGINE_VERSION = "vectorized-equal-weight-v0.1"


def run_backtest_series(
    closes_map: dict[str, list[float]],
    opens_map: dict[str, list[float]] | None = None,
    cost: float = 0.001,
) -> dict:
    """等权买入持有的向量化回测（持有窗，已与信号窗分离）。

    除指标外还返回真实的组合日收益序列，供前端画收益曲线/回撤曲线：
    - cumulative：累计净值（首元素 1.0 起点）
    - drawdown：回撤（正数，首元素 0.0）
    - returns：组合日收益
    全部来自实际价格，不编造。
    """
    if not closes_map:
        return {
            "metrics": BacktestMetrics(),
            "returns": [],
            "cumulative": [],
            "drawdown": [],
        }

    close_df = pd.DataFrame({s: pd.Series(v) for s, v in closes_map.items()})
    if opens_map:
        open_df = pd.DataFrame(
            {s: pd.Series(v) for s, v in opens_map.items()}
        ).reindex(columns=close_df.columns)
        entry = close_df.iloc[0] / open_df.iloc[0] - 1.0  # t+1 开盘 → 当日收盘
        rest = close_df.pct_change().iloc[1:]
        rets = pd.concat([entry.to_frame().T, rest])
    else:
        rets = close_df.pct_change().iloc[1:]

    port = rets.mean(axis=1).dropna()
    if len(port) < 1:
        return {
            "metrics": BacktestMetrics(),
            "returns": [],
            "cumulative": [],
            "drawdown": [],
        }

    total_return = float((1.0 + port).prod() - 1.0) - cost
    n = len(port)
    annual_return = total_return * (252.0 / n)
    cum = (1.0 + port).cumprod()
    drawdown = 1.0 - cum / cum.cummax()
    max_drawdown = float(drawdown.max())
    std = float(port.std(ddof=1))
    sharpe = float(port.mean() / std * math.sqrt(252.0)) if std > 0 else 0.0

    metrics = BacktestMetrics(
        annual_return=round(annual_return, 6),
        total_return=round(total_return, 6),
        max_drawdown=round(max_drawdown, 6),
        sharpe=round(sharpe, 6),
    )
    return {
        "metrics": metrics,
        "returns": [round(float(x), 6) for x in port],
        "cumulative": [1.0] + [round(float(x), 6) for x in cum],
        "drawdown": [0.0] + [round(float(x), 6) for x in drawdown],
    }


def run_backtest_metrics(
    closes_map: dict[str, list[float]],
    opens_map: dict[str, list[float]] | None = None,
    cost: float = 0.001,
) -> BacktestMetrics:
    """仅返回指标（保持旧接口兼容）。"""
    return run_backtest_series(closes_map, opens_map, cost)["metrics"]


def save_result(ref: str, payload: dict[str, Any]) -> None:
    """保存回测产物；自动附带引擎版本与时间戳。"""
    record = dict(payload)
    record.setdefault("engine_version", ENGINE_VERSION)
    record.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    get_result_store().put(ref, record)


def load_result(ref: str) -> dict[str, Any] | None:
    """按 ref 取回回测产物；不存在返回 None。"""
    return get_result_store().get(ref)


def clear_results() -> None:
    """清空存储（测试用）。"""
    get_result_store().clear()
