"""Yahoo Finance 数据源：用 yfinance 库，不再手搓 HTTP。

yfinance 内部处理了 Yahoo 的 crumb/cookie、限流重试与错误码，比直接请求更稳；
close 用复权价（auto_adjust=True），下游动量/回测需要的就是复权序列。
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf

from ..tools.schemas import MarketRecord
from .base import MarketDataError, MarketDataProvider


class YahooProvider(MarketDataProvider):
    name = "yfinance"

    def fetch(self, symbol: str, start: str, end: str, market: str = "US") -> list[MarketRecord]:
        try:
            df = yf.Ticker(symbol).history(start=start, end=end, interval="1d", auto_adjust=True)
        except Exception as exc:  # noqa: BLE001 — 数据源异常统一交给服务层降级
            raise MarketDataError(f"yfinance: {exc}") from exc
        return _df_to_records(df, symbol)

    def fetch_market_cap(self, symbol: str) -> float | None:
        """取市值（美元）；失败/缺失返回 None（服务层继续降级）。"""
        try:
            cap = None
            try:
                cap = yf.Ticker(symbol).fast_info["market_cap"]
            except Exception:  # noqa: BLE001 — fast_info 缺字段/异常时回退 info
                cap = None
            if cap is None:
                cap = (yf.Ticker(symbol).info or {}).get("marketCap")
            return float(cap) if cap else None
        except Exception:  # noqa: BLE001
            return None


def fetch_company_profile(symbol: str) -> dict[str, str]:
    """用 yfinance ``Ticker.info`` 取公司概况：name / sector / exchange。

    字段缺失时返回空字符串；网络/数据源异常由上层工具兜底（不在这里吞）。
    """
    info = yf.Ticker(symbol).info or {}
    return {
        "name": str(info.get("longName") or info.get("shortName") or ""),
        "sector": str(info.get("sector") or ""),
        "exchange": str(info.get("fullExchangeName") or info.get("exchange") or ""),
    }


def _df_to_records(df: pd.DataFrame, symbol: str) -> list[MarketRecord]:
    """把 yfinance 返回的日线 DataFrame 转成 MarketRecord 列表（NaN 行跳过）。"""
    if df is None or df.empty:
        return []
    records: list[MarketRecord] = []
    for idx, row in df.iterrows():
        close = row.get("Close")
        if close is None or pd.isna(close):
            continue
        records.append(
            MarketRecord(
                symbol=symbol,
                date=idx.strftime("%Y-%m-%d"),
                close=float(close),
                open=float(row.get("Open") or 0.0),
                volume=float(row.get("Volume") or 0.0),
            )
        )
    return records
