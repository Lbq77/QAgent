"""行情数据源抽象 + 降级链。

降级策略：MarketDataService 按 providers 顺序逐源尝试，异常或空结果都视为失败继续降级，
全部失败抛 MarketDataError。StubMarketDataProvider 是确定性离线桩（测试/演示用）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

from ..tools.schemas import MarketRecord


class MarketDataError(Exception):
    """所有数据源都失败。"""


@dataclass
class MarketDataResult:
    source: str
    records: list[MarketRecord] = field(default_factory=list)


class MarketDataProvider(ABC):
    """单一数据源接口。fetch 返回日线记录；失败可抛异常或返回空列表（服务层继续降级）。"""

    name: ClassVar[str]

    @abstractmethod
    def fetch(self, symbol: str, start: str, end: str, market: str = "US") -> list[MarketRecord]:
        ...

    def fetch_market_cap(self, symbol: str) -> float | None:
        """市值（基本面数据，非日线）。默认不支持返回 None，具体源可覆盖。"""
        return None


class MarketDataService:
    """按优先级顺序尝试多个 provider，返回首个非空结果。"""

    def __init__(self, providers: list[MarketDataProvider]) -> None:
        if not providers:
            raise ValueError("MarketDataService needs at least one provider")
        self.providers = providers

    def fetch(self, symbol: str, start: str, end: str, market: str = "US") -> MarketDataResult:
        errors: list[str] = []
        for p in self.providers:
            try:
                records = p.fetch(symbol, start, end, market)
                if records:
                    return MarketDataResult(source=p.name, records=records)
                errors.append(f"{p.name}: empty")
            except Exception as exc:  # noqa: BLE001 — 数据源异常统一降级，不外抛
                errors.append(f"{p.name}: {exc}")
        raise MarketDataError(f"all sources failed for {symbol}: " + "; ".join(errors))

    def fetch_market_cap(self, symbol: str) -> float | None:
        """按优先级逐个源取市值，返回首个非 None；都不支持返回 None。"""
        for p in self.providers:
            try:
                cap = p.fetch_market_cap(symbol)
            except Exception:  # noqa: BLE001 — 单个源失败继续降级
                continue
            if cap is not None:
                return cap
        return None


class StubMarketDataProvider(MarketDataProvider):
    """确定性离线桩，与 StubLLM 同哲学：无网络也能跑通。"""

    name = "stub"

    def fetch(self, symbol: str, start: str, end: str, market: str = "US") -> list[MarketRecord]:
        return [MarketRecord(symbol=symbol, date=end or "2024-12-31", close=100.0)]


def build_default_service() -> MarketDataService:
    """默认真实源：yfinance（内部已处理 Yahoo 的限流/重试）。

    后续可在此追加带 key 的源（Alpha Vantage / Tiingo 等）形成更长的降级链。
    """
    from .yahoo import YahooProvider

    return MarketDataService([YahooProvider()])
