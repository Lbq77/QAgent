"""工具输入/输出 Schema 与策略配置模型（Day 3）。

每个工具一个 input_model + output_model；quant 的 strategy_config 在这里
落成 StrategyConfig 模型，作为 run_backtest / audit_strategy 的校验契约，
呼应 nodes 注释「JSON 配置 → Schema 校验」。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..state import FactCheckResult, RiskReport


class StrategyConfig(BaseModel):
    """选股/回测策略配置（JSON 由程序校验后确定性执行）。"""

    market: str = "US"
    universe: list[str] = Field(default_factory=list)
    ranking: dict[str, str] = Field(default_factory=dict)  # {"factor": ..., "direction": ...}
    top_n: int = 3
    cost: float = 0.001
    t_plus_1: bool = True


# ---- research 工具 ----


class GetMarketDataInput(BaseModel):
    symbols: list[str] = Field(default_factory=list)
    start: str = ""
    end: str = ""
    market: str = "US"


class MarketRecord(BaseModel):
    symbol: str
    date: str
    close: float


class GetMarketDataOutput(BaseModel):
    snapshot_id: str
    source: str = ""
    records: list[MarketRecord] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)


class SearchFinancialKnowledgeInput(BaseModel):
    query: str


class KnowledgeItem(BaseModel):
    title: str
    url: str
    snippet: str = ""


class SearchFinancialKnowledgeOutput(BaseModel):
    matches: list[KnowledgeItem] = Field(default_factory=list)


class GetCompanyProfileInput(BaseModel):
    symbol: str


class GetCompanyProfileOutput(BaseModel):
    symbol: str
    name: str = ""
    sector: str = ""
    exchange: str = ""


# ---- quant 工具 ----


class CalculateFactorInput(BaseModel):
    factor: str
    direction: str = "desc"
    universe: list[str] = Field(default_factory=list)
    start: str = ""
    end: str = ""


class FactorScore(BaseModel):
    symbol: str
    value: float


class CalculateFactorOutput(BaseModel):
    factor: str
    direction: str
    ranking: list[FactorScore] = Field(default_factory=list)


class ScreenStocksInput(BaseModel):
    universe: list[str] = Field(default_factory=list)
    top_n: int = 3


class ScreenStocksOutput(BaseModel):
    selected: list[str] = Field(default_factory=list)


class RunBacktestInput(BaseModel):
    strategy_config: StrategyConfig


class BacktestMetrics(BaseModel):
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0


class RunBacktestOutput(BaseModel):
    ref: str
    metrics: BacktestMetrics = Field(default_factory=BacktestMetrics)


class GetBacktestResultInput(BaseModel):
    ref: str


# ---- risk 工具 ----


class AuditStrategyInput(BaseModel):
    strategy_config: StrategyConfig
    backtest_result_ref: str = ""


class ValidateReportClaimsInput(BaseModel):
    claims: list[str] = Field(default_factory=list)
    backtest_result_ref: str = ""
