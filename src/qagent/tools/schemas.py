"""工具输入/输出 Schema 与策略配置模型。

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
    # 因子信号窗长度（交易日）：只用前 lookback 个交易日算因子，其后才是回测持有窗
    lookback_days: int = 60


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
    open: float = 0.0
    volume: float = 0.0


class GetMarketDataOutput(BaseModel):
    snapshot_id: str
    source: str = ""
    records: list[MarketRecord] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)


class SearchFinancialKnowledgeInput(BaseModel):
    query: str
    mode: str = "global"  # local | global


class KnowledgeItem(BaseModel):
    title: str
    url: str
    snippet: str = ""


class SearchFinancialKnowledgeOutput(BaseModel):
    answer: str = ""
    matches: list[KnowledgeItem] = Field(default_factory=list)


class GetCompanyProfileInput(BaseModel):
    symbol: str


class GetCompanyProfileOutput(BaseModel):
    symbol: str
    name: str = ""
    sector: str = ""
    exchange: str = ""


class SearchNewsInput(BaseModel):
    query: str


class NewsItem(BaseModel):
    title: str
    url: str = ""
    snippet: str = ""
    source: str = ""


class SearchNewsOutput(BaseModel):
    query: str = ""
    items: list[NewsItem] = Field(default_factory=list)


# ---- quant 工具 ----


class CalculateFactorInput(BaseModel):
    factor: str
    direction: str = "desc"
    universe: list[str] = Field(default_factory=list)
    start: str = ""
    end: str = ""
    market: str = "US"


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
    factor: str = ""
    direction: str = "desc"
    start: str = ""
    end: str = ""
    market: str = "US"


class ScreenStocksOutput(BaseModel):
    selected: list[str] = Field(default_factory=list)


class RunBacktestInput(BaseModel):
    strategy_config: StrategyConfig
    start: str = ""
    end: str = ""
    snapshot_id: str = ""  # 行情快照 id；有则优先取快照，无则重新拉取


class BacktestMetrics(BaseModel):
    annual_return: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0
    var_95: float = 0.0  # 95% 置信度单日 VaR（正数，损失幅度：0.03 = 单日最多亏 3%）
    annual_volatility: float = 0.0  # 年化波动率（小数：0.25 = 25%）


class RunBacktestOutput(BaseModel):
    ref: str
    metrics: BacktestMetrics = Field(default_factory=BacktestMetrics)
    # 三档成本对比（0 倍 / 默认 / 2 倍）
    cost_scenarios: dict[str, BacktestMetrics] = Field(default_factory=dict)


class GetBacktestResultInput(BaseModel):
    ref: str


# ---- risk 工具 ----


class AuditStrategyInput(BaseModel):
    strategy_config: StrategyConfig
    backtest_result_ref: str = ""


class Claim(BaseModel):
    """报告里的一条可核查断言（数字核查）。

    field：要核查的字段（annual_return / max_drawdown / sharpe …）
    value：报告声称的值；source_ref：支撑这条断言的证据（回测 ref / 快照 id）
    """

    field: str
    value: float = 0.0
    source_ref: str = ""


class ValidateReportClaimsInput(BaseModel):
    claims: list[Claim] = Field(default_factory=list)
    backtest_result_ref: str = ""
