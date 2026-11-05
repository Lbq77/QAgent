"""Research 工具。

get_market_data 在 Day 4 接真实行情（Yahoo chart API，多 host 降级）；
search_financial_knowledge / get_company_profile 仍为确定性桩（Day 4 后半段接 GraphRAG）。
"""
from __future__ import annotations

import uuid
from datetime import date, timedelta

from ..data import MarketDataError, MarketDataService, build_default_service
from .base import Tool
from .schemas import (
    GetCompanyProfileInput,
    GetCompanyProfileOutput,
    GetMarketDataInput,
    GetMarketDataOutput,
    KnowledgeItem,
    SearchFinancialKnowledgeInput,
    SearchFinancialKnowledgeOutput,
)


class GetMarketData(Tool):
    name = "get_market_data"
    description = "拉取股票日线行情（Yahoo chart API，多 host 降级）。"
    input_model = GetMarketDataInput
    output_model = GetMarketDataOutput
    permission = "read-only"

    def __init__(self, service: MarketDataService | None = None) -> None:
        self._service = service or build_default_service()

    def execute(self, args: GetMarketDataInput) -> GetMarketDataOutput:
        start, end = _normalize_range(args.start, args.end)
        records = []
        missing = []
        sources: set[str] = set()
        for sym in args.symbols:
            try:
                res = self._service.fetch(sym, start, end, args.market)
                records.extend(res.records)
                sources.add(res.source)
            except MarketDataError:
                missing.append(sym)
        return GetMarketDataOutput(
            snapshot_id=f"snap_{uuid.uuid4().hex[:8]}",
            source="+".join(sorted(sources)),
            records=records,
            missing=missing,
        )


def _normalize_range(start: str, end: str) -> tuple[str, str]:
    """空日期兜底：end 默认今天，start 默认一年前（research-only 查询没有日期时也能拉到数据）。"""
    today = date.today()
    end = end or today.isoformat()
    start = start or (today - timedelta(days=365)).isoformat()
    return start, end


class SearchFinancialKnowledge(Tool):
    name = "search_financial_knowledge"
    description = "检索金融知识库（Day 4 接 GraphRAG，当前为确定性桩）。"
    input_model = SearchFinancialKnowledgeInput
    output_model = SearchFinancialKnowledgeOutput
    permission = "read-only"

    def execute(self, args: SearchFinancialKnowledgeInput) -> SearchFinancialKnowledgeOutput:
        return SearchFinancialKnowledgeOutput(
            matches=[KnowledgeItem(title=f"stub: {args.query}", url="stub://kb/1")]
        )


class GetCompanyProfile(Tool):
    name = "get_company_profile"
    description = "查询公司概况（Day 4 接真实数据，当前为确定性桩）。"
    input_model = GetCompanyProfileInput
    output_model = GetCompanyProfileOutput
    permission = "read-only"

    def execute(self, args: GetCompanyProfileInput) -> GetCompanyProfileOutput:
        return GetCompanyProfileOutput(symbol=args.symbol.upper(), name=args.symbol.upper())
