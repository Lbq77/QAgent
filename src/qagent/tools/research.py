"""Research 工具。

get_market_data 接真实行情（Yahoo，多 host 降级）；
search_financial_knowledge 接 GraphRAG（本地 graphrag.api 查询，未启用时回退为桩）；
get_company_profile 接 yfinance 公司概况（名称/行业/交易所）。
"""
from __future__ import annotations

import uuid
from datetime import date, timedelta

from ..data import (
    GraphRAGClient,
    MarketDataError,
    MarketDataService,
    NewsProvider,
    build_default_graphrag_client,
    build_default_news_client,
    build_default_service,
)
from ..data.yahoo import fetch_company_profile
from .base import Tool
from .schemas import (
    GetCompanyProfileInput,
    GetCompanyProfileOutput,
    GetMarketDataInput,
    GetMarketDataOutput,
    KnowledgeItem,
    NewsItem,
    SearchFinancialKnowledgeInput,
    SearchFinancialKnowledgeOutput,
    SearchNewsInput,
    SearchNewsOutput,
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

        snapshot_id = f"snap_{uuid.uuid4().hex[:8]}"
        # 行情快照外置：完整 records 按 snapshot_id 存库，下游 quant 按 id 取回，避免重复拉取
        from ..store import save_market_snapshot

        save_market_snapshot(
            snapshot_id,
            {
                "source": "+".join(sorted(sources)),
                "records": [r.model_dump() for r in records],
                "missing": missing,
            },
        )
        return GetMarketDataOutput(
            snapshot_id=snapshot_id,
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
    description = "检索金融知识库（GraphRAG，local/global；未启用时回退为桩）。"
    input_model = SearchFinancialKnowledgeInput
    output_model = SearchFinancialKnowledgeOutput
    permission = "read-only"

    def __init__(self, client: GraphRAGClient | None = None) -> None:
        # client 显式传入用于测试注入；None 时按配置构建（未启用则为 None → 桩）
        self._client = client or build_default_graphrag_client()

    def execute(self, args: SearchFinancialKnowledgeInput) -> SearchFinancialKnowledgeOutput:
        if self._client is None:
            return SearchFinancialKnowledgeOutput(
                answer=f"[stub] 未接入 GraphRAG：{args.query}",
                matches=[KnowledgeItem(title=f"stub: {args.query}", url="stub://kb/1")],
            )
        answer = self._client.search(args.query, args.mode)
        return SearchFinancialKnowledgeOutput(
            answer=answer,
            matches=[KnowledgeItem(title=args.query, url=f"graphrag://{args.mode}", snippet=answer)],
        )


class GetCompanyProfile(Tool):
    name = "get_company_profile"
    description = "查询公司概况（名称/行业/交易所，yfinance）。"
    input_model = GetCompanyProfileInput
    output_model = GetCompanyProfileOutput
    permission = "read-only"

    def __init__(self, fetch_profile=None) -> None:
        # fetch_profile 显式传入用于测试注入；None 时用 yfinance 真实源
        self._fetch = fetch_profile or fetch_company_profile

    def execute(self, args: GetCompanyProfileInput) -> GetCompanyProfileOutput:
        try:
            data = self._fetch(args.symbol)
        except Exception:  # noqa: BLE001 — 网络/数据源异常降级为空字段
            data = {}
        return GetCompanyProfileOutput(
            symbol=args.symbol.upper(),
            name=data.get("name", ""),
            sector=data.get("sector", ""),
            exchange=data.get("exchange", ""),
        )


class SearchNews(Tool):
    name = "search_news"
    description = "检索财经新闻/舆情（默认桩；配 Tavily API key 接真实新闻源）。"
    input_model = SearchNewsInput
    output_model = SearchNewsOutput
    permission = "read-only"

    def __init__(self, client: NewsProvider | None = None) -> None:
        # client 显式传入用于测试注入；None 时按配置构建（未配 key 则为 None → 桩）
        self._client = client or build_default_news_client()

    def execute(self, args: SearchNewsInput) -> SearchNewsOutput:
        if self._client is None:
            # 桩：如实标记，不把「没接入」伪装成真实新闻
            return SearchNewsOutput(
                query=args.query,
                items=[NewsItem(title=f"[stub] 未接入新闻源：{args.query}", url="stub://news/1", source="stub")],
            )
        try:
            items = self._client.search(args.query)
        except Exception:  # noqa: BLE001 — 新闻源异常降级为空结果，由 research 标记「无结果」
            items = []
        return SearchNewsOutput(query=args.query, items=items)
