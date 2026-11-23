"""行情数据源与降级链。

免费无 key 源经实测：Yahoo Finance chart API 可用（requests + UA 头），
stooq 已加 JS 反爬、纯 HTTP 拿不到 CSV。故 主用 Yahoo（query1），
降级到 Yahoo query2 备用 host；后续可再按同样接口加带 key 的源。
"""
from .base import (
    MarketDataError,
    MarketDataProvider,
    MarketDataResult,
    MarketDataService,
    StubMarketDataProvider,
    build_default_service,
)
from .graphrag import GraphRAGClient, GraphRAGError, build_default_graphrag_client
from .news import (
    NewsError,
    NewsProvider,
    StubNewsProvider,
    TavilyNewsProvider,
    build_default_news_client,
)

__all__ = [
    "MarketDataError",
    "MarketDataProvider",
    "MarketDataResult",
    "MarketDataService",
    "StubMarketDataProvider",
    "build_default_service",
    "GraphRAGClient",
    "GraphRAGError",
    "build_default_graphrag_client",
    "NewsError",
    "NewsProvider",
    "StubNewsProvider",
    "TavilyNewsProvider",
    "build_default_news_client",
]
