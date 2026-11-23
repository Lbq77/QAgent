"""新闻 / 舆情数据源。

默认桩（离线可跑通，结果如实标记 [stub]）；配 Tavily API key 后接真实新闻搜索
（Tavily 有免费额度）。检索词由 research 节点的 ReAct 循环让模型自己写，但数据
只做只读检索、结果作为证据引用，不进入回测信号，不破坏「数字来自确定性工具」。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from ..tools.schemas import NewsItem


class NewsError(Exception):
    """新闻检索失败。"""


class NewsProvider(ABC):
    """单一新闻源接口：search 返回结构化新闻条目；失败抛 NewsError 或返回空列表。"""

    name: ClassVar[str]

    @abstractmethod
    def search(self, query: str) -> list[NewsItem]:
        ...


class StubNewsProvider(NewsProvider):
    """确定性离线桩：返回一条明确标记 [stub] 的条目，不伪装成真实新闻。"""

    name = "stub"

    def search(self, query: str) -> list[NewsItem]:
        return [NewsItem(title=f"[stub] 新闻检索：{query}", url="stub://news/1", source="stub")]


class TavilyNewsProvider(NewsProvider):
    """Tavily 新闻搜索（https://api.tavily.com/search，免费额度）。

    用 requests 直连，不引入额外 SDK；结果截断 snippet 避免 State/报告膨胀。
    """

    name = "tavily"

    def __init__(self, api_key: str, max_results: int = 5) -> None:
        self._api_key = api_key
        self._max_results = max_results

    def search(self, query: str) -> list[NewsItem]:
        import requests  # 惰性导入：默认桩无需该依赖

        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": self._api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": self._max_results,
                "include_answer": False,
            },
            timeout=15,
        )
        resp.raise_for_status()
        items: list[NewsItem] = []
        for r in resp.json().get("results", []):
            items.append(
                NewsItem(
                    title=str(r.get("title") or ""),
                    url=str(r.get("url") or ""),
                    snippet=str(r.get("content") or "")[:500],
                    source="tavily",
                )
            )
        return items


def build_default_news_client() -> NewsProvider | None:
    """按配置构建新闻源；未配 Tavily key 时返回 None（工具回退到桩）。"""
    from ..config import get_settings

    s = get_settings()
    if not s.tavily_api_key:
        return None
    return TavilyNewsProvider(s.tavily_api_key)
