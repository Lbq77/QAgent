"""Research 的工具路由：LLM 只选工具名，参数一律由 task_constraints 填充。"""
from __future__ import annotations

import json

from qagent.config import Settings
from qagent.context import RuntimeContext
from qagent.data import MarketDataService, StubMarketDataProvider
from qagent.data.base import MarketDataProvider, MarketRecord
from qagent.llm import StubLLM
from qagent.runtime import AgentRuntime
from qagent.tools import build_default_registry

QUERY = "研究 AAPL 在 2024-01-01 到 2024-12-31 的 20 日动量并回测"

_PLAN = {
    "goal": "研究 AAPL 动量并回测",
    "steps": ["research", "quant", "risk_audit", "fact_check", "report"],
    "needs_research": True,
    "needs_backtest": True,
    "needs_risk_audit": True,
    "missing_params": [],
    "market": "US",
    "universe": ["AAPL"],
    "date_range": {"start": "2024-01-01", "end": "2024-12-31"},
    "factor": "momentum_20d",
    "factor_direction": "desc",
    "top_n": 3,
}


class _RouteLLM:
    """planner 出合法 JSON；research 路由阶段自选「知识库 + 行情」两个工具名。"""

    is_stub = False

    def invoke(self, system: str, user: str) -> str:
        if "任务规划器" in system:
            return json.dumps(_PLAN, ensure_ascii=False)
        if "从下面可用的工具" in system:  # research_route prompt
            return json.dumps({"tools": ["search_financial_knowledge", "get_market_data"]}, ensure_ascii=False)
        return "该策略表现平稳。本内容仅供研究学习，不构成投资建议。"


def _runtime(llm, market_data) -> AgentRuntime:
    ctx = RuntimeContext(
        settings=Settings(model_provider="stub"),
        llm=llm,
        tool_registry=build_default_registry(market_data=market_data),
    )
    return AgentRuntime(ctx)


def test_research_route_selects_tools(monkeypatch):
    from qagent.tools import research

    class _FakeClient:
        def search(self, query: str, method: str = "global") -> str:
            return "fake graphrag answer"

    monkeypatch.setattr(research, "build_default_graphrag_client", lambda: _FakeClient())

    rt = _runtime(_RouteLLM(), MarketDataService([StubMarketDataProvider()]))
    s = rt.run(QUERY)
    node = next(t for t in s.trace if t.node == "research_node")
    tools = [tc.tool for tc in node.tool_calls]
    assert "search_financial_knowledge" in tools
    assert "get_market_data" in tools


def test_research_stub_defaults_to_market_data():
    rt = _runtime(StubLLM(), MarketDataService([StubMarketDataProvider()]))
    s = rt.run(QUERY)
    node = next(t for t in s.trace if t.node == "research_node")
    assert [tc.tool for tc in node.tool_calls] == ["get_market_data"]


def test_research_params_come_from_constraints_not_llm(monkeypatch):
    """LLM 只能选工具名，symbols/日期必须来自 task_constraints，不能是 LLM 编造的值。"""
    from qagent.tools import research

    monkeypatch.setattr(research, "build_default_graphrag_client", lambda: None)

    class _RecordingProvider(MarketDataProvider):
        name = "rec"

        def __init__(self):
            self.calls = []

        def fetch(self, symbol: str, start: str, end: str, market: str = "US"):
            self.calls.append((symbol, start, end, market))
            return [MarketRecord(symbol=symbol, date=end or "2024-12-31", close=100.0, open=100.0)]

    provider = _RecordingProvider()
    rt = _runtime(_RouteLLM(), MarketDataService([provider]))
    s = rt.run(QUERY)
    assert s.status.value in ("succeeded", "running")  # 至少跑到了 research 之后
    # 拉行情用的必须是 planner 给出的 AAPL，不是任何 LLM 编造的代码
    assert provider.calls and provider.calls[0][0] == "AAPL"
    assert provider.calls[0][1] == "2024-01-01"
