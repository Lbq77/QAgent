"""Research 的 ReAct 循环：LLM 每轮选一个无参动作，参数一律由 task_constraints 填充。"""
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


class _SeqLLM:
    """planner 出合法 JSON；research 阶段按给定动作序列逐个吐，耗尽后 done；其余返回报告文案。

    action 为 str 时是工具名或 "done"；为 dict 时原样输出（用于带 reason 的 done）。
    """

    is_stub = False

    def __init__(self, actions: list[dict | str]):
        self._actions = list(actions)
        self._i = 0

    def invoke(self, system: str, user: str) -> str:
        if "任务规划器" in system:
            return json.dumps(_PLAN, ensure_ascii=False)
        if "从下面可用的工具" in system:  # research_react prompt
            if self._i < len(self._actions):
                action = self._actions[self._i]
                self._i += 1
                if isinstance(action, str):
                    return json.dumps({"action": action}, ensure_ascii=False)
                return json.dumps(action, ensure_ascii=False)
            return json.dumps({"action": "done"}, ensure_ascii=False)
        return "该策略表现平稳。本内容仅供研究学习，不构成投资建议。"


def _runtime(llm, market_data) -> AgentRuntime:
    ctx = RuntimeContext(
        settings=Settings(model_provider="stub"),
        llm=llm,
        tool_registry=build_default_registry(market_data=market_data),
    )
    return AgentRuntime(ctx)


def _research_tools(state) -> list[str]:
    node = next(t for t in state.trace if t.node == "research_node")
    return [tc.tool for tc in node.tool_calls]


def test_research_react_selects_tools_across_rounds(monkeypatch):
    from qagent.tools import research

    class _FakeClient:
        def search(self, query: str, method: str = "global") -> str:
            return "fake graphrag answer"

    monkeypatch.setattr(research, "build_default_graphrag_client", lambda: _FakeClient())

    llm = _SeqLLM(["search_financial_knowledge", "get_market_data"])
    rt = _runtime(llm, MarketDataService([StubMarketDataProvider()]))
    s = rt.run(QUERY)
    tools = _research_tools(s)
    assert "search_financial_knowledge" in tools
    assert "get_market_data" in tools


def test_research_stub_defaults_to_market_data():
    rt = _runtime(StubLLM(), MarketDataService([StubMarketDataProvider()]))
    s = rt.run(QUERY)
    assert _research_tools(s) == ["get_market_data"]


def test_research_params_come_from_constraints_not_llm(monkeypatch):
    """LLM 只能选动作，symbols/日期必须来自 task_constraints，不能是 LLM 编造的值。"""
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
    rt = _runtime(_SeqLLM(["get_market_data"]), MarketDataService([provider]))
    s = rt.run(QUERY)
    assert s.status.value in ("succeeded", "running")
    assert provider.calls and provider.calls[0][0] == "AAPL"
    assert provider.calls[0][1] == "2024-01-01"


def test_research_react_stops_on_repeat_tool():
    """模型重复调同一工具 → 无进展，循环只跑一次就收尾，不会失控。"""
    llm = _SeqLLM(["get_market_data", "get_market_data", "get_market_data"])
    rt = _runtime(llm, MarketDataService([StubMarketDataProvider()]))
    s = rt.run(QUERY)
    assert _research_tools(s) == ["get_market_data"]


def test_research_react_records_unknown_on_empty_result():
    """工具返回空 → 标记未知，不把「没查到」伪装成「查过了」。"""

    class _EmptyProvider(MarketDataProvider):
        name = "empty"

        def fetch(self, symbol: str, start: str, end: str, market: str = "US"):
            return []

    rt = _runtime(_SeqLLM(["get_market_data"]), MarketDataService([_EmptyProvider()]))
    s = rt.run(QUERY)
    assert "未查到/证据不足" in s.research_context
    assert "get_market_data 无结果" in s.research_context


def test_research_react_surfaces_done_reason():
    """模型 done 并给出 reason → reason 作为未知线索写进研究摘要。"""
    llm = _SeqLLM([{"action": "done", "reason": "知识库与行情均未覆盖该标的"}])
    rt = _runtime(llm, MarketDataService([StubMarketDataProvider()]))
    s = rt.run(QUERY)
    assert "知识库与行情均未覆盖该标的" in s.research_context
