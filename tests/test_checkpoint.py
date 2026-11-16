"""SQLite Checkpoint Saver 持久化 + 跨进程断点恢复。"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from qagent.checkpoint import SqliteSaver
from qagent.config import Settings
from qagent.context import RuntimeContext
from qagent.data import MarketDataService, StubMarketDataProvider
from qagent.llm import StubLLM
from qagent.runtime import AgentRuntime
from qagent.tools import build_default_registry


class _MiniState(TypedDict):
    answer: str


def _mini_graph(db_path: str):
    def node(state: _MiniState):
        ans = interrupt("confirm?")
        return {"answer": ans}

    g = StateGraph(_MiniState)
    g.add_node("n", node)
    g.add_edge(START, "n")
    g.add_edge("n", END)
    return g.compile(checkpointer=SqliteSaver(str(db_path)))


def test_sqlite_saver_persists_across_instances(tmp_path):
    db = tmp_path / "cp.sqlite"
    cfg = {"configurable": {"thread_id": "t1"}}

    app = _mini_graph(db)
    r1 = app.invoke({"answer": ""}, config=cfg)
    assert "__interrupt__" in r1  # 挂起

    # 新实例（新连接）读同一文件，能找到挂起 checkpoint 并恢复
    app2 = _mini_graph(db)
    r2 = app2.invoke(Command(resume="approved"), config=cfg)
    assert r2 == {"answer": "approved"}


def test_sqlite_saver_get_tuple_roundtrip(tmp_path):
    db = tmp_path / "cp.sqlite"
    cfg = {"configurable": {"thread_id": "t2"}}

    app = _mini_graph(db)
    app.invoke({"answer": ""}, config=cfg)

    saver = SqliteSaver(str(db))
    tup = saver.get_tuple(cfg)
    assert tup is not None
    assert tup.config["configurable"]["thread_id"] == "t2"
    saver.close()


def test_cross_process_confirm_recovery(tmp_path):
    """run 停在一个进程，全新 runtime（新进程）用同一 sqlite 恢复人工确认。"""
    db = tmp_path / "cp.sqlite"

    def make_runtime() -> AgentRuntime:
        ctx = RuntimeContext(
            settings=Settings(
                model_provider="stub",
                checkpoint_backend="sqlite",
                checkpoint_db=db,
            ),
            llm=StubLLM(),
            tool_registry=build_default_registry(
                market_data=MarketDataService([StubMarketDataProvider()])
            ),
        )
        return AgentRuntime(ctx)

    rt1 = make_runtime()
    s = rt1.run("研究 AAPL 在 2024-01-01 到 2024-12-31 的 20 日动量并回测")
    assert s.awaiting_confirmation is True
    tid = s.task_id

    # 模拟新进程：全新 runtime，同一个 sqlite 文件
    rt2 = make_runtime()
    final = rt2.confirm(tid, {"action": "approve"})
    assert final.status.value == "succeeded"
    assert final.task_id == tid
    assert "report_node" in [t.node for t in final.trace]
