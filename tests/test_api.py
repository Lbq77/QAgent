"""FastAPI + SSE 回归测试（TestClient，离线 stub 数据）。"""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from qagent.api import app, set_runtime
from qagent.config import Settings
from qagent.context import RuntimeContext
from qagent.data import MarketDataService, StubMarketDataProvider
from qagent.llm import StubLLM
from qagent.runtime import AgentRuntime
from qagent.tools import build_default_registry

QUERY = "研究 AAPL 在 2024-01-01 到 2024-12-31 的 20 日动量并回测"


def _client() -> TestClient:
    ctx = RuntimeContext(
        settings=Settings(model_provider="stub"),
        llm=StubLLM(),
        tool_registry=build_default_registry(
            market_data=MarketDataService([StubMarketDataProvider()])
        ),
    )
    set_runtime(AgentRuntime(ctx))
    return TestClient(app)


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        event = data = None
        for line in block.split("\n"):
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        if event and data is not None:
            events.append((event, data))
    return events


def test_run_stream_emits_nodes_then_interrupt():
    client = _client()
    resp = client.get("/api/run-stream", params={"query": QUERY})
    assert resp.status_code == 200
    events = _parse_sse(resp.text)

    nodes = [d["node"] for e, d in events if e == "node"]
    interrupts = [d for e, d in events if e == "interrupt"]

    assert "research" in nodes and "quant" in nodes  # 流式进度
    assert len(interrupts) == 1  # 停在 human_confirm
    assert interrupts[0]["task_id"]
    assert "strategy_config" in interrupts[0]


def test_confirm_approve_returns_final_report():
    client = _client()
    resp = client.get("/api/run-stream", params={"query": QUERY})
    interrupts = [d for e, d in _parse_sse(resp.text) if e == "interrupt"]
    tid = interrupts[0]["task_id"]

    r = client.post("/api/confirm", json={"task_id": tid, "action": "approve"})
    assert r.status_code == 200
    s = r.json()
    assert s["status"] == "succeeded"
    assert s["final_report"]


def test_confirm_reject_cancels():
    client = _client()
    resp = client.get("/api/run-stream", params={"query": QUERY})
    tid = [d for e, d in _parse_sse(resp.text) if e == "interrupt"][0]["task_id"]

    s = client.post("/api/confirm", json={"task_id": tid, "action": "reject"}).json()
    assert s["status"] == "cancelled"


def test_confirm_bad_task_returns_400():
    client = _client()
    r = client.post("/api/confirm", json={"task_id": "nope", "action": "approve"})
    assert r.status_code == 400


def test_task_lookup():
    client = _client()
    resp = client.get("/api/run-stream", params={"query": QUERY})
    tid = [d for e, d in _parse_sse(resp.text) if e == "interrupt"][0]["task_id"]

    t = client.get(f"/api/task/{tid}")
    assert t.status_code == 200
    assert t.json()["task_id"] == tid


def test_experiments_endpoint():
    client = _client()
    r = client.get("/api/experiments")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_index_serves_frontend():
    client = _client()
    r = client.get("/")
    assert r.status_code == 200
    assert "QAgent" in r.text


def test_demo_mode_returns_real_series():
    """demo=true 用合成行情，confirm 应返回真实计算的指标 + 收益曲线（供前端画图）。"""
    client = _client()
    resp = client.get("/api/run-stream", params={"query": QUERY, "demo": True})
    interrupts = [d for e, d in _parse_sse(resp.text) if e == "interrupt"]
    assert interrupts  # 合成数据足够，能走到人工确认
    tid = interrupts[0]["task_id"]

    s = client.post("/api/confirm", json={"task_id": tid, "action": "approve", "demo": True}).json()
    assert s["metrics"]["annual_return"] != 0.0
    series = s["series"]
    assert len(series["cumulative"]) > 2  # 有真实曲线可画
    assert len(series["benchmark"]) == len(series["cumulative"])
    assert max(series["drawdown"]) >= 0
    assert len(series["stocks"]) == 3  # 三只个股各自一条曲线
