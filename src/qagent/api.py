"""FastAPI + SSE 服务：把 Agent Runtime 暴露成可演示的 Web 服务。

- GET  /api/run-stream?query=...   SSE 流式跑图（node 进度 → interrupt → 结束）
- POST /api/confirm                人工确认（approve/modify/reject），返回最终状态
- GET  /api/task/{task_id}         查任务当前状态
- GET  /api/experiments            实验记录（list_experiments）
- GET  /                           前端页面（web/index.html）

生产用 `uvicorn qagent.api:app` 启动（离线环境已含 fastapi/httpx，缺 uvicorn 时用
TestClient 跑测试、加装 uvicorn 即可启动服务）。
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .backtest import load_result
from .runtime import AgentRuntime
from .state import AgentState

app = FastAPI(title="QAgent")

WEB_DIR = Path(__file__).parent / "web"

_runtime: AgentRuntime | None = None
_demo_runtime: AgentRuntime | None = None


class _DemoProvider:
    """离线合成行情（确定性）：给任意 symbol 生成 100 根带趋势的日线，用于演示收益曲线。

    明确标注为「合成数据」，不冒充真实行情。
    """

    name = "demo"

    def fetch(self, symbol: str, start: str, end: str, market: str = "US"):
        import hashlib
        import math
        from datetime import date, timedelta

        from .data.base import MarketRecord

        h = int(hashlib.md5(symbol.encode()).hexdigest(), 16)
        drift = ((h % 9) - 4) * 0.0012  # 每期漂移，随 symbol 不同有涨有跌
        price = 80.0 + (h % 120)
        d0 = date(2023, 1, 2)
        records = []
        for i in range(100):
            wave = math.sin(i / 5) * 0.004
            price *= 1.0 + drift + wave
            records.append(
                MarketRecord(
                    symbol=symbol,
                    date=(d0 + timedelta(days=i)).isoformat(),
                    close=round(price, 4),
                    open=round(price, 4),
                    volume=1_000_000.0,
                )
            )
        return records


def _build_demo_runtime() -> AgentRuntime:
    from .config import Settings
    from .context import RuntimeContext
    from .data import MarketDataService
    from .llm import StubLLM
    from .tools import build_default_registry

    ctx = RuntimeContext(
        settings=Settings(model_provider="stub", checkpoint_backend="memory"),
        llm=StubLLM(),
        tool_registry=build_default_registry(market_data=MarketDataService([_DemoProvider()])),
    )
    return AgentRuntime(ctx)


def get_runtime(demo: bool = False) -> AgentRuntime:
    global _runtime, _demo_runtime
    if demo:
        if _demo_runtime is None:
            _demo_runtime = _build_demo_runtime()
        return _demo_runtime
    if _runtime is None:
        _runtime = AgentRuntime()
    return _runtime


def set_runtime(rt: AgentRuntime) -> None:
    """测试注入：替换全局 runtime（用 stub 数据，离线可跑）。"""
    global _runtime
    _runtime = rt


class ConfirmRequest(BaseModel):
    task_id: str
    action: str = "approve"  # approve / modify / reject
    config: dict = Field(default_factory=dict)
    reason: str = ""
    demo: bool = False


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


@app.get("/api/run-stream")
def run_stream(query: str = Query(...), task_id: str | None = None, demo: bool = False):
    """SSE：流式跑图。demo=true 时用离线合成行情，便于无网络时演示完整曲线。"""

    def gen():
        for event, payload in get_runtime(demo).stream_run(query, task_id):
            if event == "interrupt":
                payload["demo"] = demo
            yield _sse(event, payload)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/confirm")
def confirm(req: ConfirmRequest):
    decision = {"action": req.action, "reason": req.reason}
    if req.action == "modify" and req.config:
        decision["config"] = req.config
    try:
        state = get_runtime(req.demo).confirm(req.task_id, decision)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    payload = state.model_dump(mode="json")
    payload["demo"] = req.demo
    # 附加真实回测指标 + 收益曲线，供前端渲染指标卡和图表（数据来自结果存储，不编造）
    ref = state.backtest_result_ref
    if ref:
        result = load_result(ref)
        if result:
            payload["metrics"] = result.get("metrics", {})
            payload["series"] = result.get("series", {})
            payload["selected"] = result.get("selected", [])
    return payload


@app.get("/api/task/{task_id}")
def get_task(task_id: str):
    config = {"configurable": {"thread_id": task_id}}
    snap = get_runtime().graph.get_state(config)
    if not snap.values:
        raise HTTPException(status_code=404, detail="task not found")
    values = snap.values
    state = values if isinstance(values, AgentState) else AgentState.model_validate(values)
    return {"task_id": task_id, "next": list(snap.next), **state.model_dump(mode="json")}


@app.get("/api/experiments")
def experiments():
    from .store import list_experiments

    return list_experiments()


@app.get("/")
def index():
    # no-cache：避免浏览器缓存旧版前端，改完刷新即生效
    return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-cache"})
