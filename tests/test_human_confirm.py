"""人工确认（interrupt）——approve / reject / modify 三种决策路径。"""
from __future__ import annotations

import pytest

from qagent.config import Settings
from qagent.context import RuntimeContext
from qagent.data import MarketDataService, StubMarketDataProvider
from qagent.llm import StubLLM
from qagent.runtime import AgentRuntime
from qagent.tools import build_default_registry


def _runtime(raw: str | None = None) -> AgentRuntime:
    ctx = RuntimeContext(
        settings=Settings(model_provider="stub"),
        llm=StubLLM(raw=raw),
        tool_registry=build_default_registry(
            market_data=MarketDataService([StubMarketDataProvider()])
        ),
    )
    return AgentRuntime(ctx)


QUERY = "研究 AAPL 在 2024-01-01 到 2024-12-31 的 20 日动量并回测"


def test_approve_continues_to_report():
    rt = _runtime()
    s = rt.run(QUERY)
    assert s.awaiting_confirmation is True

    final = rt.confirm(s.task_id, {"action": "approve"})
    assert final.status.value == "succeeded"
    assert final.confirmation["action"] == "approve"
    assert final.final_report  # 四角色汇总报告非空
    assert "human_confirm_node" in [t.node for t in final.trace]


def test_reject_cancels_task():
    rt = _runtime()
    s = rt.run(QUERY)
    final = rt.confirm(s.task_id, {"action": "reject", "reason": "不想回测"})
    assert final.status.value == "cancelled"
    assert final.confirmation["action"] == "reject"
    assert "risk_node" not in [t.node for t in final.trace]


def test_modify_reruns_backtest_and_repauses():
    rt = _runtime()
    s = rt.run(QUERY)
    old_ref = s.backtest_result_ref

    s2 = rt.confirm(s.task_id, {"action": "modify", "config": {"top_n": 2}})
    assert s2.awaiting_confirmation is True  # 修改后重新回测，再次等待确认
    assert s2.strategy_config["top_n"] == 2
    assert s2.backtest_result_ref != old_ref  # 重新回测产生新 ref

    final = rt.confirm(s.task_id, {"action": "approve"})
    assert final.status.value == "succeeded"
    assert final.strategy_config["top_n"] == 2


def test_invalid_modify_keeps_original_and_continues():
    rt = _runtime()
    s = rt.run(QUERY)
    final = rt.confirm(s.task_id, {"action": "modify", "config": {"top_n": "nope"}})
    # 非法修改 → 视为 approve 原配置，继续到风险审计并收尾
    assert final.status.value == "succeeded"
    assert final.strategy_config["top_n"] == 3  # 原默认
    assert any(e.get("node") == "human_confirm_node" for e in final.errors)


def test_confirm_without_pending_raises():
    rt = _runtime()
    s = rt.run(QUERY)
    rt.confirm(s.task_id, {"action": "approve"})  # 正常完成
    with pytest.raises(RuntimeError):
        rt.confirm(s.task_id, {"action": "approve"})  # 已结束，无挂起点
