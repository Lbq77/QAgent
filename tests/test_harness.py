"""Harness（预算 / 超时 / 重试）单元测试。"""
from __future__ import annotations

import time

import pytest

from qagent.config import Settings
from qagent.harness import (
    Harness,
    HarnessBudgetExceeded,
    HarnessTimeoutError,
    build_harness,
)


def test_timeout_raises():
    h = Harness(timeout_seconds=0.05, retries=0)

    def slow() -> str:
        time.sleep(0.3)
        return "late"

    with pytest.raises(HarnessTimeoutError):
        h.call(slow)


def test_timeout_returns_early():
    """超时后调用方必须提前返回，而不是等慢任务跑完（回归 with ThreadPoolExecutor 阻塞退出）。"""
    h = Harness(timeout_seconds=0.5, retries=0)

    def slow() -> str:
        time.sleep(3)
        return "late"

    start = time.monotonic()
    with pytest.raises(HarnessTimeoutError):
        h.call(slow)
    elapsed = time.monotonic() - start
    assert elapsed < 1.0  # 3 秒任务，调用方必须在 1 秒内返回


def test_retry_then_success():
    h = Harness(timeout_seconds=0.05, retries=1, retry_backoff=0.01)
    calls: dict[str, int] = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            time.sleep(0.3)  # 第一次超时
        return "ok"

    assert h.call(flaky) == "ok"
    assert calls["n"] == 2  # 超时后重试一次成功


def test_budget_exceeded():
    h = Harness(max_tool_calls=2, timeout_seconds=5.0, retries=0)

    def fast() -> str:
        return "x"

    assert h.call(fast) == "x"
    assert h.call(fast) == "x"
    with pytest.raises(HarnessBudgetExceeded):
        h.call(fast)


def test_budget_counts_logical_calls_not_retries():
    h = Harness(max_tool_calls=1, timeout_seconds=0.05, retries=1, retry_backoff=0.01)
    calls: dict[str, int] = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            time.sleep(0.3)  # 第一次超时，重试成功
        return "ok"

    assert h.call(flaky) == "ok"  # 一次逻辑调用，内部重试，不占额外预算


def test_build_harness_disabled_returns_none():
    settings = Settings(model_provider="stub", tool_timeout_seconds=0, tool_retries=0)
    assert build_harness(settings) is None


def test_build_harness_enabled():
    settings = Settings(model_provider="stub", tool_timeout_seconds=1.0, tool_retries=2)
    h = build_harness(settings)
    assert isinstance(h, Harness)
    assert h.timeout_seconds == 1.0
    assert h.retries == 2
