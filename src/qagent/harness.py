"""Harness：工具调用的预算 / 超时 / 重试。

权限白名单已在 `tools.registry.ensure_agent_allowed` 落地，这里补上运行时侧的三件事：
- 预算：`max_tool_calls` 超限即停，防止 Agent 失控打爆数据源；
- 超时：同步工具在独立线程里跑，超过 `tool_timeout_seconds` 抛 `HarnessTimeoutError`；
- 重试：对超时（可恢复的瞬时错误）做有限次退避重试；Schema/权限错误属确定性错误，不重试。

说明：线程超时无法真正杀死底层线程（Python 限制），超时后后台线程仍会跑完；
这里只保证「调用方不再等待、不阻塞主流程」，适合本项目「数据源调用」这一层。
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable


class HarnessError(Exception):
    """Harness 层错误的基类。"""


class HarnessTimeoutError(HarnessError):
    """工具调用超时（可重试的瞬时错误）。"""


class HarnessBudgetExceeded(HarnessError):
    """工具调用次数超过预算（不可重试）。"""


class Harness:
    def __init__(
        self,
        *,
        max_tool_calls: int = 30,
        timeout_seconds: float = 30.0,
        retries: int = 1,
        retry_backoff: float = 0.5,
    ) -> None:
        self.max_tool_calls = max_tool_calls
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.retry_backoff = retry_backoff
        self._calls = 0

    @property
    def calls(self) -> int:
        return self._calls

    def reset(self) -> None:
        """重置本任务/本次运行的调用计数（预算按任务隔离，不跨任务累加）。"""
        self._calls = 0

    def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """预算 + 超时 + 重试，一次逻辑调用。"""
        if self._calls >= self.max_tool_calls:
            raise HarnessBudgetExceeded(
                f"tool call budget exhausted ({self.max_tool_calls})"
            )
        self._calls += 1

        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                return self._run_with_timeout(fn, args, kwargs)
            except HarnessTimeoutError as exc:
                last = exc
                if attempt < self.retries:
                    time.sleep(self.retry_backoff * (attempt + 1))
                    continue
                raise
        raise last  # type: ignore[misc]  # pragma: no cover — 循环内必 raise

    def _run_with_timeout(self, fn: Callable[..., Any], args: tuple, kwargs: dict) -> Any:
        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(fn, *args, **kwargs)
            try:
                return future.result(timeout=self.timeout_seconds)
            except TimeoutError:
                future.cancel()
                raise HarnessTimeoutError(
                    f"tool call timed out after {self.timeout_seconds}s"
                ) from None


def build_harness(settings) -> Harness | None:
    """按配置构建 Harness；超时与重试都关闭时返回 None（直接调用，不进线程）。"""
    if settings.tool_timeout_seconds <= 0 and settings.tool_retries <= 0:
        return None
    return Harness(
        max_tool_calls=settings.max_tool_calls,
        timeout_seconds=settings.tool_timeout_seconds,
        retries=settings.tool_retries,
        retry_backoff=settings.tool_retry_backoff_seconds,
    )
