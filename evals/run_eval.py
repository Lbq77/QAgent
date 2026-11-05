"""测试集 v1 评测 runner（Day 3）。

读取 evals/cases.json，用 StubLLM 端到端跑图，逐条对照期望字段，
打印汇总并返回进程退出码（全部通过为 0）。

期望字段支持：status / nodes / needs_research / needs_backtest /
needs_risk_audit / factor / top_n / missing_contains / pending_question_contains /
date_range。

用法（项目根目录）：
    .venv/bin/python evals/run_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from qagent.config import Settings
from qagent.context import RuntimeContext
from qagent.data import MarketDataService, StubMarketDataProvider
from qagent.llm import StubLLM
from qagent.runtime import AgentRuntime
from qagent.state import AgentState
from qagent.tools import build_default_registry

CASES_PATH = Path(__file__).with_name("cases.json")


def _runtime(raw: str | None = None) -> AgentRuntime:
    ctx = RuntimeContext(
        settings=Settings(model_provider="stub"),
        llm=StubLLM(raw=raw),
        tool_registry=build_default_registry(
            market_data=MarketDataService([StubMarketDataProvider()])
        ),
    )
    return AgentRuntime(ctx)


def _llm_raw(case: dict[str, Any]) -> str | None:
    raw = case.get("llm_raw")
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    return json.dumps(raw, ensure_ascii=False)


def evaluate(state: AgentState, expect: dict[str, Any]) -> list[str]:
    """对照期望字段，返回失败原因列表（空列表 = 通过）。"""
    failures: list[str] = []
    plan = state.task_plan
    cons = state.task_constraints

    if "status" in expect and state.status.value != expect["status"]:
        failures.append(f"status: got {state.status.value}, want {expect['status']}")

    if "nodes" in expect:
        got = [t.node for t in state.trace]
        if got != expect["nodes"]:
            failures.append(f"nodes: got {got}, want {expect['nodes']}")

    for flag in ("needs_research", "needs_backtest", "needs_risk_audit"):
        if flag in expect and getattr(plan, flag) != expect[flag]:
            failures.append(f"{flag}: got {getattr(plan, flag)}, want {expect[flag]}")

    if "factor" in expect and plan.factor != expect["factor"]:
        failures.append(f"factor: got {plan.factor}, want {expect['factor']}")

    if "top_n" in expect and plan.top_n != expect["top_n"]:
        failures.append(f"top_n: got {plan.top_n}, want {expect['top_n']}")

    if "missing_contains" in expect:
        for p in expect["missing_contains"]:
            if p not in plan.missing_params:
                failures.append(f"missing_contains: {p} not in {plan.missing_params}")

    if "pending_question_contains" in expect:
        pq = state.pending_question
        for sub in expect["pending_question_contains"]:
            if sub not in pq:
                failures.append(f"pending_question_contains: {sub!r} not in {pq!r}")

    if "date_range" in expect and cons.date_range != expect["date_range"]:
        failures.append(f"date_range: got {cons.date_range}, want {expect['date_range']}")

    return failures


def run_all() -> list[tuple[str, bool, list[str]]]:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    results = []
    for case in cases:
        state = _runtime(_llm_raw(case)).run(case["query"])
        failures = evaluate(state, case.get("expect", {}))
        results.append((case["id"], not failures, failures))
    return results


def main() -> int:
    results = run_all()
    for cid, ok, failures in results:
        mark = "PASS" if ok else "FAIL"
        line = f"[{mark}] {cid}"
        if failures:
            line += "  " + " | ".join(failures)
        print(line)
    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"\n{len(results)} cases: {n_pass} passed, {len(results) - n_pass} failed")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
