"""最小 Agent Runtime（Day 2：run + resume）。

把 State + 图 + 模型绑在一起，暴露 `run()` / `resume()`；支持命令行冒烟运行。
resume 用同一 task_id 把补充信息并回原问题继续，不新建任务。
"""
from __future__ import annotations

from uuid import uuid4

from .context import RuntimeContext, build_context
from .graph.builder import build_graph
from .state import AgentState, TaskStatus


class AgentRuntime:
    def __init__(self, ctx: RuntimeContext | None = None) -> None:
        self.ctx = ctx or build_context()
        self.graph = build_graph(self.ctx)
        self._tasks: dict[str, str] = {}  # task_id -> 原始 user_query（进程内）

    def _invoke(self, user_query: str, task_id: str) -> AgentState:
        initial = AgentState(
            task_id=task_id,
            user_query=user_query,
            status=TaskStatus.PENDING,
            current_step="planner_node",
        )
        # 每次调用用独立 thread，避免与已结束的 checkpoint 冲突（真正断点恢复见 Day 7/8）。
        thread = f"{task_id}#{uuid4().hex[:6]}"
        result = self.graph.invoke(initial, config={"configurable": {"thread_id": thread}})
        if isinstance(result, dict):
            return AgentState.model_validate(result)
        return result

    def run(self, user_query: str, task_id: str | None = None) -> AgentState:
        task_id = task_id or f"task_{uuid4().hex[:8]}"
        self._tasks[task_id] = user_query
        return self._invoke(user_query, task_id)

    def resume(self, task_id: str, answer: str) -> AgentState:
        """用户补充参数后继续原任务：同一 task_id，把答案并回原问题重跑。"""
        original = self._tasks.get(task_id)
        if original is None:
            raise KeyError(
                f"找不到任务 {task_id}：resume 需与 run 在同一进程内。"
                "跨进程/断点续跑在 Day 7/8 用 LangGraph interrupt + SQLite checkpoint 实现。"
            )
        merged = f"{original}\n补充信息：{answer}"
        return self._invoke(merged, task_id)


def _print_state(state: AgentState) -> None:
    p = state.task_plan
    c = state.task_constraints
    print(f"task_id  : {state.task_id}")
    print(f"status   : {state.status.value}")
    print(f"goal     : {p.goal}")
    print(f"needs    : research={p.needs_research} backtest={p.needs_backtest} audit={p.needs_risk_audit}")
    print(f"factor   : {p.factor} ({p.factor_direction})")
    print(f"missing  : {p.missing_params or '(none)'}")
    if state.pending_question:
        print(f"question : {state.pending_question}")
    print(f"market   : {c.market}  universe={c.universe or '(none)'}  date_range={c.date_range or '(none)'}")
    print(f"config   : {state.strategy_config or '(none)'}")
    print(f"errors   : {state.errors or '(none)'}")
    print("trace    :")
    for t in state.trace:
        mark = "OK" if t.status.value == "succeeded" else t.status.value.upper()
        extra = f"  [{t.note}]" if t.note else ""
        print(f"  [{mark}] {t.node}{extra}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="QAgent 最小 Runtime 冒烟运行")
    parser.add_argument(
        "query",
        nargs="?",
        default="研究 AAPL、MSFT、NVDA 在 2024-01-01 到 2024-12-31 的 20 日动量并做历史回测",
    )
    args = parser.parse_args()

    runtime = AgentRuntime()
    state = runtime.run(args.query)
    _print_state(state)


if __name__ == "__main__":
    main()
