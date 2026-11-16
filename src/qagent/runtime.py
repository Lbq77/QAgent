"""Agent Runtime（run + resume；interrupt 人工确认；SQLite 断点恢复）。

把 State + 图 + 模型绑在一起，暴露三个入口：
- run(query)            启动任务；若停在 human_confirm 则返回 awaiting_confirmation 的状态
- resume(task_id, answer) 缺参（await_input）后补充信息，同一 task 重跑
- confirm(task_id, decision) 人工确认：approve / modify / reject（LangGraph interrupt 恢复）

thread_id 与 task_id 一一对应，因此配合 SQLite checkpointer 可跨进程恢复：
新进程里 `confirm(task_id, decision)` 也能找到之前的挂起点继续。
"""
from __future__ import annotations

from uuid import uuid4

from langgraph.types import Command

from .context import RuntimeContext, build_context
from .graph.builder import build_graph
from .state import AgentState, TaskStatus


class AgentRuntime:
    def __init__(self, ctx: RuntimeContext | None = None) -> None:
        self.ctx = ctx or build_context()
        self.graph = build_graph(self.ctx)
        # task_id -> 运行簿记（进程内；跨进程恢复靠 SQLite checkpointer）
        self._tasks: dict[str, dict] = {}

    # ---- 内部 ----

    def _to_state(self, result, config) -> AgentState:
        """把 graph 返回结果归一成 AgentState；命中 interrupt 时标记 awaiting_confirmation。"""
        if isinstance(result, dict) and "__interrupt__" in result:
            snapshot = self.graph.get_state(config)
            values = snapshot.values
            state = (
                values if isinstance(values, AgentState) else AgentState.model_validate(values)
            )
            return state.model_copy(
                update={"awaiting_confirmation": True, "current_step": "human_confirm_node"}
            )
        if isinstance(result, dict):
            return AgentState.model_validate(result)
        return result

    def _invoke(self, user_query: str, task_id: str, thread_id: str) -> AgentState:
        initial = AgentState(
            task_id=task_id,
            user_query=user_query,
            status=TaskStatus.PENDING,
            current_step="planner_node",
        )
        config = {"configurable": {"thread_id": thread_id}}
        return self._to_state(self.graph.invoke(initial, config=config), config)

    def _reset_harness(self) -> None:
        """每次任务开始前重置工具调用预算——预算按任务隔离，不跨任务累加。"""
        harness = getattr(self.ctx.tool_registry, "harness", None)
        if harness is not None and hasattr(harness, "reset"):
            harness.reset()

    # ---- 公开入口 ----

    def run(self, user_query: str, task_id: str | None = None) -> AgentState:
        task_id = task_id or f"task_{uuid4().hex[:8]}"
        thread_id = task_id  # thread 与 task 一一对应（跨进程可恢复）
        self._tasks[task_id] = {"query": user_query, "thread_id": thread_id, "seq": 1}
        self._reset_harness()
        return self._invoke(user_query, task_id, thread_id)

    def resume(self, task_id: str, answer: str) -> AgentState:
        """缺参（await_input）后补充信息：同一 task，把答案并回原问题重跑。"""
        entry = self._tasks.get(task_id)
        if entry is None:
            raise KeyError(
                f"找不到任务 {task_id}：resume（缺参补充）需与 run 在同一进程内。"
                "跨进程恢复人工确认请用 confirm(task_id, decision)。"
            )
        if self.graph.get_state({"configurable": {"thread_id": entry["thread_id"]}}).next:
            raise RuntimeError(
                f"任务 {task_id} 正在等待人工确认，请用 confirm(task_id, decision)，"
                "而不是 resume（补充信息）。"
            )
        # 已结束（blocked）→ 合并补充信息后重跑，用新的子 thread 避免与旧 checkpoint 冲突
        entry["seq"] += 1
        entry["thread_id"] = f"{task_id}#{entry['seq']}"
        merged = f"{entry['query']}\n补充信息：{answer}"
        self._reset_harness()
        return self._invoke(merged, task_id, entry["thread_id"])

    def confirm(self, task_id: str, decision: dict) -> AgentState:
        """人工确认（interrupt 恢复）：decision = {"action": approve|modify|reject, ...}。

        thread_id 由 task_id 推导，因此新进程也能恢复之前挂起的确认点。
        """
        entry = self._tasks.get(task_id)
        thread_id = entry["thread_id"] if entry else task_id
        config = {"configurable": {"thread_id": thread_id}}
        if not self.graph.get_state(config).next:
            raise RuntimeError(f"任务 {task_id} 当前没有待确认的挂起点（可能已结束）")
        return self._to_state(
            self.graph.invoke(Command(resume=decision), config=config), config
        )

    def stream_run(self, user_query: str, task_id: str | None = None):
        """流式跑图，逐个 yield ('node'|'interrupt'|'done', payload)。

        供 SSE 用——`graph.stream` 按节点产出进度；命中 interrupt 时 yield 一次
        ('interrupt', payload) 后结束（挂起等 confirm 续跑）。payload 均为 JSON 安全。
        """
        task_id = task_id or f"task_{uuid4().hex[:8]}"
        thread_id = task_id  # 与 confirm 共享同一 thread，才能续跑
        self._tasks[task_id] = {"query": user_query, "thread_id": thread_id, "seq": 1}
        self._reset_harness()
        initial = AgentState(
            task_id=task_id,
            user_query=user_query,
            status=TaskStatus.PENDING,
            current_step="planner_node",
        )
        config = {"configurable": {"thread_id": thread_id}}
        for chunk in self.graph.stream(initial, config=config, stream_mode="updates"):
            if "__interrupt__" in chunk:
                value = chunk["__interrupt__"][0].value
                yield ("interrupt", {"task_id": task_id, **value})
                return  # 挂起等待人工确认，由 confirm 续跑
            for node_name in chunk:
                yield ("node", {"task_id": task_id, "node": node_name})
        # 未中断，正常走完（research-only 等路径）
        snapshot = self.graph.get_state(config)
        values = snapshot.values
        state = values if isinstance(values, AgentState) else AgentState.model_validate(values)
        yield (
            "done",
            {
                "task_id": task_id,
                "status": state.status.value,
                "final_report": state.final_report,
                "risk_level": state.risk_report.level,
                "fact_check": state.fact_check_result.model_dump(),
            },
        )


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
    if state.awaiting_confirmation:
        print(">> 等待人工确认：approve / modify / reject")
    print(f"errors   : {state.errors or '(none)'}")
    print("trace    :")
    for t in state.trace:
        mark = "OK" if t.status.value == "succeeded" else t.status.value.upper()
        extra = f"  [{t.note}]" if t.note else ""
        print(f"  [{mark}] {t.node}{extra}")
    if state.final_report:
        print("report   :")
        for line in state.final_report.splitlines():
            print(f"  {line}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="QAgent Runtime 冒烟运行（含人工确认）")
    parser.add_argument(
        "query",
        nargs="?",
        default="研究 AAPL、MSFT、NVDA 在 2024-01-01 到 2024-12-31 的 20 日动量并回测",
    )
    args = parser.parse_args()

    runtime = AgentRuntime()
    state = runtime.run(args.query)
    _print_state(state)

    while state.awaiting_confirmation:
        choice = input("\n确认策略？approve / reject / modify [approve]: ").strip().lower() or "approve"
        if choice.startswith("r"):
            decision = {"action": "reject", "reason": "user rejected"}
        elif choice.startswith("m"):
            raw = input("要改成 top_n 多少？(回车不改): ").strip()
            overrides = {"top_n": int(raw)} if raw.isdigit() else {}
            decision = {"action": "modify", "config": overrides}
        else:
            decision = {"action": "approve"}
        state = runtime.confirm(state.task_id, decision)
        _print_state(state)


if __name__ == "__main__":
    main()
