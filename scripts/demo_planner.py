"""演示：真实模型如何理解不同股票的请求，并解析成 TaskPlan/TaskConstraints。

用法（项目根目录、已激活 .venv）：
    .venv/bin/python scripts/demo_planner.py

如果顶部「模型实现」显示 StubLLM，说明没配好 .env（模型没被真正调用）。
"""
from qagent.agents.planner import build_task_plan
from qagent.context import build_context

QUERIES = [
    "研究 AAPL 的 20 日动量并回测",
    "从标普500里选出 ROE>15% 且市盈率最低的 5 只股票，比较最近一年收益",
    "回测一个 A 股小市值因子，股票池沪深300，时间范围 2020-01-01 到 2024-12-31",
]


def main() -> None:
    ctx = build_context()
    print(f"模型实现: {type(ctx.llm).__name__}")
    print()

    for q in QUERIES:
        plan, constraints, err = build_task_plan(ctx, q)
        print("=" * 64)
        print(f"输入            : {q}")
        if err:
            print(f"  ⚠ 解析回退    : {err}")
        print(f"  goal          : {plan.goal}")
        print(f"  steps         : {' -> '.join(plan.steps)}")
        print(f"  needs_research: {plan.needs_research}")
        print(f"  needs_backtest: {plan.needs_backtest}")
        print(f"  missing_params: {plan.missing_params or '(无)'}")
        print(f"  market        : {constraints.market}")
        print(f"  universe      : {constraints.universe or '(未指定)'}")
        print(f"  date_range    : {constraints.date_range or '(未指定)'}")
    print()


if __name__ == "__main__":
    main()
