"""Planner 真实模型评测：真实准确率 / 规则召回 / 误拦 / 耗时 / token。

用法（需 .env 配置好 siliconflow）：
    .venv/bin/python scripts/eval_planner.py

每条需求都带「类别 = 标准答案」，据此算真正的准确率（而非只看 JSON 是否合法）：
- 应该被拦下（缺参/不支持因子/指数）的有多少真被拦下（召回）
- 不该被拦下的有多少被误拦（误报）
- 规则相对 LLM 原始输出「挽救」了多少次（对比 raw vs 修正后）
"""
from __future__ import annotations

import time
from types import SimpleNamespace

from qagent.agents.planner import build_task_plan, parse_planner_output
from qagent.agents.prompts import SYSTEM_PROMPTS
from qagent.config import get_settings
from qagent.llm import get_llm

# (query, 类别)
# 类别即标准答案：block=True 表示「应该被规则拦下」
QUERIES: list[tuple[str, str]] = [
    # 完整需求（不应拦，应回测）
    ("研究 AAPL 在 2024-01-01 到 2024-12-31 的 20 日动量并回测", "full"),
    ("回测 MSFT 和 NVDA 在 2023-06-01 到 2024-06-01 的 60 日动量", "full"),
    ("用波动率因子比较 TSLA、AMZN、GOOG 在 2024 年的表现", "full"),
    ("计算 AAPL、MSFT 在 2024-01-01 到 2024-12-31 的市值并回测", "full"),
    ("研究 NVDA 在 2024-01-01 到 2024-12-31 的 20 日均量", "full"),
    ("回测 AAPL 在 2024-01-01 到 2024-12-31 的 5 日动量", "full"),
    ("用市值最小因子选 10 只股票 2024-01-01 到 2024-12-31 回测", "full"),
    ("研究 AAPL、MSFT、NVDA 在 2024 年全年的 20 日动量", "full"),
    ("回测 TSLA 在 2024-01-01 到 2024-12-31 的动量策略", "full"),
    ("回测 AAPL 在 2024-01-01 到 2024-12-31 的动量策略", "full"),
    # 缺日期（应拦 date_range）
    ("回测 AAPL 的动量策略", "missing_date"),
    ("研究 NVDA 的 20 日动量并回测", "missing_date"),
    ("用市值因子选股回测 MSFT", "missing_date"),
    ("回测 AAPL 和 MSFT 的波动率", "missing_date"),
    ("帮我回测一下动量", "missing_date"),
    # 缺股票池（应拦 universe）
    ("回测 2024-01-01 到 2024-12-31 的动量策略", "missing_universe"),
    ("在 2024 年用波动率因子回测", "missing_universe"),
    ("2024 全年动量选股回测", "missing_universe"),
    ("比较 2024 年动量最强的股票", "missing_universe"),
    ("回测 2024-01-01 到 2024-12-31 的小市值策略", "missing_universe"),
    # 不支持因子（应拦 factor，绝不静默替换）
    ("用估值因子选股回测 AAPL 在 2024 年", "unsupported_factor"),
    ("研究市盈率最低的股票 2024 年表现", "unsupported_factor"),
    ("用价值因子回测 MSFT 在 2024-01-01 到 2024-12-31", "unsupported_factor"),
    ("股息率最高的股票 2024 年回测", "unsupported_factor"),
    ("用基本面因子选股 2024 年", "unsupported_factor"),
    ("质量因子选股回测 2024 年", "unsupported_factor"),
    # 相对时间（不应拦，规则应解析出具体日期）
    ("回测 AAPL 过去一年的表现", "relative_date"),
    ("研究 NVDA 最近三个月的动量", "relative_date"),
    ("回测 MSFT 最近 5 年", "relative_date"),
    ("AAPL 近 6 个月的 20 日动量回测", "relative_date"),
    ("过去两年 AAPL 的波动率回测", "relative_date"),
    # 指数（应拦 universe，不臆造成分股）
    ("在标普500里用动量因子回测", "index"),
    ("纳指成分股的动量回测", "index"),
    ("沪深300 里选小市值回测", "index"),
    ("道琼斯成分股 2024 年表现", "index"),
    ("中证500 波动率回测", "index"),
    # 只研究（不应拦，不应回测）
    ("查一下 AAPL 的资料", "research_only"),
    ("NVDA 是做什么的公司", "research_only"),
    ("特斯拉的主营业务是什么", "research_only"),
    ("介绍下微软这家公司", "research_only"),
    ("查询 AAPL 的公司信息", "research_only"),
    # 垃圾输入（不应拦，回退只研究）
    ("今天天气怎么样", "garbage"),
    ("给我讲个笑话", "garbage"),
    ("1 + 1 等于几", "garbage"),
    ("你好", "garbage"),
    ("这完全不是量化问题", "garbage"),
]

# 类别 → 期望：block（该不该拦）、reason（拦下的原因）、backtest（是否该回测）、date_resolved（是否该解析出日期）
EXPECT = {
    "full": {"block": False, "backtest": True},
    "missing_date": {"block": True, "reason": "date_range"},
    "missing_universe": {"block": True, "reason": "universe"},
    "unsupported_factor": {"block": True, "reason": "factor"},
    "relative_date": {"block": False, "backtest": True, "date_resolved": True},
    "index": {"block": True, "reason": "universe"},
    "research_only": {"block": False, "backtest": False},
    "garbage": {"block": False},
}


def main() -> None:
    settings = get_settings()
    llm = get_llm(settings)
    ctx = SimpleNamespace(llm=llm)

    n = len(QUERIES)
    correct = 0
    should_block_total = sum(1 for _, c in QUERIES if EXPECT[c]["block"])
    should_block_caught = 0
    shouldnt_block_total = n - should_block_total
    false_positive = 0
    rule_saves = {"unsupported_factor": 0, "date_overridden": 0, "missing_param_added": 0}
    total_time = 0.0
    total_in = total_out = 0

    for q, cat in QUERIES:
        exp = EXPECT[cat]
        t0 = time.time()
        try:
            raw = llm.invoke(SYSTEM_PROMPTS["planner"], q)
            raw_out = parse_planner_output(raw) if raw.strip().startswith("{") else None
            plan, constraints, err = build_task_plan(ctx, q, _raw=raw)
        except Exception as exc:  # noqa: BLE001
            print(f"[异常] {q}: {type(exc).__name__}: {exc}")
            continue
        dt = time.time() - t0
        total_time += dt
        usage = getattr(llm, "last_usage", {}) or {}
        total_in += int(usage.get("input_tokens", 0))
        total_out += int(usage.get("output_tokens", 0))

        blocked = bool(plan.missing_params)
        # 1) 正确性判定
        ok = True
        if exp["block"]:
            ok = exp["reason"] in plan.missing_params
        else:
            if blocked:
                ok = False
            if exp.get("backtest") and not plan.needs_backtest:
                ok = False
            if exp.get("date_resolved") and not constraints.date_range:
                ok = False
        if ok:
            correct += 1
        # 2) 召回 / 误拦
        if exp["block"]:
            if exp["reason"] in plan.missing_params:
                should_block_caught += 1
        else:
            if blocked:
                false_positive += 1
        # 3) 规则挽救：对比 LLM 原始输出 vs 修正后
        if raw_out is not None:
            if cat == "unsupported_factor" and "factor" in plan.missing_params:
                rule_saves["unsupported_factor"] += 1
            if raw_out.date_range and raw_out.date_range != constraints.date_range:
                rule_saves["date_overridden"] += 1
            if len(plan.missing_params) > len(raw_out.missing_params or []):
                rule_saves["missing_param_added"] += 1

    print("=" * 52)
    print("Planner 真实模型评测（带标准答案）")
    print("=" * 52)
    print(f"模型        : {settings.siliconflow_model}")
    print(f"总需求数    : {n}")
    print(f"真实准确率  : {correct}/{n} = {correct / n:.1%}")
    print(f"召回（该拦被拦）: {should_block_caught}/{should_block_total} = {should_block_caught / should_block_total:.1%}")
    print(f"误拦（不该拦被拦）: {false_positive}/{shouldnt_block_total}")
    print(f"平均耗时    : {total_time / n * 1000:.0f} ms/条")
    print(f"Token 花费  : 输入 {total_in} / 输出 {total_out} / 合计 {total_in + total_out}")
    print("-" * 52)
    print("规则挽救（相对 LLM 原始输出的修正次数）:")
    for r, c in rule_saves.items():
        print(f"  {r:22s} {c}")


if __name__ == "__main__":
    main()
