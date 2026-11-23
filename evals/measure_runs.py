"""真实 LLM 评测：跑一组真实 query，统计每任务轮数 / 工具调用 / token。

注意：
- LLM 是真模型（SiliconFlow），它决定 plan / research 每轮动作 / 总结——轮数与 token 都由它产生；
- 行情 / 知识库用桩（StubMarketDataProvider + 关闭 GraphRAG）：数据内容不影响「轮数/token」这些指标，
  用桩是为了快、稳、不被本地知识库延迟或网络限流干扰。
- 报告里如实标注了「真」与「桩」的部分，不把桩当真实结果。
"""
from __future__ import annotations

import os
import re
import statistics

# 在导入 qagent 之前关闭 GraphRAG，避免 build_default_registry 构建真实 GraphRAG 客户端
os.environ["QAGENT_GRAPHRAG_ENABLED"] = "false"

from qagent.config import get_settings
from qagent.context import RuntimeContext
from qagent.data import MarketDataService, StubMarketDataProvider
from qagent.llm import get_llm
from qagent.runtime import AgentRuntime
from qagent.tools import build_default_registry

QUERIES = [
    "研究 AAPL、MSFT、NVDA 在 2024-01-01 到 2024-12-31 的 20 日动量并回测",
    "只查一下 AAPL 的公司概况和基本面资料，不用回测",
    "比较 AAPL 和 MSFT 最近一年的股价走势",
    "用 20 日动量策略在 AAPL 和 MSFT 上回测，选出最强的一只",
    "查查 NVDA 是做什么的，属于什么行业",
    "研究 TSLA 在 2023-06-01 到 2024-06-01 的波动率并回测",
]


def _research_node(state):
    return next((t for t in state.trace if t.node == "research_node"), None)


def _rounds_from_note(note: str | None) -> int:
    if not note:
        return 0
    m = re.search(r"react: (\d+) 轮", note)
    return int(m.group(1)) if m else 0


def main() -> None:
    settings = get_settings()
    llm = get_llm(settings)
    ctx = RuntimeContext(
        settings=settings,
        llm=llm,
        tool_registry=build_default_registry(
            market_data=MarketDataService([StubMarketDataProvider()])
        ),
    )
    rt = AgentRuntime(ctx)

    rows = []
    for q in QUERIES:
        calls0, tok0 = llm.calls, llm.total_tokens
        try:
            state = rt.run(q)
            if state.awaiting_confirmation:
                state = rt.confirm(state.task_id, {"action": "approve"})
        except Exception as exc:  # noqa: BLE001 — 单条失败不影响整体统计
            rows.append(
                {
                    "query": q[:30],
                    "status": f"ERROR:{type(exc).__name__}",
                    "graph": 0,
                    "rounds": 0,
                    "r_tools": 0,
                    "llm_calls": llm.calls - calls0,
                    "tokens": llm.total_tokens - tok0,
                }
            )
            continue

        rn = _research_node(state)
        rows.append(
            {
                "query": q[:30],
                "status": state.status.value,
                "graph": len(state.trace),
                "rounds": _rounds_from_note(rn.note if rn else None),
                "r_tools": len(rn.tool_calls) if rn else 0,
                "llm_calls": llm.calls - calls0,
                "tokens": llm.total_tokens - tok0,
            }
        )

    print(f"provider={settings.model_provider}  model={settings.siliconflow_model}\n")
    print(f"{'query':<32}{'status':<10}{'graph':>6}{'rounds':>7}{'r_tools':>8}{'llm':>5}{'tok':>7}")
    for r in rows:
        print(
            f"{r['query']:<32}{r['status']:<10}{r['graph']:>6}{r['rounds']:>7}"
            f"{r['r_tools']:>8}{r['llm_calls']:>5}{r['tokens']:>7}"
        )

    rounds = [r["rounds"] for r in rows if r["rounds"] > 0]
    tokens = [r["tokens"] for r in rows if r["tokens"] > 0]
    llm_calls = [r["llm_calls"] for r in rows if r["llm_calls"] > 0]
    print("\n--- 汇总 ---")
    print(f"任务数: {len(rows)}")
    if rounds:
        print(f"research 轮数: 平均 {statistics.mean(rounds):.2f}  最长 {max(rounds)}")
    if llm_calls:
        print(f"LLM 调用/任务: 平均 {statistics.mean(llm_calls):.2f}  最长 {max(llm_calls)}")
    if tokens:
        print(
            f"token/任务: 平均 {statistics.mean(tokens):.0f}  最长 {max(tokens)}  总 {sum(tokens)}"
        )


if __name__ == "__main__":
    main()
