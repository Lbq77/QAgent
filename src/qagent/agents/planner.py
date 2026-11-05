"""Planner 的规划逻辑（Day 2：解析 LLM 结构化输出）。

调用模型生成 JSON，解析成 PlannerOutput（计划 + 约束 + 因子 + top_n），再做确定性收尾：
1. 因子归一化（白名单 + 方向兜底）；
2. 相对时间 → 具体日期（resolve_date_range，覆盖 LLM）；
3. 指数识别（标普500/纳指等，确定性关键字）；
4. 缺参补充（回测必需：股票池、时间范围）。

解析失败回退「只做研究、不臆造回测」，并返回错误信息由节点记录。
"""
from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field

from ..dates import resolve_date_range
from ..factors import UnknownFactorError, detect_unsupported_factor, resolve_factor
from ..state import TaskConstraints, TaskPlan
from .prompts import SYSTEM_PROMPTS

# 常见指数名 → 识别用（成分股解析 Day 4 接入，当前只捕获意图并阻塞提问，不臆造成分股）
INDEX_PATTERNS: list[tuple[str, list[str]]] = [
    ("标普500", ["标普500", "标普 500", "标准普尔500", "s&p 500", "s&p500", "sp500", "spx"]),
    ("纳斯达克100", ["纳斯达克100", "纳指100", "纳指", "nasdaq 100", "nasdaq100", "ndx"]),
    ("道琼斯", ["道琼斯", "道指", "dow jones", "djia"]),
    ("沪深300", ["沪深300", "csi 300"]),
    ("中证500", ["中证500", "csi 500"]),
]


def detect_index(text: str) -> str:
    """从用户原文识别指数名，识别不到返回空串。"""
    t = text.lower()
    for name, patterns in INDEX_PATTERNS:
        for p in patterns:
            if p in t:
                return name
    return ""


class PlannerOutput(BaseModel):
    """Planner 的完整结构化输出：计划字段 + 约束字段 + 因子 + top_n。"""

    goal: str = ""
    steps: list[str] = Field(default_factory=list)
    needs_research: bool = True
    needs_backtest: bool = True
    needs_risk_audit: bool = True
    missing_params: list[str] = Field(default_factory=list)
    market: str = "US"
    universe: list[str] = Field(default_factory=list)  # 股票池
    date_range: dict[str, str] = Field(default_factory=dict)
    factor: str = "momentum_20d"
    factor_direction: str = "desc"
    top_n: int = 3


def _extract_json(text: str) -> str:
    """从模型输出里抠出 JSON：去代码围栏，再取第一个 { 到最后一个 }。"""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("model output contains no JSON object")
    return text[start : end + 1]


def parse_planner_output(raw: str) -> PlannerOutput:
    data = json.loads(_extract_json(raw))
    return PlannerOutput.model_validate(data)


def _split(out: PlannerOutput, index: str = "") -> tuple[TaskPlan, TaskConstraints]:
    plan = TaskPlan(
        goal=out.goal,
        steps=out.steps,
        needs_research=out.needs_research,
        needs_backtest=out.needs_backtest,
        needs_risk_audit=out.needs_risk_audit,
        missing_params=out.missing_params,
        factor=out.factor,
        factor_direction=out.factor_direction,
        top_n=out.top_n,
    )
    constraints = TaskConstraints(
        market=out.market,
        universe=out.universe,
        index=index,
        date_range=out.date_range,
    )
    return plan, constraints


def build_task_plan(ctx, user_query: str) -> tuple[TaskPlan, TaskConstraints, str | None]:
    """调 LLM → 解析 → 归一化/日期/指数/缺参 → 返回 (计划, 约束, 解析错误或 None)。"""
    raw = ctx.llm.invoke(SYSTEM_PROMPTS["planner"], user_query)
    try:
        out = parse_planner_output(raw)
        err = None
    except Exception as exc:  # 解析失败回退：只做只读研究，不臆造回测（PRD §6.2）
        out = PlannerOutput(
            goal=user_query,
            steps=["research", "report"],
            needs_research=True,
            needs_backtest=False,
            needs_risk_audit=False,
        )
        err = f"planner parse failed, fallback to research-only: {exc}"

    # 1) 因子：先确定性检测「用户是否提到不支持的因子」，再校验 LLM 给的因子
    unsupported = detect_unsupported_factor(user_query) if out.needs_backtest else ""
    if unsupported:
        # 用户要的因子不在白名单，无论 LLM 输出什么都阻塞，绝不静默替换
        out.factor = unsupported
    else:
        try:
            ranking = resolve_factor(out.factor, out.factor_direction)
            out.factor, out.factor_direction = ranking["factor"], ranking["direction"]
        except UnknownFactorError:
            unsupported = out.factor  # LLM 直接给了未知因子名，保留原话供提问

    # 2) 日期由确定性代码解析（LLM 不知道「今天」，覆盖 LLM 结果）
    resolved = resolve_date_range(user_query)
    if resolved:
        out.date_range = resolved

    # 3) 指数识别（确定性关键字）
    index = detect_index(user_query)

    # 4) 缺参补充：回测必需股票池 + 时间范围；因子只阻塞「未知因子」，不阻塞「未指定」
    missing = list(out.missing_params)
    if out.needs_backtest and not out.date_range:
        if "date_range" not in missing:
            missing.append("date_range")
    else:
        missing = [p for p in missing if p != "date_range"]

    if out.needs_backtest and not out.universe:
        if "universe" not in missing:
            missing.append("universe")
    else:
        missing = [p for p in missing if p != "universe"]

    # LLM 自己往 missing_params 塞的 factor 一律去掉；只有代码判定的「不支持的因子」才阻塞
    missing = [p for p in missing if p != "factor"]
    if unsupported:
        missing.append("factor")
    out.missing_params = missing

    plan, constraints = _split(out, index=index)
    return plan, constraints, err
