"""LangGraph 共享 State（PRD §6.3、§6.4）。

原则：State 只保存摘要与引用 ID，完整结果放到外部存储，下游按 ID 读取。
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskStatus(str, Enum):
    """任务级状态（PRD §6.4）。"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class NodeStatus(str, Enum):
    """单个节点的执行状态。"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class TaskPlan(BaseModel):
    """Planner 的结构化任务计划（PRD §5.1）。"""

    goal: str = ""
    steps: list[str] = Field(default_factory=list)
    needs_research: bool = False
    needs_backtest: bool = False
    needs_risk_audit: bool = False
    missing_params: list[str] = Field(default_factory=list)
    # 因子（Day 2.5：Planner 从用户意图提取，Quant 经 factors.resolve_factor 归一化后使用）
    factor: str = "momentum_20d"
    factor_direction: str = "desc"
    # 选股数量（用户说「市值最小的10只」→ 10）
    top_n: int = 3


class TaskConstraints(BaseModel):
    """市场 / 股票池 / 时间范围等硬约束。"""

    market: str = "US"
    universe: list[str] = Field(default_factory=list)
    # 指数名（如「标普500」），代码识别；成分股解析 Day 4 接入
    index: str = ""
    date_range: dict[str, str] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)


# ---- 各节点结构化输出（Agent工程版 D1–2「结构化输出」）----
# Day 3–6 接入真实工具后，节点内部仍以这些模型为返回契约。


class ResearchOutput(BaseModel):
    """Research 节点产物：摘要 + 证据引用 + 快照 ID。"""

    summary: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    snapshot_id: str = ""
    unknowns: list[str] = Field(default_factory=list)


class QuantOutput(BaseModel):
    """Quant 节点产物：策略配置（JSON，由程序校验）+ 回测结果引用。"""

    strategy_config: dict[str, Any] = Field(default_factory=dict)
    backtest_result_ref: str = ""


class RiskReport(BaseModel):
    """Risk Auditor 产物：风险等级 + 发现 + 偏差/成本标记。"""

    level: str = "low"
    findings: list[str] = Field(default_factory=list)
    survivorship_bias: bool = True
    cost_check: str = ""


class FactCheckResult(BaseModel):
    """FactChecker 产物：核查数量 + 对不上的项。"""

    claims_checked: int = 0
    mismatches: list[str] = Field(default_factory=list)


class TraceRecord(BaseModel):
    """最小节点轨迹；Day 9 会细化为 tool_traces 完整表。"""

    node: str
    status: NodeStatus
    started_at: str = Field(default_factory=_now)
    ended_at: str | None = None
    error: str | None = None
    note: str | None = None


class AgentState(BaseModel):
    """四角色共享的结构化 State。

    字段按 PRD §6.3 组织：只保留下游需要的信息与引用 ID。
    """

    # 身份与输入
    task_id: str = ""
    user_query: str = ""

    # Planner 产物
    task_plan: TaskPlan = Field(default_factory=TaskPlan)
    task_constraints: TaskConstraints = Field(default_factory=TaskConstraints)

    # 各 Agent 产物（摘要 + 引用 ID）
    research_context: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    strategy_config: dict[str, Any] = Field(default_factory=dict)
    snapshot_id: str = ""
    backtest_result_ref: str = ""
    risk_report: RiskReport = Field(default_factory=RiskReport)
    fact_check_result: FactCheckResult = Field(default_factory=FactCheckResult)
    artifact_refs: dict[str, str] = Field(default_factory=dict)

    # 人工确认（Day 1 为自动通过桩，Day 7 换成 LangGraph interrupt）
    confirmation: dict[str, Any] = Field(default_factory=dict)
    awaiting_confirmation: bool = False

    # 缺参时向用户提出的问题（await_input 生成，用于补充参数后 resume）
    pending_question: str = ""

    # 执行簿记
    current_step: str = ""
    status: TaskStatus = TaskStatus.PENDING
    errors: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[TraceRecord] = Field(default_factory=list)

    # 必要对话消息占位（Day 2+ 接入 LLM 对话）
    messages: list[dict[str, Any]] = Field(default_factory=list)
