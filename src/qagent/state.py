"""LangGraph 共享 State。

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
    """任务级状态。"""

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
    """Planner 的结构化任务计划。"""

    goal: str = ""
    steps: list[str] = Field(default_factory=list)
    needs_research: bool = False
    needs_backtest: bool = False
    needs_risk_audit: bool = False
    missing_params: list[str] = Field(default_factory=list)
    # 因子（Planner 从用户意图提取，Quant 经 factors.resolve_factor 归一化后使用）
    factor: str = "momentum_20d"
    factor_direction: str = "desc"
    # 选股数量（用户说「市值最小的10只」→ 10）
    top_n: int = 3


class TaskConstraints(BaseModel):
    """市场 / 股票池 / 时间范围等硬约束。"""

    market: str = "US"
    universe: list[str] = Field(default_factory=list)
    # 指数名（如「标普500」），代码识别；成分股解析 接入
    index: str = ""
    date_range: dict[str, str] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)


# ---- 各节点结构化输出（Agent工程版 D1–2「结构化输出」）----
# 接入真实工具后，节点内部仍以这些模型为返回契约。


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
    """Risk Auditor 产物：风险等级 + 发现 + 各检查项结论（起接真实审计）。"""

    level: str = "low"  # low / medium / high
    findings: list[str] = Field(default_factory=list)  # 发现的问题（未来函数/t+1/成本/执行失败）
    survivorship_bias: bool = True  # 固定股票池必然存在幸存者偏差（披露项，非失败）
    cost_check: str = ""  # 成本检查结论
    lookahead_check: str = ""  # 未来函数检查结论（信号窗/持有窗是否分离）
    execution_check: str = ""  # 执行检查结论（是否 t+1 开盘）


class FactCheckResult(BaseModel):
    """FactChecker 产物：核查数量 + 对不上的项。"""

    claims_checked: int = 0
    mismatches: list[str] = Field(default_factory=list)


class ToolCallRecord(BaseModel):
    """一次工具调用的轨迹（细化为 tool_traces 完整表）。"""

    tool: str
    agent: str = ""
    status: str = "succeeded"  # succeeded / failed
    duration_ms: float = 0.0
    input_summary: str = ""
    output_summary: str = ""
    error: str | None = None


class TraceRecord(BaseModel):
    """节点轨迹；起携带该节点发起的工具调用明细（tool_calls）。"""

    node: str
    status: NodeStatus
    started_at: str = Field(default_factory=_now)
    ended_at: str | None = None
    error: str | None = None
    note: str | None = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)


class AgentState(BaseModel):
    """四角色共享的结构化 State。

    字段按 组织：只保留下游需要的信息与引用 ID。
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
    final_report: str = ""  # report 节点汇总四角色产物的最终文本

    # 人工确认（为自动通过桩，换成 LangGraph interrupt）
    confirmation: dict[str, Any] = Field(default_factory=dict)
    awaiting_confirmation: bool = False

    # 缺参时向用户提出的问题（await_input 生成，用于补充参数后 resume）
    pending_question: str = ""

    # 执行簿记
    current_step: str = ""
    status: TaskStatus = TaskStatus.PENDING
    errors: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[TraceRecord] = Field(default_factory=list)

    # 必要对话消息占位（+ 接入 LLM 对话）
    messages: list[dict[str, Any]] = Field(default_factory=list)
