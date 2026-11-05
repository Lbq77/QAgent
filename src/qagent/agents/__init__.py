"""四角色定义（PRD §5）。

Day 1 只声明角色边界（职责 / 工具白名单 / 权限），Day 3 才接入统一工具系统。
工具白名单是 Harness 的权限依据，不等于系统级沙箱（PRD §13.5）。
"""
from __future__ import annotations

from typing import Any

AGENT_DEFINITIONS: dict[str, dict[str, Any]] = {
    "planner": {
        "role": "任务规划与流程调度",
        "tools": [],  # 只调度与读状态，不执行代码
        "permission": "orchestrate-only",
    },
    "research": {
        "role": "金融资料与市场信息研究",
        "tools": [
            "get_market_data",
            "search_financial_knowledge",
            "get_company_profile",
        ],
        "permission": "read-only",
    },
    "quant": {
        "role": "量化分析与策略执行",
        "tools": [
            "calculate_factor",
            "screen_stocks",
            "run_backtest",
            "get_backtest_result",
        ],
        "permission": "restricted-compute",
    },
    "risk_auditor": {
        "role": "策略与研究结果审计",
        "tools": ["audit_strategy", "validate_report_claims"],
        "permission": "read-only-audit",
    },
}
