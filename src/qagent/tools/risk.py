"""Risk 审计工具（Day 3：确定性桩，Day 6/8 接真实审计）。"""
from __future__ import annotations

from ..state import FactCheckResult, RiskReport
from .base import Tool
from .schemas import AuditStrategyInput, ValidateReportClaimsInput


class AuditStrategy(Tool):
    name = "audit_strategy"
    description = "审计策略配置的偏差/成本/未来函数风险（Day 6/8 接真实审计）。"
    input_model = AuditStrategyInput
    output_model = RiskReport
    permission = "read-only-audit"

    def execute(self, args: AuditStrategyInput) -> RiskReport:
        return RiskReport(
            level="low",
            findings=[],
            survivorship_bias=True,
            cost_check="t+1 confirmed, cost != 0",
        )


class ValidateReportClaims(Tool):
    name = "validate_report_claims"
    description = "核查报告数字是否来自工具结果（Day 11 接真实核查）。"
    input_model = ValidateReportClaimsInput
    output_model = FactCheckResult
    permission = "read-only-audit"

    def execute(self, args: ValidateReportClaimsInput) -> FactCheckResult:
        return FactCheckResult(claims_checked=len(args.claims), mismatches=[])
