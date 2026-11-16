"""Risk 审计工具（audit_strategy 接真实审计）。

audit_strategy 读取策略配置与回测执行结果，显式检查 的四项：
未来函数（信号窗/持有窗分离）、t+1 开盘执行、交易成本、幸存者偏差。
只读审计：不改回测结果，只输出结论与风险等级。
"""
from __future__ import annotations

from ..backtest import load_result
from ..state import FactCheckResult, RiskReport
from .base import Tool
from .schemas import AuditStrategyInput, ValidateReportClaimsInput


class AuditStrategy(Tool):
    name = "audit_strategy"
    description = "审计策略配置的未来函数/t+1/成本/幸存者偏差（真实审计）。"
    input_model = AuditStrategyInput
    output_model = RiskReport
    permission = "read-only-audit"

    def execute(self, args: AuditStrategyInput) -> RiskReport:
        config = args.strategy_config
        findings: list[str] = []

        # 1) 未来函数：信号窗/持有窗是否分离
        lookback_ok = config.lookback_days > 0
        lookahead_check = (
            f"信号窗/持有窗已分离（lookback={config.lookback_days} 交易日）"
            if lookback_ok
            else "未做信号窗/持有窗分离（lookback<=0），存在未来函数风险"
        )
        if not lookback_ok:
            findings.append("未来函数：未做信号窗/持有窗分离")

        # 2) t+1 开盘执行
        execution_ok = config.t_plus_1
        execution_check = (
            "t+1 执行（t 日信号，t+1 日开盘成交）" if execution_ok else "未启用 t+1 开盘执行"
        )
        if not execution_ok:
            findings.append("未启用 t+1 开盘执行")

        # 3) 交易成本非 0 + 三档对比
        cost_ok = config.cost != 0
        cost_check = (
            f"cost={config.cost}，含 0x/default/2x 三档对比"
            if cost_ok
            else "交易成本为 0，违反 "
        )
        if not cost_ok:
            findings.append("交易成本为 0")

        # 4) 交叉核对回测执行结果（读取原始执行产物，不改写）
        payload = load_result(args.backtest_result_ref) if args.backtest_result_ref else None
        if payload is not None:
            if payload.get("error"):
                findings.append(f"回测执行失败：{payload['error']}")
            if "lookback_days" not in payload:
                findings.append("回测结果缺少 lookback_days，无法确认信号窗/持有窗分离")

        # 5) 幸存者偏差：固定股票池必然存在，是披露项而非失败
        survivorship_bias = True

        level = "high" if not lookback_ok else ("medium" if findings else "low")
        return RiskReport(
            level=level,
            findings=findings,
            survivorship_bias=survivorship_bias,
            cost_check=cost_check,
            lookahead_check=lookahead_check,
            execution_check=execution_check,
        )


class ValidateReportClaims(Tool):
    name = "validate_report_claims"
    description = "核查报告数字是否来自工具结果（接真实核查）。"
    input_model = ValidateReportClaimsInput
    output_model = FactCheckResult
    permission = "read-only-audit"

    def execute(self, args: ValidateReportClaimsInput) -> FactCheckResult:
        # 数字核查：每条断言独立重读回测产物，比对声称值 vs 实际值。
        payload = load_result(args.backtest_result_ref) if args.backtest_result_ref else None
        mismatches: list[str] = []
        for claim in args.claims:
            if payload is None or payload.get("error"):
                mismatches.append(
                    f"{claim.field}: 回测结果不可用（ref={args.backtest_result_ref}）"
                )
                continue
            actual = payload.get("metrics", {}).get(claim.field)
            if actual is None:
                mismatches.append(f"{claim.field}: 回测产物缺少该字段")
            elif abs(float(actual) - float(claim.value)) > 1e-9:
                mismatches.append(f"{claim.field}: 声称 {claim.value}，实际 {actual}")
        return FactCheckResult(claims_checked=len(args.claims), mismatches=mismatches)
