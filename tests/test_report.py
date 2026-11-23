"""报告节点的 LLM 总结与投资建议（数字来自工具、LLM 只做自然语言总结）。"""
from __future__ import annotations

import json

from qagent.config import Settings
from qagent.context import RuntimeContext
from qagent.data import MarketDataService, StubMarketDataProvider
from qagent.llm import StubLLM
from qagent.runtime import AgentRuntime
from qagent.tools import build_default_registry

QUERY = "研究 AAPL 在 2024-01-01 到 2024-12-31 的 20 日动量并回测"

_PLAN = {
    "goal": "研究 AAPL 动量并回测",
    "steps": ["research", "quant", "risk_audit", "fact_check", "report"],
    "needs_research": True,
    "needs_backtest": True,
    "needs_risk_audit": True,
    "missing_params": [],
    "market": "US",
    "universe": ["AAPL"],
    "date_range": {"start": "2024-01-01", "end": "2024-12-31"},
    "factor": "momentum_20d",
    "factor_direction": "desc",
    "top_n": 3,
}


class _FakeLLM:
    """非 stub 的假 LLM：planner 返回合法 JSON，report 返回一段总结文本。"""

    is_stub = False

    def invoke(self, system: str, user: str) -> str:
        if "任务规划器" in system:
            return json.dumps(_PLAN, ensure_ascii=False)
        return "（LLM 总结）该策略回测表现平稳，需关注回撤风险。本内容仅供研究学习，不构成投资建议。"


class _HallucinatingLLM:
    """report 阶段编造数字的 LLM，且用「年化收益率达到45.6%」这种难抓的写法。"""

    is_stub = False

    def invoke(self, system: str, user: str) -> str:
        if "任务规划器" in system:
            return json.dumps(_PLAN, ensure_ascii=False)
        return "该策略回测表现优异，年化收益率达到45.6%，夏普比率为 3.5。建议坚定持有。本内容仅供研究学习，不构成投资建议。"


def _runtime(llm) -> AgentRuntime:
    ctx = RuntimeContext(
        settings=Settings(model_provider="stub"),
        llm=llm,
        tool_registry=build_default_registry(
            market_data=MarketDataService([StubMarketDataProvider()])
        ),
    )
    return AgentRuntime(ctx)


def test_report_includes_llm_summary():
    rt = _runtime(_FakeLLM())
    s = rt.run(QUERY)
    final = rt.confirm(s.task_id, {"action": "approve"})
    assert "分析师总结与建议" in final.final_report
    assert "（LLM 总结）" in final.final_report  # LLM 生成的总结被采纳
    assert "不构成投资建议" in final.final_report


def test_report_stub_falls_back_to_deterministic_summary():
    rt = _runtime(StubLLM())
    s = rt.run(QUERY)
    final = rt.confirm(s.task_id, {"action": "approve"})
    assert "分析师总结与建议" in final.final_report
    assert "不构成投资建议" in final.final_report  # 兜底模板也带免责声明


def test_report_flags_hallucinated_number():
    """LLM 总结编造「年化收益率达到45.6%」→ 数字核查抓到后应改用确定性模板 + 披露。"""
    rt = _runtime(_HallucinatingLLM())
    s = rt.run(QUERY)
    final = rt.confirm(s.task_id, {"action": "approve"})

    assert "事实核查不符" in final.final_report  # 披露了不符
    assert "45.60%" in final.final_report        # 声称值被记进 mismatch
    assert any("百分比" in m for m in final.fact_check_result.mismatches)
    # 错误的 LLM 总结被弃用，报告里的「分析师总结」已是确定性模板（不含 45.6% 的正文）
    summary_section = final.final_report.split("## 分析师总结与建议", 1)[1]
    assert "45.6%" not in summary_section


def _verify_summary_claims_direct(summary: str, metrics: dict) -> list[str]:
    from qagent.graph.nodes import _verify_summary_claims

    mismatches, _ = _verify_summary_claims(summary, {"metrics": metrics})
    return mismatches


def test_verify_summary_ignores_confidence_level():
    """95% 置信、VaR(95%) 是常量，不应被当成编造的指标数字。"""
    metrics = {
        "annual_return": 0.1,
        "total_return": 0.1,
        "max_drawdown": 0.05,
        "var_95": 0.03,
        "annual_volatility": 0.2,
    }
    summary = "年化收益 10%、最大回撤 5%、波动率 20%、VaR(95%) 为 3%，95% 置信水平下。"
    assert _verify_summary_claims_direct(summary, metrics) == []


def test_verify_summary_still_catches_fabrication():
    """真正的编造数字（与所有真实指标都对不上）仍会被抓。"""
    metrics = {"annual_return": 0.1, "max_drawdown": 0.05, "var_95": 0.03, "annual_volatility": 0.2}
    summary = "年化收益率高达 88%！"
    assert any("88.00%" in m for m in _verify_summary_claims_direct(summary, metrics))
