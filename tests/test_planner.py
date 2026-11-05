"""planner 解析逻辑的离线测试（不依赖网络/API key，确定性）。"""
from qagent.agents.planner import build_task_plan, parse_planner_output
from qagent.llm import StubLLM

RAW_WITH_FENCE = '''```json
{
  "goal": "研究 AAPL 的 20 日动量并回测",
  "steps": ["research", "quant", "risk_audit", "report"],
  "needs_research": true,
  "needs_backtest": true,
  "needs_risk_audit": true,
  "missing_params": [],
  "market": "US",
  "universe": ["AAPL"],
  "date_range": {}
}
```'''


def test_parse_strips_fence_and_extracts_constraints():
    out = parse_planner_output(RAW_WITH_FENCE)
    assert out.goal == "研究 AAPL 的 20 日动量并回测"
    assert out.steps == ["research", "quant", "risk_audit", "report"]
    # market/universe 属于约束字段，被正确解析
    assert out.universe == ["AAPL"]
    assert out.market == "US"


def test_parse_extracts_missing_params():
    out = parse_planner_output(
        '{"goal":"g","missing_params":["start_date"],"universe":["TSLA","NVDA"]}'
    )
    assert out.missing_params == ["start_date"]
    assert out.universe == ["TSLA", "NVDA"]


def test_build_task_plan_parses_stub_json():
    # stub 现在返回合法 JSON，正常解析，err 为 None
    class _Ctx:
        def __init__(self):
            self.llm = StubLLM()

    plan, constraints, err = build_task_plan(_Ctx(), "研究 AAPL 动量")
    assert err is None
    assert plan.goal == "研究 AAPL 动量"
    assert constraints.universe == ["AAPL", "MSFT", "NVDA"]


def test_build_task_plan_falls_back_on_garbage():
    # 注入非法输出 → 回退默认计划并返回解析错误
    class _Ctx:
        def __init__(self):
            self.llm = StubLLM(raw="这不是 JSON")

    plan, constraints, err = build_task_plan(_Ctx(), "研究 AAPL 动量")
    assert err is not None
    assert plan.steps  # 有默认步骤
    assert constraints.universe == []
