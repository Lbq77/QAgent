"""Day 2 条件路由单元测试。"""
from qagent.graph import routing
from qagent.state import AgentState, TaskPlan, TaskStatus


def _state(**plan_kwargs) -> AgentState:
    return AgentState(task_id="t", user_query="q", task_plan=TaskPlan(**plan_kwargs))


def test_route_missing_params():
    s = _state(missing_params=["date_range"], needs_research=True)
    assert routing.route_after_planner(s) == "await_input"


def test_route_skip_research():
    s = _state(needs_research=False, needs_backtest=True)
    assert routing.route_after_planner(s) == "quant"


def test_route_research_only():
    s = _state(needs_research=True, needs_backtest=False)
    assert routing.route_after_research(s) == "report"


def test_route_reject():
    s = _state(needs_backtest=True)
    s.confirmation = {"action": "reject"}
    assert routing.route_after_confirm(s) == "end"


def test_route_failed():
    s = _state(needs_backtest=True)
    s.status = TaskStatus.FAILED
    assert routing.route_after_risk(s) == "error"
