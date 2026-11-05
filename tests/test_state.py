from qagent.state import AgentState, TaskStatus


def test_default_state():
    s = AgentState(task_id="t1", user_query="hello")
    assert s.status == TaskStatus.PENDING
    assert s.task_plan.missing_params == []
    assert s.evidence_refs == []


def test_task_status_values():
    assert {s.value for s in TaskStatus} == {
        "pending",
        "running",
        "succeeded",
        "failed",
        "blocked",
        "skipped",
        "cancelled",
    }
