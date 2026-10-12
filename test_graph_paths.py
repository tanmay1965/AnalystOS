"""Graph transition tests: success, clarification/resume, retry, and terminal failure."""

import uuid

import pandas as pd
from langgraph.types import Command

from agent import nodes
from agent.graph import analyst_graph


def make_state(question: str) -> dict:
    df = pd.DataFrame({
        "region": ["North", "South", "East", "South"],
        "customer_segment": ["SMB", "Enterprise", "Consumer", "Enterprise"],
        "revenue_inr": [1000, 2500, 1800, 3000],
        "quantity": [2, 5, 3, 6],
        "order_date": ["2026-01-01", "2026-01-12", "2026-02-03", "2026-02-10"],
    })
    return {
        "question": question,
        "data_json": df.to_json(orient="split", date_format="iso", default_handler=str),
        "data_profile": {},
        "session_id": "test-session",
        "analysis_plan": {},
        "analysis_results": {},
        "findings": [],
        "validation_passed": False,
        "validation_errors": [],
        "retry_count": 0,
        "error": "",
        "final_report": "",
        "run_status": "running",
        "clarification_needed": False,
        "clarification_answer": "",
    }


def config() -> dict:
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def test_success_routes_to_report():
    result = analyst_graph.invoke(
        make_state("Which region generates the most revenue?"),
        config=config(),
    )
    assert result["validation_passed"] is True
    assert result["run_status"] == "completed"
    assert "South" in result["final_report"]


def test_ambiguous_question_interrupts_and_resumes():
    thread_config = config()
    paused = analyst_graph.invoke(make_state("Analyze the data"), config=thread_config)

    assert paused.get("__interrupt__"), "Expected the graph to pause for clarification."

    resumed = analyst_graph.invoke(
        Command(resume="Show the monthly revenue trend"),
        config=thread_config,
    )

    assert not resumed.get("__interrupt__")
    assert resumed["validation_passed"] is True
    assert resumed["analysis_plan"]["operation"] == "time_trend"
    assert "monthly periods" in resumed["final_report"]


def test_failed_first_attempt_retries_with_profile_then_succeeds(monkeypatch):
    real_execute = nodes.execute_analysis
    calls = {"count": 0}

    def fail_once(plan, df):
        calls["count"] += 1
        if calls["count"] == 1:
            raise ValueError("intentional first-attempt failure")
        return real_execute(plan, df)

    monkeypatch.setattr(nodes, "execute_analysis", fail_once)
    result = analyst_graph.invoke(
        make_state("Which region generates the most revenue?"),
        config=config(),
    )

    assert calls["count"] == 2
    assert result["validation_passed"] is True
    assert result["analysis_plan"]["operation"] == "profile"
    assert result["retry_count"] == 1


def test_repeated_failure_stops_after_one_retry(monkeypatch):
    calls = {"count": 0}

    def always_fail(plan, df):
        calls["count"] += 1
        raise ValueError("intentional repeated failure")

    monkeypatch.setattr(nodes, "execute_analysis", always_fail)
    result = analyst_graph.invoke(
        make_state("Which region generates the most revenue?"),
        config=config(),
    )

    assert calls["count"] == 2  # one original attempt plus one retry
    assert result["validation_passed"] is False
    assert result["retry_count"] == 2
    assert result["run_status"] == "failed"
    assert "Could Not Be Validated" in result["final_report"]


def test_explicit_profile_request_does_not_interrupt():
    result = analyst_graph.invoke(
        make_state("Summarize the dataset and report missing values."),
        config=config(),
    )
    assert not result.get("__interrupt__")
    assert result["validation_passed"] is True
