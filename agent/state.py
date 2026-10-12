"""Shared state for the deterministic AnalystOS LangGraph workflow."""

from typing import Any, TypedDict


class AnalystState(TypedDict, total=False):
    question: str
    # JSON-encoded dataframe prevents checkpoint serialization issues.
    data_json: str
    # Optional compatibility field for uncheckpointed direct calls/tests.
    df: Any
    data_profile: dict
    session_id: str
    analysis_plan: dict | list[str]
    analysis_results: dict
    findings: list[str]
    validation_passed: bool
    validation_errors: list[str]
    retry_count: int
    error: str
    final_report: str
    run_status: str
    clarification_answer: str
    clarification_needed: bool
