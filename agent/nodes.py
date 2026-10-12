"""Deterministic nodes for the AnalystOS LangGraph workflow (no LLM)."""

import pandas as pd
from langgraph.types import interrupt

from agent.tools import (
    build_fallback_plan,
    dataframe_from_state,
    execute_analysis,
    validate_plan,
    validate_result,
)


PROFILE_INTENT_WORDS = (
    "profile", "summary", "summarize", "describe dataset", "dataset overview",
    "missing value", "missing data", "duplicate", "column names", "how many rows",
    "how many columns", "data quality", "numeric statistics", "numerical statistics", "audit", "inconsistency", "negative quantity", "revenue mismatch",
)


def clarification_node(state: dict) -> dict:
    """Pause for user input when a request is too vague to map to a supported tool."""
    df = dataframe_from_state(state)
    question = str(state.get("question", "")).strip()
    normalised_question = question.lower().replace("_", " ")

    try:
        tentative_plan = build_fallback_plan(question, df)
        is_profile_fallback = tentative_plan.get("operation") == "profile"
    except Exception:
        is_profile_fallback = True

    explicit_profile_request = any(
        phrase in normalised_question for phrase in PROFILE_INTENT_WORDS
    )

    # Specific profile/data-quality requests do not need clarification.
    if not is_profile_fallback or explicit_profile_request:
        return {"clarification_needed": False}

    answer = interrupt({
        "message": (
            "Your request is broad, so AnalystOS needs one detail before it can "
            "choose a supported analysis operation. What would you like to investigate?"
        ),
        "original_question": question,
        "suggestions": [
            "Which region generates the most revenue?",
            "Show the monthly revenue trend.",
            "Which product has the highest quantity sold?",
            "Summarize missing values and duplicate rows.",
        ],
    })

    answer = str(answer or "").strip()
    if not answer:
        # The UI prevents empty answers; this guards API callers too.
        return {
            "clarification_needed": True,
            "validation_errors": ["Clarification answer was empty."],
        }

    return {
        "question": f"{question}\nUser clarification: {answer}",
        "clarification_answer": answer,
        "clarification_needed": False,
        "validation_errors": [],
    }


def planner_node(state: dict) -> dict:
    """Select a supported operation; use profile as a safe retry fallback."""
    df = dataframe_from_state(state)
    question = str(state.get("question", ""))
    retry_count = int(state.get("retry_count", 0))
    prior_errors = state.get("validation_errors", [])

    try:
        if prior_errors and retry_count >= 1:
            plan = {
                "operation": "profile",
                "chart_type": "none",
                "rationale": "Safe fallback selected after the first attempt failed validation.",
            }
        else:
            plan = build_fallback_plan(question, df)
        plan = validate_plan(plan, df)
        return {
            "analysis_plan": plan,
            "validation_errors": [],
            "error": "",
            "findings": [
                "A deterministic plan was selected from supported Pandas operations."
            ],
        }
    except Exception as exc:
        fallback = validate_plan(
            {
                "operation": "profile",
                "chart_type": "none",
                "rationale": "Safe planning fallback.",
            },
            df,
        )
        return {
            "analysis_plan": fallback,
            "validation_errors": [],
            "error": "",
            "findings": [
                f"Planner selected a safe profile fallback ({type(exc).__name__})."
            ],
        }



def execute_tool_node(state: dict) -> dict:
    """Execute the approved operation against the restored DataFrame."""

    try:
        df = dataframe_from_state(state)

        results = execute_analysis(
            state["analysis_plan"],
            df,
        )

        return {
            "analysis_results": results,
            "error": "",
            "validation_errors": [],
        }

    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"

        return {
            "analysis_results": {},
            "error": message,
            "validation_errors": [message],
        }



def critic_node(state: dict) -> dict:
    """Validate result structure, dataset counts, and execution status."""
    df = dataframe_from_state(state)
    results = state.get("analysis_results", {})
    errors = list(state.get("validation_errors", []))

    if state.get("error"):
        errors.append(f"Tool execution failed: {state['error']}")

    check = validate_result(results, df) if results else {
        "passed": False,
        "errors": ["Analysis returned no results."],
    }
    errors.extend(check["errors"])
    question = str(state.get("question", "")).lower()
    plan = state.get("analysis_plan", {}) or {}
    operation = results.get("operation")
    if any(term in question for term in ("audit", "data quality", "missing value", "duplicate", "inconsisten", "negative quantity", "revenue mismatch")):
        if operation != "data_quality_audit":
            errors.append("Question asks for a data-quality audit but the selected operation is not data_quality_audit.")
    if any(term in question for term in ("monthly", "trend", "over time", "timeline")):
        if operation != "time_trend":
            errors.append("Question asks for a time trend but the selected operation is not time_trend.")
    if "line chart" in question and plan.get("chart_type") != "line":
        errors.append("The user explicitly requested a line chart, but the plan does not use line.")
    if "bar chart" in question and plan.get("chart_type") != "bar":
        errors.append("The user explicitly requested a bar chart, but the plan does not use bar.")
    errors = list(dict.fromkeys(errors))
    passed = not errors and bool(results)
    current_retries = int(state.get("retry_count", 0))

    return {
        "validation_passed": passed,
        "validation_errors": [] if passed else errors,
        "retry_count": current_retries if passed else current_retries + 1,
        "error": "" if passed else (state.get("error") or "; ".join(errors)),
        "findings": list(state.get("findings", [])) + (
            ["Automated result consistency checks passed."]
            if passed else [
                "Validation detected an issue; the workflow will retry once if possible."
            ]
        ),
    }


def route_after_critic(state: dict) -> str:
    """Route success to report, one failure to retry, repeated failures to stop."""
    if state.get("validation_passed"):
        return "respond"
    if int(state.get("retry_count", 0)) < 2:
        return "retry"
    return "failure"


def response_node(state: dict) -> dict:
    """Generate a readable Markdown report using only calculated evidence."""
    df = dataframe_from_state(state)
    results = state.get("analysis_results", {})
    question = str(state.get("question", ""))
    plan = state.get("analysis_plan", {})
    lines = [
        "# AnalystOS Investigation Report",
        "",
        f"**Question:** {question}",
        "",
        "## 1. Direct Answer",
        "",
        results.get("summary", "The analysis completed."),
        "",
        "## 2. What the Data Reveals",
        "",
    ]

    table = results.get("table", [])
    operation = results.get("operation")

    if operation == "data_quality_audit":
        counts = results.get("issue_counts", {})
        if counts:
            lines.append("The audit found the following issue types:")
            for issue_type, count in sorted(counts.items()):
                lines.append(f"- **{issue_type.replace('_', ' ').title()}**: {count}")
        else:
            lines.append("No configured data-quality issues were detected.")
        lines.append("Review the affected data rows below before using these records for business decisions.")
    elif operation == "group_aggregate" and table:
        group_col = results["group_by"]
        metric = results["metric"]
        best = table[0]
        worst = table[-1]
        best_value = best.get("value")
        worst_value = worst.get("value")
        if isinstance(best_value, (int, float)) and isinstance(worst_value, (int, float)):
            difference = float(best_value) - float(worst_value)
            lines.append(
                f"The results compare **{metric}** across **{group_col}**. "
                f"The highest group is **{best.get(group_col)}** at **{best_value:,.2f}**, "
                f"while the lowest is **{worst.get(group_col)}** at **{worst_value:,.2f}**. "
                f"The difference between them is **{difference:,.2f}**."
            )
            if results.get("aggregation") == "sum":
                total = sum(
                    float(row["value"]) for row in table
                    if isinstance(row.get("value"), (int, float))
                )
                if total:
                    lines.append(
                        f"The leading group represents approximately "
                        f"**{float(best_value) / total * 100:.1f}%** of the total."
                    )
        else:
            lines.append("Group values were returned but could not be compared numerically.")
        lines.append(
            "These figures describe the observed difference; they do not establish its cause."
        )

    elif operation == "metric_summary":
        value = results.get("value")
        if isinstance(value, (int, float)):
            lines.append(
                f"The {results.get('aggregation', 'selected summary')} of "
                f"**{results.get('metric', 'the metric')}** is **{value:,.2f}**, "
                f"based on {results.get('valid_count', 0):,} valid observations."
            )
        else:
            lines.append("A numerical summary was not available for the selected metric.")

    elif operation == "time_trend" and table:
        values = [row for row in table if isinstance(row.get("value"), (int, float))]
        if values:
            highest = max(values, key=lambda row: row["value"])
            lowest = min(values, key=lambda row: row["value"])
            lines.append(
                f"Across {len(values)} monthly periods, the highest observed value was "
                f"**{highest['value']:,.2f}** in **{highest['period']}**, and the lowest was "
                f"**{lowest['value']:,.2f}** in **{lowest['period']}**."
            )
            if len(values) > 1 and highest["value"] != 0:
                change = highest["value"] - lowest["value"]
                percent = change / abs(highest["value"]) * 100
                lines.append(
                    f"The difference between these two extremes is **{change:,.2f}** "
                    f"({percent:.1f}% of the highest month's value). This compares the "
                    "extremes; it does not establish a continuous trend or explain the cause."
                )
            lines.append("The timeline shows when values differed, but not why they changed.")

    elif operation == "value_counts" and table:
        top = table[0]
        column = results.get("column", "")
        lines.append(
            f"The most frequent listed value in **{column or 'the selected column'}** "
            f"is **{top.get(column, 'Unknown')}**, appearing "
            f"**{top.get('count', 0):,}** times. Frequency alone does not measure revenue or profit."
        )

    elif operation == "profile":
        summary = results.get("numeric_summary", {})
        if summary:
            for column, stats in list(summary.items())[:5]:
                if isinstance(stats, dict) and isinstance(stats.get("mean"), (int, float)):
                    lines.append(f"The mean of **{column}** is **{stats['mean']:,.2f}**.")
        else:
            lines.append("No numeric columns were available for summary statistics.")

    if table:
        columns = list(table[0].keys())
        lines.extend([
            "", "### Calculated Results", "",
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ])
        row_limit = 100 if operation == "data_quality_audit" else 20
        for row in table[:row_limit]:
            cells = [
                str(row.get(col, "")).replace("|", "\\|").replace("\n", " ")
                for col in columns
            ]
            lines.append("| " + " | ".join(cells) + " |")
        if len(table) > row_limit:
            lines.append(f"\nShowing the first {row_limit} of {len(table):,} rows.")

    lines.extend(["", "## 3. Business Interpretation", ""])
    if operation == "group_aggregate":
        lines.append(
            "The comparison identifies which groups contribute higher and lower values for "
            "the selected metric. To investigate the drivers, compare products or periods "
            "within the leading group. For revenue decisions, check profit or margin too."
        )
    elif operation == "time_trend":
        lines.append(
            "The time series highlights periods worth investigating. Compare with seasonality, "
            "promotions, or recorded business events before attributing a cause."
        )
    elif operation == "value_counts":
        lines.append(
            "The frequency distribution reveals common values. Assess business importance using "
            "a relevant metric rather than frequency alone."
        )
    else:
        lines.append(
            "Use the summary with business context and other relevant measures before making decisions."
        )

    lines.extend([
        "", "## 4. Recommended Next Checks", "",
        "- Verify source records for unusually high or low values.",
        "- Compare another relevant dimension or period if the data supports it.",
        "- Review completeness and complementary measures before deciding on an action.",
        "", "## 5. Data Quality and Limitations", "",
        f"- Rows: {len(df):,}",
        f"- Columns: {len(df.columns):,}",
        f"- Missing values: {int(df.isna().sum().sum()):,}",
        f"- Duplicate rows: {int(df.duplicated().sum()):,}",
        "- Calculations describe the uploaded dataset and do not prove causation.",
        "", "## Investigation Process", "",
    ])

    if isinstance(plan, dict):
        plan_items = [
            f"Selected operation: {plan.get('operation', 'not specified')}",
            f"Rationale: {plan.get('rationale', 'not specified')}",
        ]
    else:
        plan_items = [str(item) for item in plan] if plan else []
    lines.extend(f"{i}. {item}" for i, item in enumerate(plan_items, start=1))
    lines.extend([
        "", "## Validation", "",
        "Automated checks confirmed the result structure and dataset row/column counts.",
    ])
    return {"final_report": "\n".join(lines), "run_status": "completed"}


def failure_node(state: dict) -> dict:
    """Stop after bounded retry attempts and explain the failure."""
    errors = state.get("validation_errors", [])
    details = "\n".join(f"- {error}" for error in errors) or "- No detailed error was recorded."
    return {
        "run_status": "failed",
        "final_report": (
            "# AnalystOS Investigation Could Not Be Validated\n\n"
            "The analysis failed its validation checks after the retry limit. "
            "No verified conclusions should be drawn from this run.\n\n"
            f"## Issues\n\n{details}"
        ),
    }


# Compatibility aliases for older references.
def analyst_node(state: dict) -> dict:
    return execute_tool_node(state)


def report_node(state: dict) -> dict:
    return response_node(state)
