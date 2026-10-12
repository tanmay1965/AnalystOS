"""Safe, deterministic data-analysis tools for AnalystOS.

The planner maps common natural-language requests to a small set of reusable
Pandas operations. It never executes generated Python or SQL.
"""

from io import StringIO
import re
from typing import Any

import pandas as pd

SUPPORTED_OPERATIONS = {
    "profile",
    "metric_summary",
    "group_aggregate",
    "value_counts",
    "time_trend",
    "sort_and_limit",
    "data_quality_audit",
}
SUPPORTED_AGGREGATIONS = {"sum", "mean", "min", "max", "count"}
SUPPORTED_CHARTS = {"none", "bar", "line"}


def dataframe_from_state(state: dict) -> pd.DataFrame:
    """Get a DataFrame from either supported graph-state format."""
    df = state.get("df")
    if isinstance(df, pd.DataFrame):
        return df

    data_json = state.get("data_json")
    if not isinstance(data_json, str) or not data_json.strip():
        raise ValueError("Graph state must contain 'data_json' or a DataFrame in 'df'.")

    df = pd.read_json(StringIO(data_json), orient="split")
    if df.empty and len(df.columns) == 0:
        raise ValueError("The dataset is empty or invalid.")
    return df


def _normalise_name(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower().replace("_", " ").replace("-", " "))


def _find_mentioned_column(question: str, columns: list) -> Any:
    q = _normalise_name(question)
    matches = [col for col in columns if _normalise_name(col) and _normalise_name(col) in q]
    return max(matches, key=lambda col: len(_normalise_name(col))) if matches else None


def _find_metric_column(question: str, df: pd.DataFrame) -> Any:
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    if not numeric_columns:
        return None

    explicit = _find_mentioned_column(question, numeric_columns)
    if explicit is not None:
        return explicit

    q = _normalise_name(question)
    aliases = {
        "revenue": ["revenue", "sales", "income", "turnover"],
        "profit": ["profit", "margin"],
        "quantity": ["quantity", "units sold", "volume", "units"],
        "price": ["unit price", "price", "cost"],
    }
    for metric, words in aliases.items():
        if any(word in q for word in words):
            match = next((col for col in numeric_columns if metric in _normalise_name(col)), None)
            if match is not None:
                return match
            if metric == "revenue":
                match = next((col for col in numeric_columns if "sales" in _normalise_name(col)), None)
                if match is not None:
                    return match
    # Do not silently choose an arbitrary numeric column if the question
    # does not identify a metric.
    return None


def _find_group_column(question: str, df: pd.DataFrame) -> Any:
    # 'str' is supported in newer Pandas; fallback keeps older versions working.
    try:
        categorical_columns = df.select_dtypes(include=["object", "str", "category", "bool", "string"]).columns.tolist()
    except (TypeError, ValueError):
        categorical_columns = df.select_dtypes(include=["object", "category", "bool", "string"]).columns.tolist()

    explicit = _find_mentioned_column(question, categorical_columns)
    if explicit is not None:
        return explicit

    q = _normalise_name(question)
    preferred_terms = [
        "customer segment", "segment", "region", "category", "product",
        "department", "country", "city", "branch", "status",
    ]
    for term in preferred_terms:
        if term in q:
            match = next(
                (col for col in categorical_columns
                 if term in _normalise_name(col) and df[col].nunique(dropna=True) <= 100),
                None,
            )
            if match is not None:
                return match
    return None


def _find_date_column(df: pd.DataFrame) -> Any:
    possible_columns = [
        col for col in df.columns
        if any(token in _normalise_name(col) for token in ["date", "time", "timestamp"])
    ]
    for col in possible_columns:
        parsed = pd.to_datetime(df[col], errors="coerce")
        if len(df) and parsed.notna().mean() >= 0.5:
            return col
    return None


def _requested_direction(q: str) -> bool:
    """Return True for ascending/lowest requests, otherwise False."""
    low_terms = ["lowest", "smallest", "minimum", "bottom", "least", "cheapest", "fewest"]
    high_terms = ["highest", "largest", "maximum", "top", "most", "greatest", "best"]
    has_low = any(term in q for term in low_terms)
    has_high = any(term in q for term in high_terms)
    return has_low and not has_high


def build_fallback_plan(question: str, df: pd.DataFrame) -> dict:
    """Map natural-language questions to deterministic, validated operations."""
    q = _normalise_name(question)
    metric_col = _find_metric_column(question, df)
    group_col = _find_group_column(question, df)
    date_col = _find_date_column(df)

    audit_terms = (
        "data quality", "data-quality", "audit", "missing value", "missing values",
        "missing data", "duplicate", "duplicates", "inconsisten", "negative quantity",
        "invalid record", "invalid records", "anomal", "revenue mismatch", "mismatch",
    )
    audit_requested = any(term in q for term in audit_terms)
    if audit_requested:
        return {
            "operation": "data_quality_audit",
            "chart_type": "none",
            "rationale": "Audit missing values, duplicate rows, negative numeric values, and derived-revenue inconsistencies.",
        }

    trend_requested = any(word in q for word in [
        "trend", "over time", "monthly", "quarterly", "yearly", "growth", "decline", "timeline",
    ])
    count_requested = any(word in q for word in [
        "count", "frequency", "distribution", "how many", "most common", "most frequent",
    ])
    row_level_requested = any(word in q for word in [
        "order", "orders", "row", "rows", "record", "records", "transaction", "transactions",
    ])
    low_direction = _requested_direction(q)
    high_direction = any(term in q for term in ["highest", "largest", "maximum", "top", "most", "greatest", "best"])

    # For a trend, extrema words describe which period to report, not the
    # aggregation to apply within each month. Sum is the default business metric.
    aggregation = "mean" if any(word in q for word in ["average", "mean", "avg"]) else "sum"
    if not trend_requested and not row_level_requested and group_col is None:
        if any(word in q for word in ["minimum", "smallest", "lowest"]):
            aggregation = "min"
        elif any(word in q for word in ["maximum", "largest", "highest"]):
            aggregation = "max"

    requested_chart = "line" if any(term in q for term in ("line chart", "line graph", "as a line")) else (
        "bar" if any(term in q for term in ("bar chart", "bar graph", "as a bar")) else None
    )

    # Monthly/period trend takes precedence over category inference.
    if trend_requested and date_col is not None and metric_col is not None:
        chart_type = requested_chart or "line"
        return {
            "operation": "time_trend",
            "metric": str(metric_col),
            "date_column": str(date_col),
            "aggregation": aggregation,
            "chart_type": chart_type,
            "rationale": f"Aggregate {metric_col} by calendar month using {aggregation}; preserve the requested {chart_type} chart type.",
        }

    if row_level_requested and metric_col is not None and (high_direction or low_direction):
        match = re.search(r"\b(?:top|bottom)\s+(\d+)\b", q)
        limit = int(match.group(1)) if match else 1
        return {
            "operation": "sort_and_limit", "metric": str(metric_col),
            "ascending": bool(low_direction), "limit": max(1, min(limit, 100)),
            "chart_type": requested_chart or "none",
            "rationale": "Sort individual rows by the requested numerical metric.",
        }

    if group_col is not None and metric_col is not None:
        return {
            "operation": "group_aggregate", "metric": str(metric_col),
            "group_by": str(group_col), "aggregation": aggregation,
            "ascending": bool(low_direction), "chart_type": requested_chart or "bar",
            "rationale": "Compare a numerical metric across groups using the requested aggregation.",
        }

    if group_col is not None and count_requested:
        return {
            "operation": "value_counts", "column": str(group_col),
            "chart_type": requested_chart or "bar",
            "rationale": "Count the frequency of categorical values.",
        }

    if metric_col is not None:
        return {
            "operation": "metric_summary", "metric": str(metric_col),
            "aggregation": aggregation, "chart_type": requested_chart or "none",
            "rationale": "Calculate a numerical summary.",
        }

    return {
        "operation": "profile", "chart_type": "none",
        "rationale": "Use a dataset profile when the request cannot be mapped safely to a supported operation.",
    }


def validate_plan(plan: dict, df: pd.DataFrame) -> dict:
    """Validate operation, columns, aggregation, and limits before execution."""
    if not isinstance(plan, dict):
        raise ValueError("The analysis plan must be a dictionary.")
    result = dict(plan)
    operation = result.get("operation")
    if operation not in SUPPORTED_OPERATIONS:
        raise ValueError(f"Unsupported operation: {operation}")

    chart_type = result.get("chart_type", "none")
    if chart_type not in SUPPORTED_CHARTS:
        raise ValueError(f"Unsupported chart type: {chart_type}")
    result["chart_type"] = chart_type

    if operation in {"metric_summary", "group_aggregate", "time_trend", "sort_and_limit"}:
        metric = result.get("metric")
        if metric not in df.columns:
            raise ValueError(f"Unknown metric column: {metric}")
        if not pd.api.types.is_numeric_dtype(df[metric]):
            raise ValueError(f"Metric column '{metric}' must be numeric.")

    if operation in {"metric_summary", "group_aggregate", "time_trend"}:
        aggregation = result.get("aggregation", "sum")
        if aggregation not in SUPPORTED_AGGREGATIONS:
            raise ValueError(f"Unsupported aggregation: {aggregation}")
        result["aggregation"] = aggregation

    if operation == "group_aggregate":
        group_by = result.get("group_by")
        if group_by not in df.columns:
            raise ValueError(f"Unknown grouping column: {group_by}")
        if df[group_by].nunique(dropna=True) > 100:
            raise ValueError("Grouping column has more than 100 unique values.")
        result["ascending"] = bool(result.get("ascending", False))

    if operation == "value_counts":
        column = result.get("column")
        if column not in df.columns:
            raise ValueError(f"Unknown column: {column}")
        if df[column].nunique(dropna=True) > 100:
            raise ValueError("Column has more than 100 unique values.")

    if operation == "time_trend":
        date_column = result.get("date_column")
        if date_column not in df.columns:
            raise ValueError(f"Unknown date column: {date_column}")
        parsed = pd.to_datetime(df[date_column], errors="coerce")
        if not parsed.notna().any():
            raise ValueError(f"Date column '{date_column}' has no valid dates.")

    if operation == "sort_and_limit":
        limit = result.get("limit", 1)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("Limit must be an integer between 1 and 100.")
        result["ascending"] = bool(result.get("ascending", False))
        result["limit"] = limit

    return result


def _safe_scalar(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _profile_dataset(df: pd.DataFrame) -> dict:
    numeric_summary = {}
    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        numeric_summary = numeric_df.describe().round(2).to_dict()
        numeric_summary = {
            str(column): {str(stat): _safe_scalar(value) for stat, value in stats.items()}
            for column, stats in numeric_summary.items()
        }
    return {
        "operation": "profile",
        "summary": f"The dataset contains {len(df):,} rows and {len(df.columns):,} columns.",
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_summary": numeric_summary,
        "table": [],
        "chart_type": "none",
    }


def execute_analysis(plan: dict, df: pd.DataFrame) -> dict:
    """Execute one supported, validated operation with Pandas."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a Pandas DataFrame.")
    if df.empty:
        raise ValueError("Cannot analyze an empty dataset.")

    validated_plan = validate_plan(plan, df)
    operation = validated_plan["operation"]
    base = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "chart_type": validated_plan.get("chart_type", "none"),
    }

    if operation == "data_quality_audit":
        issues = []
        # Keep source row numbers human-readable (CSV header is row 1, first data row is row 2).
        for col in df.columns:
            for idx in df.index[df[col].isna()].tolist():
                issues.append({"issue_type": "missing_value", "data_row": int(df.index.get_loc(idx)) + 1,
                               "row_index": int(idx), "column": str(col), "details": f"Missing value in {col}.",
                               "expected": None, "actual": None})
        duplicate_mask = df.duplicated(keep="first")
        # Report both the duplicate's source data row and the first matching row.
        # Data row numbers are 1-based and exclude the CSV header.
        for idx in df.index[duplicate_mask].tolist():
            duplicate_position = int(df.index.get_loc(idx))
            current_values = df.iloc[duplicate_position]
            prior_match_position = next(
                (
                    pos for pos in range(duplicate_position)
                    if df.iloc[pos].equals(current_values)
                ),
                None,
            )
            issues.append({
                "issue_type": "duplicate_row",
                "data_row": duplicate_position + 1,
                "original_data_row": (prior_match_position + 1) if prior_match_position is not None else None,
                "row_index": int(idx),
                "column": "(entire row)",
                "details": (
                    f"Exact duplicate of data row {prior_match_position + 1}."
                    if prior_match_position is not None else "Exact duplicate of an earlier record."
                ),
                "expected": "unique row",
                "actual": "duplicate",
            })
        for col in df.select_dtypes(include="number").columns:
            for idx in df.index[pd.to_numeric(df[col], errors="coerce") < 0].tolist():
                issues.append({"issue_type": "negative_value", "data_row": int(df.index.get_loc(idx)) + 1,
                               "row_index": int(idx), "column": str(col), "details": f"Negative value in {col}; review whether valid.",
                               "expected": "non-negative where business rules require it", "actual": _safe_scalar(df.at[idx, col])})
        qty_col = next((c for c in df.columns if _normalise_name(c) in {"quantity", "qty", "units sold"}), None)
        price_col = next((c for c in df.columns if "unit price" in _normalise_name(c) or _normalise_name(c) in {"price", "unit cost"}), None)
        revenue_col = next((c for c in df.columns if "revenue" in _normalise_name(c)), None)
        if qty_col is not None and price_col is not None and revenue_col is not None:
            qty = pd.to_numeric(df[qty_col], errors="coerce")
            price = pd.to_numeric(df[price_col], errors="coerce")
            revenue = pd.to_numeric(df[revenue_col], errors="coerce")
            expected_revenue = qty * price
            mismatch_mask = revenue.notna() & expected_revenue.notna() & ((revenue - expected_revenue).abs() > 0.01)
            for idx in df.index[mismatch_mask].tolist():
                issues.append({"issue_type": "revenue_mismatch", "data_row": int(df.index.get_loc(idx)) + 1,
                               "row_index": int(idx), "column": str(revenue_col),
                               "details": f"Recorded revenue does not equal {qty_col} × {price_col}.",
                               "expected": _safe_scalar(expected_revenue.loc[idx]), "actual": _safe_scalar(revenue.loc[idx])})
            missing_revenue_mask = revenue.isna() & expected_revenue.notna()
            for idx in df.index[missing_revenue_mask].tolist():
                # Already reported as missing_value; add the expected derived value for audit context.
                issues.append({"issue_type": "revenue_missing_expected", "data_row": int(df.index.get_loc(idx)) + 1,
                               "row_index": int(idx), "column": str(revenue_col),
                               "details": "Revenue is missing although quantity and unit price are available.",
                               "expected": _safe_scalar(expected_revenue.loc[idx]), "actual": None})
        issue_counts = pd.Series([item["issue_type"] for item in issues]).value_counts().to_dict() if issues else {}
        summary = f"Data-quality audit found {len(issues)} issue entries across {len(df):,} data rows."
        return {**base, "operation": operation, "summary": summary,
                "issue_counts": {str(k): int(v) for k, v in issue_counts.items()},
                "table": issues, "chart_type": "none"}

    if operation == "profile":
        result = _profile_dataset(df)
        result.update(base)
        return result

    if operation == "metric_summary":
        metric = validated_plan["metric"]
        aggregation = validated_plan["aggregation"]
        series = pd.to_numeric(df[metric], errors="coerce").dropna()
        if series.empty:
            raise ValueError(f"No valid numeric values found for {metric}.")
        value = _safe_scalar(getattr(series, aggregation)())
        return {
            **base, "operation": operation, "metric": str(metric),
            "aggregation": aggregation, "value": value,
            "valid_count": int(series.count()),
            "summary": f"The {aggregation} of {metric} is {float(value):,.2f}, based on {int(series.count()):,} valid observations.",
            "table": [{"metric": str(metric), "aggregation": aggregation, "value": value, "valid_count": int(series.count())}],
        }

    if operation == "sort_and_limit":
        metric = validated_plan["metric"]
        ascending = validated_plan["ascending"]
        limit = validated_plan["limit"]
        working = df.copy()
        working[metric] = pd.to_numeric(working[metric], errors="coerce")
        working = working.dropna(subset=[metric])
        if working.empty:
            raise ValueError(f"No valid numeric values found for {metric}.")
        result_df = working.sort_values(by=metric, ascending=ascending, kind="stable").head(limit)
        records = [{str(column): _safe_scalar(value) for column, value in row.items()} for _, row in result_df.iterrows()]
        direction = "lowest" if ascending else "highest"
        return {
            **base, "operation": operation, "metric": str(metric),
            "ascending": ascending, "limit": limit,
            "summary": f"Returned {len(records)} row(s) sorted by {metric}, from {direction} to {'highest' if ascending else 'lowest'}.",
            "table": records,
        }

    if operation == "group_aggregate":
        metric = validated_plan["metric"]
        group_by = validated_plan["group_by"]
        aggregation = validated_plan["aggregation"]
        working = df[[group_by, metric]].copy()
        working[metric] = pd.to_numeric(working[metric], errors="coerce")
        grouped = working.groupby(group_by, dropna=False, observed=False)[metric].agg(["sum", "mean", "min", "max", "count"])
        grouped["value"] = grouped[aggregation]
        grouped = grouped.sort_values("value", ascending=validated_plan.get("ascending", False), na_position="last")
        table = []
        for key, row in grouped.iterrows():
            table.append({
                str(group_by): "(Missing)" if pd.isna(key) else str(key),
                "value": _safe_scalar(row["value"]), "sum": _safe_scalar(row["sum"]),
                "mean": _safe_scalar(row["mean"]), "min": _safe_scalar(row["min"]),
                "max": _safe_scalar(row["max"]), "count": int(row["count"]),
            })
        best = table[0] if table else None
        direction = "lowest" if validated_plan.get("ascending", False) else "highest"
        summary = f"Compared {metric} by {group_by} using {aggregation} and sorted from {direction} to the other groups."
        if best:
            summary += f" The leading group is {best[str(group_by)]}, with a value of {best['value']:,.2f}."
        return {
            **base, "operation": operation, "metric": str(metric), "group_by": str(group_by),
            "aggregation": aggregation, "ascending": validated_plan.get("ascending", False),
            "summary": summary, "table": table,
        }

    if operation == "value_counts":
        column = validated_plan["column"]
        counts = df[column].value_counts(dropna=False).head(100)
        table = [{str(column): "(Missing)" if pd.isna(key) else str(key), "count": int(count)} for key, count in counts.items()]
        return {**base, "operation": operation, "column": str(column), "summary": f"Calculated the frequency of values in {column}.", "table": table}

    if operation == "time_trend":
        metric = validated_plan["metric"]
        date_column = validated_plan["date_column"]
        aggregation = validated_plan["aggregation"]
        working = df[[date_column, metric]].copy()
        working[date_column] = pd.to_datetime(working[date_column], errors="coerce")
        working[metric] = pd.to_numeric(working[metric], errors="coerce")
        working = working.dropna(subset=[date_column, metric])
        if working.empty:
            raise ValueError("No valid date and metric pairs were found.")
        working["period"] = working[date_column].dt.to_period("M").astype(str)
        grouped = working.groupby("period")[metric].agg(aggregation).sort_index()
        table = [{"period": str(period), "value": _safe_scalar(value)} for period, value in grouped.items()]
        return {**base, "operation": operation, "metric": str(metric), "date_column": str(date_column), "aggregation": aggregation, "summary": f"Calculated monthly {aggregation} {metric} across {len(table)} periods.", "table": table}

    raise ValueError(f"Unsupported operation: {operation}")


def validate_result(result: dict, df: pd.DataFrame) -> dict:
    """Validate result structure and input counts before report generation."""
    errors = []
    if not isinstance(result, dict):
        errors.append("The result must be a dictionary.")
    else:
        if result.get("rows") != len(df):
            errors.append("Row count does not match the input dataset.")
        if result.get("columns") != len(df.columns):
            errors.append("Column count does not match the input dataset.")
        if result.get("operation") not in SUPPORTED_OPERATIONS:
            errors.append("The result has an unsupported operation.")
        if not isinstance(result.get("summary"), str) or not result.get("summary"):
            errors.append("The result summary is missing.")
        if not isinstance(result.get("table"), list):
            errors.append("The result table must be a list.")
        if result.get("operation") == "data_quality_audit":
            table = result.get("table", [])
            if not isinstance(table, list):
                errors.append("Data-quality audit table must be a list.")
            elif any(not isinstance(item, dict) or "issue_type" not in item or "row_index" not in item for item in table):
                errors.append("Data-quality audit rows must include issue_type and row_index.")
        if result.get("operation") == "time_trend":
            table = result.get("table", [])
            if result.get("aggregation") != "mean" and "sum" not in str(result.get("summary", "")).lower() and result.get("aggregation") == "sum":
                errors.append("Time-trend summary does not describe the selected sum aggregation.")
            periods = [item.get("period") for item in table if isinstance(item, dict)]
            if periods != sorted(periods):
                errors.append("Time-trend periods are not in chronological order.")
        if result.get("operation") == "sort_and_limit":
            metric = result.get("metric")
            table = result.get("table", [])
            if metric not in df.columns:
                errors.append("Sorted result references an unknown metric column.")
            elif not table:
                errors.append("Sorted result contains no rows.")
            else:
                expected = pd.to_numeric(df[metric], errors="coerce").max()
                if not result.get("ascending", False) and result.get("limit", 1) == 1:
                    actual = pd.to_numeric(pd.Series([table[0].get(metric)]), errors="coerce").iloc[0]
                    if pd.notna(expected) and actual != expected:
                        errors.append("Highest-row result does not match the dataset maximum.")
                elif result.get("ascending", False) and result.get("limit", 1) == 1:
                    expected = pd.to_numeric(df[metric], errors="coerce").min()
                    actual = pd.to_numeric(pd.Series([table[0].get(metric)]), errors="coerce").iloc[0]
                    if pd.notna(expected) and actual != expected:
                        errors.append("Lowest-row result does not match the dataset minimum.")
    return {"passed": not errors, "errors": errors}


def create_chart(result: dict):
    """Create a chart that respects the planned chart type and calculated table."""
    import plotly.express as px
    chart_type = result.get("chart_type", "none")
    operation = result.get("operation")
    table = result.get("table", [])
    if chart_type == "none" or not table or operation == "data_quality_audit":
        return None
    plotted = pd.DataFrame(table)

    def draw(x, y, title, labels=None):
        if x not in plotted.columns or y not in plotted.columns:
            return None
        if chart_type == "line":
            return px.line(plotted, x=x, y=y, markers=True, title=title, labels=labels or {})
        return px.bar(plotted, x=x, y=y, title=title, labels=labels or {})

    if operation == "group_aggregate":
        group_by = result.get("group_by")
        return draw(group_by, "value", f"{result.get('aggregation', 'sum').title()} of {result.get('metric', 'metric')} by {group_by}",
                    {group_by: str(group_by).replace("_", " ").title(), "value": str(result.get("metric", "Value")).replace("_", " ").title()})
    if operation == "value_counts":
        column = result.get("column")
        return draw(column, "count", f"Value distribution for {column}", {"count": "Count"})
    if operation == "time_trend":
        return draw("period", "value", f"Monthly {str(result.get('metric', 'metric')).replace('_', ' ').title()}",
                    {"period": "Month", "value": str(result.get("metric", "Value")).replace("_", " ").title()})
    if operation == "sort_and_limit":
        metric = result.get("metric")
        label = next((c for c in plotted.columns if c != metric and not pd.api.types.is_numeric_dtype(plotted[c])), None)
        if label:
            return draw(label, metric, f"{metric.replace('_', ' ').title()} by {label.replace('_', ' ').title()}")
    return None
