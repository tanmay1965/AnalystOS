"""Create clean, paginated PDF investigation reports for AnalystOS."""
from __future__ import annotations

import io
import math
import re
from difflib import SequenceMatcher
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    PageBreak, KeepTogether, HRFlowable, CondPageBreak,
)

# Consistent palette used throughout the PDF.
NAVY = colors.HexColor("#183153")
BLUE = colors.HexColor("#2457C5")
TEAL = colors.HexColor("#0F766E")      # darker teal: better contrast for headings on white
PALE_BLUE = colors.HexColor("#EEF4FF")
LIGHT_GREY = colors.HexColor("#F4F6F8")
MID_GREY = colors.HexColor("#64748B")
DARK = colors.HexColor("#172033")
WHITE = colors.white

# Register a Unicode font so the Indian rupee symbol renders correctly in PDFs.
# The full family (regular/bold/italic/bold-italic) is registered so that inline
# <b> and <i> tags inside Paragraphs resolve to real faces instead of failing.
_FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
try:
    pdfmetrics.registerFont(TTFont("AnalystOSUnicode", str(_FONT_DIR / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("AnalystOSUnicode-Bold", str(_FONT_DIR / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("AnalystOSUnicode-Italic", str(_FONT_DIR / "DejaVuSans-Oblique.ttf")))
    pdfmetrics.registerFont(TTFont("AnalystOSUnicode-BoldItalic", str(_FONT_DIR / "DejaVuSans-BoldOblique.ttf")))
    pdfmetrics.registerFontFamily(
        "AnalystOSUnicode",
        normal="AnalystOSUnicode", bold="AnalystOSUnicode-Bold",
        italic="AnalystOSUnicode-Italic", boldItalic="AnalystOSUnicode-BoldItalic",
    )
    BODY_FONT = "AnalystOSUnicode"
    BOLD_FONT = "AnalystOSUnicode-Bold"
    CURRENCY = "₹"
except Exception:
    BODY_FONT = "Helvetica"
    BOLD_FONT = "Helvetica-Bold"
    CURRENCY = "Rs. "   # Helvetica has no rupee glyph

# Titles that the exporter adds itself; they are never repeated from the report text.
_EXCLUDED_TITLES = {
    "analystos investigation report", "investigation report", "direct answer", "answer",
    "findings", "key findings", "calculated results", "colored visualization",
    "visualization", "what the data reveals",
}

_ID_TOKENS = {"id", "ids", "year", "row", "code", "pin", "pincode", "zip", "phone", "index", "sku"}
_MONEY_TOKENS = {"revenue", "price", "amount", "sales", "cost", "profit", "value", "sum",
                 "inr", "spend", "income", "fee", "fees", "charges", "salary"}
_NOT_MONEY_TOKENS = {"count", "qty", "quantity", "pct", "percent", "percentage", "ratio", "rate"}
_TIME_TOKENS = {"year", "month", "quarter", "week", "date", "period"}
_AUDIT_COLUMNS = ["issue_type", "data_row", "original_data_row", "column",
                  "details", "expected", "actual"]


# ----------------------------------------------------------------------------
# Number / text helpers
# ----------------------------------------------------------------------------
def _tokens(column: Any) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", str(column).lower()) if t]


def _is_id_like(column: Any) -> bool:
    """Identifier-style columns (order_id, year, data_row...) must not get thousand separators."""
    toks = _tokens(column)
    if "count" in toks or "total" in toks:
        return False
    return any(t in _ID_TOKENS for t in toks)


def _is_money_column(column: Any) -> bool:
    toks = set(_tokens(column))
    return bool(toks & _MONEY_TOKENS) and not (toks & _NOT_MONEY_TOKENS)


def _indian_number(value: Any, currency: bool = False) -> str:
    """Format numeric values using Indian comma grouping without scientific notation."""
    try:
        number = float(value)
        if pd.isna(number):
            return "—"
        rounded = round(number, 2)
        if rounded.is_integer():
            text = f"{int(rounded):,}"
        else:
            text = f"{rounded:,.2f}"
            if not currency:
                text = text.rstrip("0").rstrip(".")
        # Convert international grouping (1,234,567) to Indian grouping (12,34,567).
        sign = ""
        if text.startswith("-"):
            sign, text = "-", text[1:]
        whole, dot, decimal = text.partition(".")
        if len(whole.replace(",", "")) > 3:
            digits = whole.replace(",", "")
            last_three = digits[-3:]
            rest = digits[:-3]
            groups = []
            while rest:
                groups.insert(0, rest[-2:])
                rest = rest[:-2]
            whole = ",".join(groups + [last_three])
        text = sign + whole + (dot + decimal if dot else "")
        return f"{CURRENCY}{text}" if currency else text
    except (TypeError, ValueError, OverflowError):
        return str(value) if value is not None else "—"


def _trim(x: float, decimals: int) -> str:
    return f"{x:.{decimals}f}".rstrip("0").rstrip(".")


def _compact_number(value: Any, currency: bool = False) -> str:
    """Short Indian-style label (₹4.82 Cr, ₹31.2 L, 12.5 K) for chart axes and bar labels."""
    try:
        n = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(n):
        return "—"
    sign = "-" if n < 0 else ""
    a = abs(n)
    if a >= 1e7:
        s = f"{_trim(a / 1e7, 2)} Cr"
    elif a >= 1e5:
        s = f"{_trim(a / 1e5, 2)} L"
    elif a >= 1e3:
        s = f"{_trim(a / 1e3, 1)} K"
    else:
        s = _trim(a, 2)
    return f"{sign}{CURRENCY if currency else ''}{s}"


def _clean_text(value: Any) -> str:
    """Escape plain text for ReportLab Paragraphs and remove Markdown decoration."""
    text = "" if value is None else str(value)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"^\s*[-•]\s+", "", text)
    return escape(text).replace("\n", "<br/>")


def _inline_markup(value: Any) -> str:
    """Escape text and turn **bold** / *italic* into real PDF bold / italic (code ticks dropped)."""
    text = "" if value is None else str(value)
    text = escape(text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)([^*]+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", text)
    return text.replace("\n", "<br/>")


def _strip_markup(text: str) -> str:
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    return text


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", _strip_markup(text).lower()).strip()


def _metric_words(column: str) -> str:
    """'revenue_inr' -> 'revenue' for use in sentences."""
    words = [w for w in str(column).replace("_", " ").split() if w.lower() != "inr"]
    return " ".join(words).lower() or str(column)


def _pretty_header(column: Any) -> str:
    special = {"inr": "(INR)", "id": "ID", "pct": "%", "qty": "Qty", "sku": "SKU"}
    words = str(column).replace("_", " ").split()
    return " ".join(special.get(w.lower(), w.capitalize()) for w in words) or str(column)


# ----------------------------------------------------------------------------
# DataFrame helpers
# ----------------------------------------------------------------------------
def _as_dataframe(results: Any) -> pd.DataFrame:
    """Accept the common result shapes used by AnalystOS nodes."""
    if isinstance(results, pd.DataFrame):
        return results.copy()
    if isinstance(results, dict):
        for key in ("table", "data", "records", "rows", "result_table"):
            value = results.get(key)
            if isinstance(value, pd.DataFrame):
                return value.copy()
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return pd.DataFrame(value)
            if isinstance(value, dict):
                try:
                    return pd.DataFrame(value)
                except (ValueError, TypeError):
                    pass
    if isinstance(results, list) and results and isinstance(results[0], dict):
        return pd.DataFrame(results)
    return pd.DataFrame()


def _metric_value(results: Any, names: tuple[str, ...], default: Any = "N/A") -> Any:
    if isinstance(results, dict):
        for name in names:
            if name in results and results[name] is not None:
                return results[name]
        for parent in ("data_quality", "dataset_stats", "profile", "metrics"):
            nested = results.get(parent)
            if isinstance(nested, dict):
                for name in names:
                    if name in nested and nested[name] is not None:
                        return nested[name]
    return default


def _find_value_column(df: pd.DataFrame) -> str | None:
    if df.empty:
        return None
    preferred = ("value", "sum", "revenue_inr", "revenue", "total", "amount", "count", "mean")
    for name in preferred:
        if name in df.columns and pd.api.types.is_numeric_dtype(df[name]):
            return name
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
               and not pd.api.types.is_bool_dtype(df[c]) and not _is_id_like(c)]
    return numeric[0] if numeric else None


def _find_label_column(df: pd.DataFrame, value_col: str | None) -> str | None:
    if df.empty:
        return None
    for name in ("product", "region", "category", "customer_segment", "month", "date", "order_date", "name", "label"):
        if name in df.columns and name != value_col:
            return name
    return next((c for c in df.columns if c != value_col and not pd.api.types.is_numeric_dtype(df[c])), None)


def _format_cell(value: Any, column: str) -> str:
    if value is None:
        return "—"
    if not isinstance(value, (list, dict)):
        try:
            if pd.isna(value):
                return "—"
        except (TypeError, ValueError):
            pass
    is_number = (isinstance(value, (int, float, np.integer, np.floating))
                 and not isinstance(value, (bool, np.bool_)))
    if is_number:
        if _is_id_like(column):
            return str(int(value)) if float(value).is_integer() else str(value)
        return _indian_number(value, currency=_is_money_column(column))
    if hasattr(value, "strftime"):
        return value.strftime("%d %b %Y")
    return str(value)


def _audit_column_map(df: pd.DataFrame) -> dict[str, Any]:
    """Map normalised column names ('Issue Type' -> 'issue_type') to the real column labels."""
    return {re.sub(r"[\s\-]+", "_", str(c).strip().lower()): c for c in df.columns}


def _is_audit_table(df: pd.DataFrame) -> bool:
    """Audit tables are recognised by their columns, whatever the operation is called."""
    if df.empty:
        return False
    present = set(_audit_column_map(df)) & set(_AUDIT_COLUMNS)
    if "issue_type" in present:
        return len(present) >= 2                      # issue_type plus at least one traceability field
    return len(present) >= 4                          # no issue_type, but clearly audit-shaped


def _is_audit_report(results: Any, df: pd.DataFrame) -> bool:
    """Audit report = named as one by the operation OR shaped like one by its columns."""
    named = isinstance(results, dict) and results.get("operation") == "data_quality_audit"
    return named or _is_audit_table(df)


# ----------------------------------------------------------------------------
# Page furniture
# ----------------------------------------------------------------------------
def _draw_footer(canvas, doc) -> None:
    canvas.saveState()
    page_w, _ = doc.pagesize
    canvas.setStrokeColor(colors.HexColor("#D8DEE8"))
    canvas.setLineWidth(0.5)
    canvas.line(doc.leftMargin, 15 * mm, page_w - doc.rightMargin, 15 * mm)
    canvas.setFont(BODY_FONT, 8)
    canvas.setFillColor(MID_GREY)
    # Keep generation metadata in the footer so it cannot be stranded on a
    # mostly-empty final page after a long table.
    generated = datetime.now().strftime("Generated %d %b %Y, %I:%M %p")
    canvas.drawString(doc.leftMargin, 10 * mm, "AnalystOS  •  Investigation Report")
    canvas.drawCentredString(page_w / 2, 10 * mm, generated)
    canvas.drawRightString(page_w - doc.rightMargin, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


# ----------------------------------------------------------------------------
# Chart
# ----------------------------------------------------------------------------
def _shorten(label: str, limit: int = 22) -> str:
    return label if len(label) <= limit else label[: limit - 1].rstrip() + "…"


def _label_is_chartable(df: pd.DataFrame, label_col: str, value_col: str, operation: str) -> bool:
    """A label column is suitable only if it names distinct, short categories or periods."""
    if label_col == value_col:
        return False
    series = df[label_col]
    if pd.api.types.is_bool_dtype(series):
        return False
    tokens = set(_tokens(label_col))
    time_like = bool(tokens & _TIME_TOKENS) or pd.api.types.is_datetime64_any_dtype(series)
    if pd.api.types.is_numeric_dtype(series) and not (tokens & _TIME_TOKENS):
        return False                                  # bare numbers (ids, quantities) are not labels
    if "date" in tokens and operation != "time_trend":
        return False                                  # day-level dates mean row-level data
    pair = df[[label_col, value_col]].copy()
    pair[value_col] = pd.to_numeric(pair[value_col], errors="coerce")
    pair = pair.dropna()
    if len(pair) < 2:
        return False                                  # nothing to compare
    if pair[label_col].duplicated().any():
        return False                                  # repeated labels: bars would overlap
    if not time_like and len(pair) > 50:
        return False                                  # a long list of rows, not a summary
    if pair[label_col].astype(str).str.len().median() > 40:
        return False                                  # free text, not a category name
    return True


def _pick_chart_columns(df: pd.DataFrame, operation: str = "") -> tuple[str, str] | None:
    """Return (label_col, value_col) only when both are suitable for a chart, else None."""
    value_pref = ("value", "sum", "revenue_inr", "revenue", "total", "amount", "count", "mean")
    label_pref = ("product", "region", "category", "customer_segment", "month", "date", "order_date", "name", "label")
    value_order = [c for c in value_pref if c in df.columns] + [c for c in df.columns if c not in value_pref]
    label_order = [c for c in label_pref if c in df.columns] + [c for c in df.columns if c not in label_pref]
    for value_col in value_order:
        series = df[value_col]
        if (not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)
                or _is_id_like(value_col)):
            continue                                  # id / year / code columns are not metrics
        if pd.to_numeric(series, errors="coerce").notna().sum() < 2:
            continue
        for label_col in label_order:
            if _label_is_chartable(df, label_col, value_col, operation):
                return label_col, value_col
    return None


def _make_chart(df: pd.DataFrame, question: str = "", chart_type: str = "bar", operation: str = "", results: Any = None) -> bytes | None:
    if df.empty or operation == "data_quality_audit" or _is_audit_table(df):
        return None
    picked = _pick_chart_columns(df, operation)
    if picked is None:
        return None                                   # no suitable metric + label pair: skip the chart
    label_col, value_col = picked
    chart_df = df[[label_col, value_col]].copy()
    chart_df[value_col] = pd.to_numeric(chart_df[value_col], errors="coerce")
    chart_df = chart_df.dropna()
    if pd.api.types.is_datetime64_any_dtype(chart_df[label_col]):
        chart_df = chart_df.sort_values(label_col)
        chart_df[label_col] = chart_df[label_col].dt.strftime("%d %b %Y")
    elif pd.api.types.is_numeric_dtype(chart_df[label_col]):
        chart_df = chart_df.sort_values(label_col)    # e.g. a numeric year column
        chart_df[label_col] = chart_df[label_col].map(
            lambda v: str(int(v)) if float(v).is_integer() else str(v))
    else:
        chart_df[label_col] = chart_df[label_col].astype(str)

    is_time = operation == "time_trend" or bool(set(_tokens(label_col)) & _TIME_TOKENS)
    total_points = len(chart_df)
    if is_time:
        chart_df = chart_df.tail(36)          # keep chronological order, most recent periods
        truncated = total_points > len(chart_df)
    else:
        chart_df = chart_df.sort_values(value_col, ascending=False).head(15)
        truncated = total_points > len(chart_df)
    n = len(chart_df)
    xs = list(range(n))
    values = chart_df[value_col].tolist()
    labels = [_shorten(s) for s in chart_df[label_col].tolist()]

    q_lower = question.lower()
    metric_name = str((results or {}).get("metric", value_col)).replace("_", " ").strip().lower()
    label_name = str(label_col).replace("_", " ").title()
    if "revenue" in q_lower or "revenue" in metric_name:
        metric_label = "Revenue (INR)"
    elif "sales" in q_lower or "sales" in metric_name:
        metric_label = "Sales (INR)"
    else:
        metric_label = metric_name.title()
    is_inr = "(INR)" in metric_label or _is_money_column(value_col)

    fig, ax = plt.subplots(figsize=(8.8, 3.9), constrained_layout=True)
    box = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85)
    if chart_type == "line":
        ax.plot(xs, values, color="#2457C5", marker="o", linewidth=2, markersize=4)
        if n <= 12:
            targets = set(xs)
        else:   # too many points to label them all without overlap: label the extremes only
            targets = {int(np.argmax(values)), int(np.argmin(values))}
        hi = int(np.argmax(values))
        for i in sorted(targets):
            above = i == hi or n <= 12
            ax.annotate(_compact_number(values[i], is_inr), (xs[i], values[i]),
                        xytext=(0, 7 if above else -9), textcoords="offset points",
                        ha="center", va="bottom" if above else "top", fontsize=7.5, bbox=box)
    else:
        bar_colors = ["#2457C5" if v >= 0 else "#C0392B" for v in values]
        bars = ax.bar(xs, values, color=bar_colors, width=0.66)
        for bar, value in zip(bars, values):
            ax.annotate(_compact_number(value, is_inr),
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3 if value >= 0 else -3), textcoords="offset points",
                        ha="center", va="bottom" if value >= 0 else "top", fontsize=7.5)

    if operation == "time_trend":
        chart_title = f"Monthly {metric_label.replace(' (INR)', '')}"
    elif operation == "group_aggregate":
        chart_title = f"{metric_label.replace(' (INR)', '')} by {label_name}"
    else:
        chart_title = f"{metric_label} by {label_name}"
    if truncated:
        chart_title += f" (last {n} of {total_points})" if is_time else f" (top {n} of {total_points})"
    ax.set_title(chart_title, fontsize=12, fontweight="bold", color="#183153", pad=14)
    ax.set_ylabel(metric_label, color="#475569")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(y=0.15)

    # Thin the x labels when there are many, and angle them so they never overprint.
    step = max(1, math.ceil(n / 18))
    ax.set_xticks(xs[::step])
    short_labels = n <= 5 and max(len(s) for s in labels) <= 14
    ax.set_xticklabels(labels[::step],
                       rotation=0 if short_labels else 35,
                       ha="center" if short_labels else "right",
                       rotation_mode="default" if short_labels else "anchor", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: _compact_number(v, is_inr)))

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


# ----------------------------------------------------------------------------
# Result table
# ----------------------------------------------------------------------------
def _fit_widths(weights: list[float], mins: list[float], total: float) -> list[float]:
    """Proportional column widths that never drop below each column's unbreakable minimum."""
    n = len(weights)
    w = [total * x / sum(weights) for x in weights]
    for _ in range(6):
        short = [i for i in range(n) if w[i] < mins[i] - 0.01]
        if not short:
            break
        need = sum(mins[i] - w[i] for i in short)
        donors = [i for i in range(n) if i not in short and w[i] > mins[i] + 0.01]
        spare = sum(w[i] - mins[i] for i in donors)
        if spare <= 0:
            break
        take = min(need, spare)
        for i in short:
            w[i] += (mins[i] - w[i]) * take / need
        for i in donors:
            w[i] -= (w[i] - mins[i]) * take / spare
    return w


def _first_rows_height(table: Table, width: float, rows: int = 4) -> float:
    """Height of the header row plus the first few data rows (all of them for short tables)."""
    table.wrap(width, 10 ** 6)
    heights = list(getattr(table, "_rowHeights", None) or [])
    return float(sum(heights[:rows]))


def _make_result_table(df: pd.DataFrame, available_width: float, styles) -> Table | None:
    if df.empty:
        return None
    # Audit tables are wide: select the most useful traceability fields and use
    # landscape pages. Other reports retain their original result columns.
    is_audit = _is_audit_table(df)
    if is_audit:
        cmap = _audit_column_map(df)
        wanted = [name for name in _AUDIT_COLUMNS if name in cmap]
        display_df = df[[cmap[name] for name in wanted]].copy()
        display_df.columns = wanted                   # normalised names drive widths and headers
        if "issue_type" in display_df.columns:
            display_df["issue_type"] = display_df["issue_type"].astype(str).str.replace("_", " ").str.title()
    else:
        display_df = df.iloc[:, :12].copy()
    cols = list(display_df.columns)
    col_count = max(1, len(cols))

    # Font size adapts to how many columns have to share the page.
    # Audit tables prioritize fitting more readable rows per page; regular result
    # tables retain the larger default typography.
    size = (6.8 if col_count <= 6 else 6.5 if col_count <= 9 else 6.3) if is_audit else (
        8.5 if col_count <= 6 else 8 if col_count <= 9 else 7.5
    )
    # Compact audit typography deliberately increases rows per page for long findings.
    lead = size + (1.1 if is_audit else 3)
    pad = 1.8 if is_audit else 4
    head_left = ParagraphStyle("TH_L", parent=styles["TableHeader"], fontSize=size, leading=lead)
    head_right = ParagraphStyle("TH_R", parent=head_left, alignment=TA_RIGHT)
    cell_left = ParagraphStyle("TC_L", parent=styles["TableCell"], fontSize=size, leading=lead)
    cell_right = ParagraphStyle("TC_R", parent=cell_left, alignment=TA_RIGHT)
    numeric_cols = {
        c for c in cols
        if pd.api.types.is_numeric_dtype(display_df[c]) and not pd.api.types.is_bool_dtype(display_df[c])
    }

    header_texts = [_pretty_header(c) for c in cols]
    headers = [Paragraph(_clean_text(h), head_right if c in numeric_cols else head_left)
               for c, h in zip(cols, header_texts)]
    body = [[_format_cell(row[col], str(col)) for col in cols]
            for _, row in display_df.head(100).iterrows()]
    rows = [headers]
    for texts in body:
        rows.append([Paragraph(_clean_text(t), cell_right if c in numeric_cols else cell_left)
                     for c, t in zip(cols, texts)])

    # Column widths: audit tables keep their hand-tuned proportions; everything else is sized
    # from the actual content so short columns stay narrow and text columns get room.
    if is_audit and "details" in cols:
        proportions = {
            "issue_type": 0.13, "data_row": 0.07, "original_data_row": 0.09,
            "column": 0.11, "details": 0.36, "expected": 0.12, "actual": 0.12,
        }
        weights = [proportions.get(str(c), 1 / col_count) for c in cols]
    else:
        weights = []
        for j in range(col_count):
            lens = sorted(len(r[j]) for r in body) or [4]
            typical = lens[int(0.9 * (len(lens) - 1))]
            longest_head_word = max((len(w) for w in header_texts[j].split()), default=5)
            weights.append(max(min(typical, 36), longest_head_word, 5) + 2)

    # Minimum width: the longest unbreakable token (header word or number) must fit on one line.
    mins = []
    for j in range(col_count):
        tokens = header_texts[j].split()
        tokens += [t for r in body for t in r[j].split() if len(t) <= 24]
        widest = max((pdfmetrics.stringWidth(t, BODY_FONT, size) for t in tokens), default=20)
        mins.append(widest + 2 * pad + 1)
    col_widths = _fit_widths(weights, mins, available_width)

    table = Table(rows, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("VALIGN", (0, 0), (-1, 0), "MIDDLE"),
        ("VALIGN", (0, 1), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D5DCE7")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE_BLUE]),
        ("LEFTPADDING", (0, 0), (-1, -1), pad),
        ("RIGHTPADDING", (0, 0), (-1, -1), pad),
        ("TOPPADDING", (0, 0), (-1, -1), 1 if is_audit else 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1 if is_audit else 4),
    ]))
    return table


# ----------------------------------------------------------------------------
# Computed "direct answer" sentences
# ----------------------------------------------------------------------------
def _computed_ranking(df: pd.DataFrame, question: str) -> dict | None:
    """Generate a reliable lowest/highest statement from the returned result table."""
    if df.empty:
        return None
    q = question.lower()
    wants_low = bool(re.search(r"\b(lowest|least|smallest|minimum|min|bottom)\b", q))
    wants_high = bool(re.search(r"\b(highest|most|largest|maximum|max|top)\b", q))
    if not wants_low and not wants_high:
        return None
    value_col = _find_value_column(df)
    label_col = _find_label_column(df, value_col)
    if not value_col or not label_col:
        return None
    numeric = pd.to_numeric(df[value_col], errors="coerce")
    valid = df.loc[numeric.notna()].copy()
    if valid.empty:
        return None
    valid[value_col] = pd.to_numeric(valid[value_col], errors="coerce")
    words = _metric_words(value_col)

    def sentence(direction: str, row: pd.Series) -> str:
        return (f"{_clean_text(row[label_col])} has the {direction} {words} "
                f"({_format_cell(row[value_col], value_col)})")

    low_row = valid.loc[valid[value_col].idxmin()]
    high_row = valid.loc[valid[value_col].idxmax()]
    parts, labels, directions = [], [], set()
    if wants_high:
        parts.append(sentence("highest", high_row))
        labels.append(str(high_row[label_col]))
        directions.add("highest")
    if wants_low:
        parts.append(sentence("lowest", low_row))
        labels.append(str(low_row[label_col]))
        directions.add("lowest")
    # "text" is what the Direct Answer box shows; the rest lets Findings recognise restatements.
    return {"text": ", while ".join(parts) + ".", "labels": labels,
            "directions": directions, "metric": words}


def _computed_time_trend(df: pd.DataFrame, question: str) -> dict | None:
    """Summarize time-trend extrema and flag the current month if it may be incomplete."""
    if df.empty:
        return None
    value_col = _find_value_column(df)
    label_col = _find_label_column(df, value_col)
    if not value_col or not label_col:
        return None
    values = pd.to_numeric(df[value_col], errors="coerce")
    valid = df.loc[values.notna()].copy()
    if valid.empty:
        return None
    valid[value_col] = pd.to_numeric(valid[value_col], errors="coerce")
    highest = valid.loc[valid[value_col].idxmax()]
    lowest = valid.loc[valid[value_col].idxmin()]
    difference = float(highest[value_col]) - float(lowest[value_col])
    words = "revenue" if "revenue" in question.lower() else _metric_words(value_col)
    text = (
        f"Across {len(valid)} periods, the highest observed {words} was "
        f"{_format_cell(highest[value_col], value_col)} in {escape(str(highest[label_col]))}, while the lowest "
        f"was {_format_cell(lowest[value_col], value_col)} in {escape(str(lowest[label_col]))}. "
        f"The difference between the two is {_format_cell(difference, value_col)}."
    )
    # A YYYY-MM label matching the current month can represent a partial month.
    current_period = datetime.now().strftime("%Y-%m")
    periods = valid[label_col].astype(str).str.strip()
    if periods.eq(current_period).any():
        text += (
            f" The value for {current_period} may be incomplete because the current month has not ended; "
            "compare it with full months cautiously."
        )
    return {"text": text, "labels": [str(highest[label_col]), str(lowest[label_col])],
            "directions": {"highest", "lowest"}, "metric": words}


def _digit_keys(text: str) -> set[str]:
    """Digit signatures of the numbers in a text, ignoring commas, currency and trailing zeros."""
    keys = set()
    for m in re.findall(r"\d[\d,]*(?:\.\d+)?", text):
        digits = re.sub(r"\D", "", m)
        if len(digits) >= 3:
            keys.add(digits.rstrip("0") or "0")
    return keys


def _duplicates_answer(line: str, answer_norms: set[str], info: dict | None) -> bool:
    """True when a Findings line only restates what the Direct Answer box already says."""
    normalized = _norm(line)
    if not normalized:
        return False
    if normalized in answer_norms:
        return True
    if len(normalized) >= 20:
        for a in answer_norms:
            if len(a) >= 20 and (a in normalized or normalized in a
                                 or SequenceMatcher(None, a, normalized).ratio() >= 0.85):
                return True
    if info:
        if any(p in normalized for p in ("highest observed", "lowest observed",
                                         "highest group is", "lowest is")):
            return True
        labels = [_norm(l) for l in info["labels"]]
        hits = [l for l in labels if l and l in normalized]
        if hits:
            words = set(normalized.split())
            says_direction = bool(words & set(info["directions"]))
            says_metric = _norm(info["metric"]) in normalized
            same_number = bool(_digit_keys(_strip_markup(line)) & _digit_keys(info["text"]))
            if same_number or (says_direction and says_metric) or len(hits) == len(labels) >= 2:
                return True
    return False


def _first_answer(report: str) -> list[str]:
    """Return the lines of the first meaningful paragraph of the narrative report."""
    for para in re.split(r"\n\s*\n", report):
        if "AnalystOS Investigation Report" in para:
            continue
        kept: list[str] = []
        for raw in para.splitlines():
            s = raw.strip()
            if not s or s.startswith(("|", "---", "```")):
                continue
            s = re.sub(r"^#{1,6}\s*", "", s)
            n = _norm(re.sub(r"^\d+[.)]\s+", "", s))      # ignore numbering like "1. Direct Answer"
            if not n or n in _EXCLUDED_TITLES or n.startswith(("question ", "investigation question ")):
                continue
            s = re.sub(r"^[-•]\s+", "", s)
            kept.append(s)
        if kept:
            return kept
    return []


def _fmt_metric(value: Any) -> str | None:
    if isinstance(value, (dict, list, tuple, set)):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)):
        return _indian_number(value)
    return str(value)


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------
def generate_investigation_pdf(
    question: str,
    report: str | None = None,
    results: Any = None,
    *,
    report_text: str | None = None,
) -> tuple[bytes, str]:
    """Return a formatted PDF and filename.

    Supports both ``report=...`` and the older caller convention
    ``report_text=...`` so app.py does not fail on a keyword mismatch.
    """
    question = question or "Data investigation"
    # Backward-compatible alias used by some app.py versions.
    if report is None:
        report = report_text
    report = report or "No narrative report was provided."
    df = _as_dataframe(results)

    buffer = io.BytesIO()
    is_audit = _is_audit_report(results, df)
    # Wide result tables (more than 7 columns) need landscape pages to stay readable.
    wide_table = (not is_audit) and (not df.empty) and min(len(df.columns), 12) > 7
    use_landscape = is_audit or wide_table
    page_size = landscape(A4) if use_landscape else A4
    doc = SimpleDocTemplate(
        buffer, pagesize=page_size,
        rightMargin=12 * mm if use_landscape else 17 * mm,
        leftMargin=12 * mm if use_landscape else 17 * mm,
        topMargin=14 * mm, bottomMargin=21 * mm,
        title="AnalystOS Investigation Report", author="AnalystOS",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontName=BOLD_FONT,
                              fontSize=21, leading=26, textColor=NAVY, alignment=TA_LEFT, spaceAfter=5))
    styles.add(ParagraphStyle(name="Question", parent=styles["Normal"], fontName=BODY_FONT, fontSize=10.5, leading=15,
                              textColor=DARK, spaceAfter=12))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], fontName=BOLD_FONT,
                              fontSize=12.5, leading=15, textColor=TEAL, spaceBefore=8, spaceAfter=4,
                              keepWithNext=True))
    styles.add(ParagraphStyle(name="SectionFlow", parent=styles["Section"], keepWithNext=False))
    styles.add(ParagraphStyle(name="SubSection", parent=styles["Normal"], fontName=BOLD_FONT,
                              fontSize=10.5, leading=14, textColor=NAVY, spaceBefore=8, spaceAfter=3,
                              keepWithNext=True))
    styles.add(ParagraphStyle(name="BodyClean", parent=styles["BodyText"], fontName=BODY_FONT, fontSize=9.5, leading=12.5,
                              textColor=DARK, spaceAfter=3))
    styles.add(ParagraphStyle(name="BulletClean", parent=styles["BodyClean"], leftIndent=16, bulletIndent=3,
                              bulletFontName=BODY_FONT, spaceAfter=2))
    styles.add(ParagraphStyle(name="Answer", parent=styles["BodyClean"], fontSize=10.5, leading=16, spaceAfter=0))
    styles.add(ParagraphStyle(name="SmallMeta", parent=styles["Normal"], fontName=BODY_FONT, fontSize=8, leading=11,
                              textColor=MID_GREY))
    styles.add(ParagraphStyle(name="TableHeader", parent=styles["Normal"], fontName=BOLD_FONT, fontSize=8, leading=11,
                              textColor=WHITE))
    styles.add(ParagraphStyle(name="TableCell", parent=styles["Normal"], fontName=BODY_FONT, fontSize=8, leading=11,
                              textColor=DARK))
    styles.add(ParagraphStyle(name="MetricValue", parent=styles["Normal"], fontSize=13, leading=16,
                              fontName=BOLD_FONT, textColor=NAVY, alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="MetricLabel", parent=styles["Normal"], fontName=BODY_FONT, fontSize=8, leading=10,
                              textColor=MID_GREY, alignment=TA_CENTER))

    story = []
    story.append(Paragraph("AnalystOS Investigation Report", styles["ReportTitle"]))
    story.append(Paragraph(
        f'<font size="7.5" color="#64748B">INVESTIGATION QUESTION</font><br/>{_clean_text(question)}',
        styles["Question"]))
    story.append(HRFlowable(width="100%", thickness=1.2, color=BLUE, spaceAfter=10))

    # Dataset metrics strip: only tiles that actually have a value (no rows of "N/A").
    tile_defs = [
        ("Rows", ("rows", "row_count", "total_rows")),
        ("Columns", ("columns", "column_count", "total_columns")),
        ("Missing values", ("missing_values", "missing_count", "total_missing")),
        ("Duplicate rows", ("duplicate_rows", "duplicates", "duplicate_count")),
    ]
    tiles = []
    for label, names in tile_defs:
        shown = _fmt_metric(_metric_value(results, names, default=None)) if results is not None else None
        if shown is not None and shown != "None":
            tiles.append((label, shown))
    if tiles:
        metric_data = [[Paragraph(_clean_text(v), styles["MetricValue"]) for _, v in tiles],
                       [Paragraph(_clean_text(l), styles["MetricLabel"]) for l, _ in tiles]]
        metric_table = Table(metric_data, colWidths=[doc.width / len(tiles)] * len(tiles),
                             rowHeights=[9 * mm, 7 * mm])
        metric_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GREY),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#DCE3ED")),
            ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#DCE3ED")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.extend([metric_table, Spacer(1, 10)])

    # ---- Direct answer (shown as a highlighted callout) ----
    story.append(Paragraph("Direct Answer", styles["Section"]))
    computed_info = _computed_ranking(df, question)
    if computed_info is None and any(term in question.lower() for term in ("trend", "monthly", "over time", "by month")):
        computed_info = _computed_time_trend(df, question)
    computed = computed_info["text"] if computed_info else None
    answer_norms: set[str] = set()
    if computed:
        answer_html = computed
    else:
        answer_lines = _first_answer(report)
        answer_norms = {_norm(l) for l in answer_lines}
        answer_html = _inline_markup("\n".join(answer_lines)) if answer_lines \
            else "See the findings below for the analysis."
    answer_box = Table([[Paragraph(answer_html, styles["Answer"])]], colWidths=[doc.width])
    answer_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
        ("LINEBEFORE", (0, 0), (0, -1), 3, TEAL),
        ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    story.append(answer_box)
    story.append(Spacer(1, 4))

    # ---- Findings ----
    # Parse Markdown lines safely: keep real bullets / numbering, turn bold-only lines and
    # '#' headings into styled sub-headings, and never repeat titles the exporter adds itself.
    normalized_report = report.replace("\r\n", "\n").replace("\r", "\n")
    seen_lines: set[str] = set()
    findings: list[Any] = []
    # When the Direct Answer box is computed from the data, the report's own "Direct Answer"
    # paragraph says the same thing, so that one paragraph is not repeated under Findings.
    skip_answer_para, skipped_any = False, False
    for raw_line in normalized_report.splitlines():
        line = raw_line.strip()
        if not line:
            if skip_answer_para and skipped_any:
                skip_answer_para = False              # blank line ends the answer paragraph
            continue
        if line.startswith(("|", "---", "```")):
            continue
        kind, bullet = "body", None
        heading_match = re.match(r"^#{1,6}\s*(.*?)\s*$", line)
        if heading_match:
            kind, line = "heading", heading_match.group(1).strip()
        else:
            bullet_match = re.match(r"^[-•]\s+(.*)$", line) or re.match(r"^\*\s+(.*)$", line)
            number_match = re.match(r"^(\d+)[.)]\s+(.*)$", line)
            bold_only = re.fullmatch(r"\*\*(.+?)\*\*:?", line)
            if bold_only and len(bold_only.group(1)) <= 80:
                kind, line = "heading", bold_only.group(1).strip()
            elif bullet_match:
                kind, line, bullet = "bullet", bullet_match.group(1).strip(), "•"
            elif number_match:
                kind, line, bullet = "number", number_match.group(2).strip(), number_match.group(1) + "."
        if kind == "heading":
            line = re.sub(r"^\d+[.)]\s*", "", line).strip()
        if not line:
            continue
        normalized = _norm(line)
        if kind == "heading":
            skip_answer_para = normalized in {"direct answer", "answer"}
            skipped_any = False
        elif skip_answer_para:
            skipped_any = True
            continue
        if not normalized or normalized in _EXCLUDED_TITLES or normalized in seen_lines:
            continue
        if normalized.startswith("question ") or normalized.startswith("investigation question "):
            continue
        # Anything that only restates the Direct Answer box (exact, near-identical, or the same
        # label + figure) is dropped so the two sections never repeat each other.
        if _duplicates_answer(line, answer_norms, computed_info):
            continue
        seen_lines.add(normalized)
        if kind == "heading":
            findings.append(Paragraph(_inline_markup(line), styles["SubSection"]))
        elif kind in ("bullet", "number"):
            findings.append(Paragraph(_inline_markup(line), styles["BulletClean"], bulletText=bullet))
        else:
            findings.append(Paragraph(_inline_markup(line), styles["BodyClean"]))
    if findings:
        story.append(Paragraph("Findings", styles["Section"]))
        story.extend(findings)

    # ---- Visualization ----
    chart_png = _make_chart(
        df, question,
        chart_type=(results.get("chart_type", "bar") if isinstance(results, dict) else "bar"),
        operation=(results.get("operation", "") if isinstance(results, dict) else ""),
        results=results if isinstance(results, dict) else None,
    )
    if chart_png:
        story.append(Paragraph("Visualization", styles["Section"]))
        # Cap the height so a landscape page doesn't get an oversized chart.
        chart_image = Image(io.BytesIO(chart_png), width=doc.width,
                            height=min(doc.width * 0.44, 95 * mm), kind="proportional")
        chart_image.hAlign = "CENTER"
        story.append(KeepTogether([chart_image, Spacer(1, 5)]))

    # ---- Calculated results table ----
    result_table = _make_result_table(df, doc.width, styles) if not df.empty else None
    if result_table is not None:
        truncated_cols = len(df.columns) > 12 and not _is_audit_table(df)
        # Keep the heading (and optional note) together with the header row and the first few
        # data rows: if that much doesn't fit on this page the whole block moves to the next one,
        # otherwise the table starts right here and simply flows across pages.
        # Audit tables can be long and have tall wrapped rows. Only keep the
        # heading, header and first data row together; requiring four rows here
        # creates large blank gaps before the table on some pages.
        audit_table = _is_audit_table(df)
        lead_in = (30 if audit_table else 40) + (22 if truncated_cols else 0)
        rows_to_keep = 2 if audit_table else 4
        needed = lead_in + _first_rows_height(result_table, doc.width, rows=rows_to_keep) + 6
        # For audit tables, let ReportLab split the table naturally instead of moving
        # the entire table to a fresh page. The old CondPageBreak threshold often left
        # a nearly empty narrative page followed by a mostly blank table page.
        if not audit_table:
            story.append(CondPageBreak(needed))
        story.append(Paragraph("Calculated Results", styles["SectionFlow"]))
        if truncated_cols:
            story.append(Paragraph("Showing the first 12 columns to keep the table readable.", styles["SmallMeta"]))
            story.append(Spacer(1, 4))
        story.append(result_table)
        if len(df) > 100:
            story.append(Spacer(1, 5))
            story.append(Paragraph(
                f"Only the first 100 of {_indian_number(len(df))} result rows are included in this PDF.",
                styles["SmallMeta"]))

    # Generation timestamp is printed in the footer on every page. Do not add
    # it as a final story paragraph: ReportLab may push it to a mostly-empty page.
    doc.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", question).strip("_").lower()[:55] or "investigation"
    return pdf_bytes, f"analystos_{safe_name}.pdf"