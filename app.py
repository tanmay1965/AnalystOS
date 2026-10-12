
"""Streamlit UI for the integrated, LLM-free AnalystOS application."""

import uuid

import pandas as pd
import streamlit as st
from langgraph.types import Command

from agent.graph import analyst_graph
from agent.memory import delete_run, recent_runs, save_run
from agent.tools import create_chart
from agent.report_export import generate_investigation_pdf
from tools.data_profiler import profile_dataset

st.set_page_config(page_title="AnalystOS", page_icon="📊", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "active_result" not in st.session_state:
    st.session_state.active_result = None
if "active_thread_id" not in st.session_state:
    st.session_state.active_thread_id = None
if "pending_clarification" not in st.session_state:
    st.session_state.pending_clarification = False
if "saved_run_id" not in st.session_state:
    st.session_state.saved_run_id = None

st.title("📊 AnalystOS")
st.subheader("Autonomous Data Analyst — deterministic edition")
st.write(
    "Upload a CSV or Excel dataset and ask a question. AnalystOS uses a "
    "validated LangGraph workflow and Pandas to calculate results, create charts, "
    "check results, and save investigation history locally."
)
st.caption(
    "This version uses deterministic Python logic; "
    "no LLM or external AI API is called."
)

# ---------------------------------------------------------
# INVESTIGATION HISTORY WITH INDIVIDUAL DELETE
# ---------------------------------------------------------
with st.sidebar:
    st.header("🗂️ Investigation History")

    try:
        for record in recent_runs(limit=8):
            run_id = int(record["id"])
            label = f"#{run_id} · {record['question']}"

            with st.expander(label[:100]):
                st.caption(
                    f"{record['created_at']} · {record['status']}"
                )

                if record.get("final_report"):
                    st.markdown(record["final_report"])
                else:
                    st.caption(
                        "No report was saved for this investigation."
                    )

                # Each investigation has its own delete button.
                confirm_key = f"confirm_delete_run_{run_id}"

                if st.button(
                    "🗑️ Delete this investigation",
                    key=f"delete_run_{run_id}",
                    use_container_width=True,
                ):
                    st.session_state[confirm_key] = True

                if st.session_state.get(confirm_key, False):
                    st.warning(
                        f"Delete investigation #{run_id}? "
                        "This cannot be undone."
                    )

                    confirm_col, cancel_col = st.columns(2)

                    if confirm_col.button(
                        "Yes, delete",
                        key=f"confirm_delete_{run_id}",
                        type="primary",
                    ):
                        try:
                            if delete_run(run_id):
                                st.session_state.pop(confirm_key, None)

                                if (
                                    st.session_state.get("saved_run_id")
                                    == run_id
                                ):
                                    st.session_state.saved_run_id = None

                                st.success(
                                    f"Investigation #{run_id} deleted."
                                )
                                st.rerun()
                            else:
                                st.warning(
                                    "This investigation was already deleted."
                                )
                                st.session_state.pop(confirm_key, None)

                        except Exception as exc:
                            st.error(
                                f"Could not delete investigation: {exc}"
                            )

                    if cancel_col.button(
                        "Cancel",
                        key=f"cancel_delete_{run_id}",
                    ):
                        st.session_state.pop(confirm_key, None)
                        st.rerun()

    except Exception as exc:
        st.warning(f"History is unavailable: {exc}")


# ---------------------------------------------------------
# DATASET UPLOAD
# ---------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls"],
)

if uploaded_file is None:
    st.info("Upload a CSV or Excel file to get started.")
    st.stop()

try:
    if uploaded_file.name.lower().endswith(".csv"):
        try:
            df = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(
                uploaded_file,
                encoding="latin-1",
            )
    else:
        df = pd.read_excel(uploaded_file)

    if df.empty:
        st.warning("The uploaded dataset is empty.")
        st.stop()

except Exception as exc:
    st.error(f"Unable to read the uploaded dataset: {exc}")
    st.stop()

st.success(f"Loaded {uploaded_file.name}")

col1, col2, col3 = st.columns(3)
col1.metric("Rows", f"{len(df):,}")
col2.metric("Columns", f"{len(df.columns):,}")
col3.metric(
    "Missing Values",
    f"{int(df.isna().sum().sum()):,}",
)

with st.expander("📋 Dataset Preview", expanded=True):
    st.dataframe(df.head(20), use_container_width=True)

with st.expander("🔍 Dataset Profile"):
    try:
        profile = profile_dataset(df)
        st.json(profile)
    except Exception as exc:
        st.warning(f"Could not generate the full profile: {exc}")
        profile = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": [str(col) for col in df.columns],
            "missing_values": int(df.isna().sum().sum()),
        }


# ---------------------------------------------------------
# INVESTIGATION INPUT
# ---------------------------------------------------------
st.subheader("💬 Ask AnalystOS")

question = st.text_area(
    "What would you like to investigate?",
    placeholder=(
        "Example: Which region generates the most revenue? "
        "Compare all regions."
    ),
    height=90,
)

if st.button(
    "🔎 Investigate",
    type="primary",
    use_container_width=True,
):
    if not question.strip():
        st.warning("Please enter a question before investigating.")
    else:
        thread_id = str(uuid.uuid4())

        initial_state = {
            "question": question.strip(),
            "data_json": df.to_json(
                orient="split",
                date_format="iso",
                default_handler=str,
            ),
            "data_profile": profile,
            "session_id": st.session_state.session_id,
            "analysis_plan": {},
            "analysis_results": {},
            "findings": [],
            "validation_passed": False,
            "validation_errors": [],
            "retry_count": 0,
            "error": "",
            "final_report": "",
            "run_status": "running",
            "clarification_answer": "",
            "clarification_needed": False,
        }

        st.session_state.active_thread_id = thread_id
        st.session_state.active_result = None
        st.session_state.pending_clarification = False
        st.session_state.saved_run_id = None

        config = {"configurable": {"thread_id": thread_id}}

        with st.spinner("AnalystOS is investigating..."):
            try:
                result = analyst_graph.invoke(
                    initial_state,
                    config=config,
                )
            except Exception as exc:
                st.error(
                    f"Investigation failed unexpectedly: {exc}"
                )
                st.stop()

        st.session_state.active_result = result

        if result.get("__interrupt__"):
            st.session_state.pending_clarification = True
        else:
            try:
                st.session_state.saved_run_id = save_run(result)
            except Exception as exc:
                st.warning(
                    "Analysis completed, but history could not be saved: "
                    f"{exc}"
                )


# ---------------------------------------------------------
# CLARIFICATION AND RESUME
# ---------------------------------------------------------
if (
    st.session_state.pending_clarification
    and st.session_state.active_result
):
    result = st.session_state.active_result
    interrupts = result.get("__interrupt__", ())
    payload = (
        getattr(interrupts[0], "value", {})
        if interrupts else {}
    )

    st.subheader("🙋 Clarification needed")
    st.info(
        payload.get(
            "message",
            "Please clarify the analysis you want before AnalystOS continues.",
        )
    )

    suggestions = payload.get("suggestions", [])
    if suggestions:
        st.write("Examples:")
        for suggestion in suggestions:
            st.write(f"- {suggestion}")

    with st.form("clarification_form", clear_on_submit=True):
        clarification_answer = st.text_input(
            "What would you like AnalystOS to investigate?",
            placeholder="For example: Show monthly revenue trend.",
        )
        resume_clicked = st.form_submit_button(
            "▶ Resume investigation"
        )

    if resume_clicked:
        if not clarification_answer.strip():
            st.warning("Enter a clarification before resuming.")
        else:
            config = {
                "configurable": {
                    "thread_id": st.session_state.active_thread_id
                }
            }

            with st.spinner(
                "Resuming investigation with your clarification..."
            ):
                try:
                    result = analyst_graph.invoke(
                        Command(
                            resume=clarification_answer.strip()
                        ),
                        config=config,
                    )
                except Exception as exc:
                    st.error(
                        f"Could not resume the investigation: {exc}"
                    )
                    st.stop()

            st.session_state.active_result = result
            st.session_state.pending_clarification = bool(
                result.get("__interrupt__")
            )

            if not st.session_state.pending_clarification:
                try:
                    st.session_state.saved_run_id = save_run(result)
                except Exception as exc:
                    st.warning(
                        "Analysis completed, but history could not be saved: "
                        f"{exc}"
                    )

                st.rerun()


# ---------------------------------------------------------
# DISPLAY COMPLETED INVESTIGATION
# ---------------------------------------------------------
result = st.session_state.active_result

if (
    result
    and not st.session_state.pending_clarification
    and not result.get("__interrupt__")
):
    if st.session_state.saved_run_id:
        st.caption(
            "Investigation saved to SQLite · ID "
            f"{st.session_state.saved_run_id}"
        )

    if result.get("validation_passed"):
        st.success("Validation passed")
    elif result.get("run_status") == "failed":
        st.error("Investigation stopped after the retry limit.")
    else:
        st.warning("This investigation requires review.")

    plan = result.get("analysis_plan", {})
    st.subheader("🧭 Selected Analysis Plan")

    if isinstance(plan, dict):
        st.write(
            f"**Operation:** `{plan.get('operation', 'unknown')}`"
        )
        st.write(plan.get("rationale", "No rationale recorded."))
        st.json(plan)
    else:
        st.write(plan)

    results = result.get("analysis_results", {})
    chart = None

    if results:
        st.subheader("📊 Calculated Results")
        st.info(
            results.get(
                "summary",
                "Analysis results are available.",
            )
        )

        table = results.get("table", [])
        if table:
            st.dataframe(
                pd.DataFrame(table),
                use_container_width=True,
            )

        with st.expander(
            "Technical results and validation details"
        ):
            st.json(results)
            st.write(
                "**Validation errors:**",
                result.get("validation_errors", []),
            )
            st.write(
                "**Retry count:**",
                result.get("retry_count", 0),
            )

        try:
            chart = create_chart(results)
        except Exception as exc:
            st.warning(f"Unable to generate the chart: {exc}")

    if chart is not None:
        st.subheader("📈 Visualization")
        st.caption(
            "Use the camera icon in the chart toolbar to download "
            "a PNG screenshot for your hackathon submission."
        )

        st.plotly_chart(
            chart,
            use_container_width=True,
            config={
                "displaylogo": False,
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": "analystos_chart",
                    "height": 800,
                    "width": 1200,
                    "scale": 2,
                },
            },
        )

        st.download_button(
            "⬇️ Download interactive chart (.html)",
            data=chart.to_html(
                full_html=True,
                include_plotlyjs=True,
            ),
            file_name="analystos_chart.html",
            mime="text/html",
        )

    # -----------------------------------------------------
    # REPORT AND PDF DOWNLOAD
    # -----------------------------------------------------
    st.subheader("📄 Investigation Report")

    report = result.get(
        "final_report",
        "No report was generated.",
    )
    st.markdown(report)

    st.download_button(
        "⬇️ Download report (.md)",
        data=report,
        file_name="analystos_investigation_report.md",
        mime="text/markdown",
    )

    try:
        pdf_bytes, pdf_filename = generate_investigation_pdf(
            question=result.get(
                "question",
                question.strip() if question else "Investigation",
            ),
            report_text=report,
            results=results,
        )

        st.download_button(
            "📄 Download complete PDF report with graphs",
            data=pdf_bytes,
            file_name=pdf_filename,
            mime="application/pdf",
            key=(
                "download_pdf_"
                f"{st.session_state.saved_run_id or 'current'}"
            ),
            use_container_width=True,
        )

    except Exception as exc:
        st.warning(
            f"Could not generate the PDF report: {exc}"
        )

    if result.get("validation_errors"):
        with st.expander("Review validation issues"):
            for error in result["validation_errors"]:
                st.write(f"- {error}")
