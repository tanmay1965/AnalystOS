
"""Persistent SQLite storage for AnalystOS investigation history."""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "analystos_memory.db"
DB_PATH = Path(
    os.environ.get("ANALYSTOS_MEMORY_DB", str(DEFAULT_DB_PATH))
)


def _json_default(value: Any):
    if hasattr(value, "isoformat"):
        return value.isoformat()

    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass

    return str(value)


def _connect() -> sqlite3.Connection:
    """Open the existing SQLite database and ensure the table exists."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(str(DB_PATH), timeout=10)
    connection.row_factory = sqlite3.Row

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS investigations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            session_id TEXT NOT NULL DEFAULT 'default',
            question TEXT NOT NULL,
            final_report TEXT NOT NULL DEFAULT '',
            analysis_results TEXT NOT NULL DEFAULT '{}',
            analysis_plan TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL DEFAULT 'completed'
        )
        """
    )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_investigations_created_at
        ON investigations(created_at)
        """
    )

    connection.commit()
    return connection


def save_run(
    question: str | dict,
    final_report: str = "",
    analysis_results: dict | None = None,
    status: str = "completed",
    plan: Any = None,
    session_id: str = "default",
) -> int:
    """Save one run; a complete LangGraph state is also accepted."""

    if isinstance(question, dict):
        state = question

        question = str(state.get("question", ""))
        final_report = str(
            state.get("final_report", final_report) or ""
        )
        analysis_results = state.get(
            "analysis_results",
            analysis_results,
        )
        plan = state.get("analysis_plan", plan)
        status = str(state.get("run_status", status))
        session_id = str(state.get("session_id", session_id))

        if not state.get("validation_passed", True):
            status = "needs_review"

        if state.get("error"):
            status = "failed"

    question = str(question or "").strip()

    if not question:
        raise ValueError(
            "Cannot save an investigation without a question."
        )

    if analysis_results is None:
        analysis_results = {}

    if plan is None:
        plan = []

    created_at = datetime.now(timezone.utc).isoformat()

    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO investigations (
                created_at,
                session_id,
                question,
                final_report,
                analysis_results,
                analysis_plan,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                str(session_id or "default"),
                question,
                str(final_report or ""),
                json.dumps(
                    analysis_results,
                    default=_json_default,
                    ensure_ascii=False,
                ),
                json.dumps(
                    plan,
                    default=_json_default,
                    ensure_ascii=False,
                ),
                str(status or "completed"),
            ),
        )

        run_id = int(cursor.lastrowid)

    return run_id


def _decode_record(row: sqlite3.Row) -> dict:
    """Convert a SQLite row to a dictionary and decode JSON fields."""
    record = dict(row)

    for key, fallback in (
        ("analysis_results", {}),
        ("analysis_plan", []),
    ):
        try:
            record[key] = json.loads(record[key])
        except (TypeError, json.JSONDecodeError):
            record[key] = fallback

    return record


def recent_runs(
    limit: int = 5,
    session_id: str | None = None,
) -> list[dict]:
    """Return newest history records, across sessions by default."""

    limit = max(1, min(int(limit), 100))

    with _connect() as connection:
        if session_id is None:
            rows = connection.execute(
                """
                SELECT *
                FROM investigations
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT *
                FROM investigations
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (str(session_id), limit),
            ).fetchall()

    return [_decode_record(row) for row in rows]


def get_run(run_id: int) -> dict | None:
    """Return one investigation, or None if the ID does not exist."""

    with _connect() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM investigations
            WHERE id = ?
            """,
            (int(run_id),),
        ).fetchone()

    return _decode_record(row) if row is not None else None


def delete_run(run_id: int) -> bool:
    """
    Delete exactly one investigation by its unique ID.

    Returns True if one record was deleted, otherwise False.
    Other investigations are not affected.
    """

    with _connect() as connection:
        cursor = connection.execute(
            """
            DELETE FROM investigations
            WHERE id = ?
            """,
            (int(run_id),),
        )
        deleted = cursor.rowcount

    return deleted == 1


def clear_memory(session_id: str | None = None) -> int:
    """
    Delete all stored history, or only a specific session's history.

    This existing bulk-delete function is retained for compatibility.
    """

    with _connect() as connection:
        if session_id is None:
            cursor = connection.execute(
                "DELETE FROM investigations"
            )
        else:
            cursor = connection.execute(
                """
                DELETE FROM investigations
                WHERE session_id = ?
                """,
                (str(session_id),),
            )

        deleted = cursor.rowcount

    return int(deleted)
