from __future__ import annotations

import os
from typing import Any

import psycopg2

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/ssai")


def pg_conn():
    """One-liner Postgres connection (caller must close)."""
    return psycopg2.connect(DATABASE_URL)


def list_docs_for_app(conn, app_id: str) -> list[dict[str, Any]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT DISTINCT ON (doc_type)
               id, doc_type, filename, content_type, size_bytes, created_at
        FROM documents
        WHERE application_id = %s
        ORDER BY doc_type, created_at DESC
        """,
        (app_id,),
    )
    items = [
        {
            "id": r[0],
            "doc_type": r[1],
            "filename": r[2],
            "content_type": r[3],
            "size_bytes": int(r[4] or 0),
        }
        for r in cur.fetchall()
    ]
    cur.close()
    return items


def get_application(conn, app_id: str) -> dict[str, Any] | None:
    cur = conn.cursor()
    cur.execute(
        "select id, applicant_full_name, applicant_national_id, household_size, monthly_income from applications where id=%s",
        (app_id,),
    )
    row = cur.fetchone()
    cur.close()
    if not row:
        return None
    return {
        "id": row[0],
        "applicant_full_name": row[1],
        "applicant_national_id": row[2],
        "household_size": row[3],
        "monthly_income": float(row[4]) if row[4] is not None else None,
    }
