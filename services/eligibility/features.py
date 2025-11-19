from __future__ import annotations

import math

from services.core.mongo import get_parsed_for_docs
from services.core.repositories import get_application, list_docs_for_app, pg_conn


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _bank_has_tables(art: dict) -> bool:
    return bool(art.get("tables"))


def _count_text_chunks(art: dict) -> int:
    ch = art.get("chunks") or []
    return len([c for c in ch if (c.get("text") or "").strip()])


def extract_features(app_id: str) -> dict:
    """
    Deterministic features from Postgres + Mongo parsed artifacts.
    Keep schema simple & explainable; all numeric.
    """
    conn = pg_conn()
    app = get_application(conn, app_id)
    docs = list_docs_for_app(conn, app_id)
    conn.close()

    if not app:
        raise ValueError("application not found")

    parsed = get_parsed_for_docs([d["id"] for d in docs])
    have = {d["doc_type"] for d in docs}
    required = {"eid_front", "eid_back", "bank_statement"}

    # basic application features
    monthly_income = _safe_float(app.get("monthly_income"))
    household_size = _safe_float(app.get("household_size"))
    income_per_capita = (
        monthly_income / household_size if (household_size and household_size > 0) else float("nan")
    )

    # document coverage
    missing_count = len(required - have)
    has_bank = 1 if "bank_statement" in have else 0
    has_eid_front = 1 if "eid_front" in have else 0
    has_eid_back = 1 if "eid_back" in have else 0

    # parsed evidence
    bank_arts = [p for p in parsed if p.get("doc_type") == "bank_statement"]
    bank_tables = 1 if any(_bank_has_tables(p) for p in bank_arts) else 0
    total_chunks = sum(_count_text_chunks(p) for p in parsed)
    # simple normalization
    chunk_bins = min(total_chunks, 100)

    # Final feature vector (order is stable)
    feats = {
        "monthly_income": 0.0 if math.isnan(monthly_income) else monthly_income,
        "household_size": 0.0 if math.isnan(household_size) else household_size,
        "income_per_capita": 0.0 if math.isnan(income_per_capita) else income_per_capita,
        "missing_required_count": float(missing_count),
        "has_bank_statement": float(has_bank),
        "has_eid_front": float(has_eid_front),
        "has_eid_back": float(has_eid_back),
        "bank_tables_present": float(bank_tables),
        "parsed_chunk_bins": float(chunk_bins),
    }
    return feats


def feature_order() -> list[str]:
    return [
        "monthly_income",
        "household_size",
        "income_per_capita",
        "missing_required_count",
        "has_bank_statement",
        "has_eid_front",
        "has_eid_back",
        "bank_tables_present",
        "parsed_chunk_bins",
    ]
