from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class RuleResult:
    id: str
    passed: bool
    weight: float
    reason: str


def _pct(n: int, d: int) -> float:
    return 0.0 if d == 0 else round(100.0 * n / d, 1)


def evaluate_rules(
    application: dict[str, Any],
    docs: list[dict[str, Any]],
    parsed: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return status + score + per-rule rationale (deterministic)."""
    rules: list[RuleResult] = []

    # R1: Required docs present (ID front/back + bank_statement)
    required = {"eid_front", "eid_back", "bank_statement"}
    have = {d["doc_type"] for d in docs}
    missing = list(required - have)
    rules.append(
        RuleResult(
            id="required_docs",
            passed=len(missing) == 0,
            weight=0.35,
            reason="Missing: " + ", ".join(missing) if missing else "All required docs present",
        )
    )

    # R2: Documents parsed successfully (Mongo artifacts exist)
    req_docs = [d for d in docs if d["doc_type"] in required]
    have_artifacts = {p["document_id"] for p in parsed}
    parsed_ok = sum(1 for d in req_docs if d["id"] in have_artifacts)
    rules.append(
        RuleResult(
            id="parsed_ok",
            passed=(len(req_docs) > 0 and parsed_ok == len(req_docs)),
            weight=0.20,
            reason=f"Parsed {parsed_ok}/{len(req_docs)} required documents",
        )
    )

    # R3: Income vs. household size (basic affordability band)
    # (example bands; tune later)
    hh = int(application.get("household_size") or 1)
    inc = float(application.get("monthly_income") or 0.0)
    min_need = 1500 + 500 * max(0, hh - 1)  # AED; simple baseline
    rules.append(
        RuleResult(
            id="income_band",
            passed=inc < min_need,  # lower income → higher need → pass
            weight=0.30,
            reason=f"Income {inc:.0f} vs baseline {min_need:.0f} for household {hh}",
        )
    )

    # R4: Bank statement contains some text/tables (sanity)
    bank_doc_ids = {d["id"] for d in docs if d["doc_type"] == "bank_statement"}
    bank_parsed = [p for p in parsed if p["document_id"] in bank_doc_ids]
    has_data = any((p.get("chunks") or p.get("tables")) for p in bank_parsed)
    rules.append(
        RuleResult(
            id="bank_data_present",
            passed=has_data,
            weight=0.15,
            reason="Bank data detected" if has_data else "No bank text/tables parsed",
        )
    )

    eid_ids = {d["id"] for d in docs if d["doc_type"] in {"eid_front", "eid_back"}}
    eid_parsed = [p for p in parsed if p["document_id"] in eid_ids]
    eid_text = " ".join(
        [" ".join(c.get("text", "") for c in (p.get("chunks") or [])) for p in eid_parsed]
    )
    # UAE IDs are 15 digits often grouped; we just look for 10+ digits as a simple signal
    nid_regex = re.compile(r"\b\d{10,18}\b")
    found_nid = bool(nid_regex.search(eid_text))
    rules.append(
        RuleResult(
            id="eid_contains_id_number",
            passed=found_nid,
            weight=0.10,
            reason="ID number detected in EID OCR" if found_nid else "No ID-like number detected",
        )
    )

    # Aggregate score: sum weights of passed rules (0..1)
    score = round(sum(r.weight for r in rules if r.passed), 2)
    status = "ready" if score >= 0.80 else "needs_info" if score >= 0.50 else "incomplete"

    return {
        "status": status,
        "score": score,
        "rules": [r.__dict__ for r in rules],
        "coverage": {"docs_have_artifacts_pct": _pct(parsed_ok, len(docs) or 1)},
    }
