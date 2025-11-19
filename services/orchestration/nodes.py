from __future__ import annotations

from services.agents.policies import REVIEWER_SYS, render_user_prompt
from services.core.evidence import collect_evidence
from services.core.mongo import get_parsed_for_docs
from services.core.repositories import get_application, list_docs_for_app, pg_conn
from services.eligibility.rules import evaluate_rules
from services.llm.ollama_client import DEFAULT_MODEL
from services.llm.ollama_client import generate as llm_generate
from services.observability.tracing import trace_llm


def _push(state: dict, msg: str) -> None:
    state.setdefault("steps", []).append(msg)


def extract(state: dict) -> dict:
    app_id = state["application_id"]
    state.setdefault("steps", []).append("plan: load application + documents")
    conn = pg_conn()
    app = get_application(conn, app_id)
    if not app:
        state["status"] = "error"
        state["steps"].append("observe: application not found")
        conn.close()
        return state
    docs = list_docs_for_app(conn, app_id)
    conn.close()

    parsed = get_parsed_for_docs([d["id"] for d in docs])
    state["artifacts"] = parsed
    state["app_snapshot"] = app
    state["steps"].append(f"act: loaded {len(docs)} docs, {len(parsed)} parsed artifacts")
    state["status"] = "ok"
    return state


def validate(state: dict) -> dict:
    app_id = state["application_id"]
    state["steps"].append("plan: validate required documents")
    conn = pg_conn()
    docs = list_docs_for_app(conn, app_id)
    conn.close()
    required = {"eid_front", "eid_back", "bank_statement"}
    have = {d["doc_type"] for d in docs}
    missing = sorted(list(required - have))
    state["validations"] = {"required": sorted(required), "have": sorted(have), "missing": missing}
    state["steps"].append(
        f"observe: missing={missing}" if missing else "observe: all required docs present"
    )
    state["status"] = "ok" if not missing else "incomplete"
    return state


def eligibility(state: dict) -> dict:
    state["steps"].append("plan: compute deterministic rules score")
    app_id = state["application_id"]
    conn = pg_conn()
    app = get_application(conn, app_id)
    docs = list_docs_for_app(conn, app_id)
    conn.close()
    parsed = state.get("artifacts", [])
    res = evaluate_rules(app, docs, parsed)
    state["rules_result"] = res
    state["steps"].append(f"observe: status={res.get('status')} score={res.get('score')}")
    state["ml_result"] = {"status": "skipped"}
    return state


def recommend(state: dict) -> dict:
    state["steps"].append("plan: generate reviewer summary")
    app = state.get("app_snapshot", {})
    rules = state.get("rules_result", {})
    evidence = collect_evidence(state.get("artifacts", []), max_chars=1200)
    sys = REVIEWER_SYS
    user = render_user_prompt(app, rules, evidence)
    prompt = f"<system>\n{sys}\n</system>\n<user>\n{user}\n</user>"
    text = llm_generate(prompt, model=DEFAULT_MODEL, temperature=0.2, timeout_s=60)
    state["recommendation"] = text
    state["steps"].append("observe: generated recommendation")
    try:
        trace_llm(
            event="agents.recommend",
            input_payload={"model": DEFAULT_MODEL, "applicant": app.get("applicant_full_name")},
            output_text=text,
            tags=["agents", "recommend"],
        )
    except Exception:
        pass
    state["status"] = (rules or {}).get("status", "ok")
    return state
