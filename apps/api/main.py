# apps/api/main.py

import json
import logging  # added: structured logging for ops and Bandit compliance
import mimetypes
import os
import re
import uuid
from pathlib import Path
from typing import Any

import psycopg2
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from psycopg2.extras import Json
from pydantic import BaseModel, Field
from pymongo import ASCENDING, MongoClient

from apps.api.auth import PWD_HASH, USERNAME, create_access_token, verify_password
from apps.api.settings import settings
from services.agents.graph import runner as agent_runner
from services.core.evidence import collect_evidence as _collect_evidence
from services.core.mongo import get_parsed_for_docs
from services.core.repositories import get_application, list_docs_for_app
from services.core.repositories import pg_conn
from services.core.repositories import pg_conn as _conn
from services.eligibility.model import predict_with_explanations, train_from_applications
from services.eligibility.rules import evaluate_rules
from services.llm.ollama_client import DEFAULT_MODEL
from services.llm.ollama_client import generate as llm_generate
from services.observability.tracing import trace_llm
from services.ocr.image_ocr import ocr_image
from services.ocr.pdf_parser import chunk_pages, extract_pages_text
from services.ocr.xlsx_parser import parse_xlsx_tables
from services.rag.embeddings import embed
from services.rag.vector_store import search as vs_search
from services.rag.vector_store import upsert_chunks as vs_upsert

# --- logging setup (module-level) ---
logger = logging.getLogger(__name__)

app = FastAPI(title="SSA API", version="0.0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/ssai")
STORAGE_ROOT = os.getenv("STORAGE_ROOT", "/data/raw")
SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")  # defensive
MONGO_URL = os.getenv("MONGO_URL", "mongodb://mongo:27017")
MONGO_DB = os.getenv("MONGO_DB", "ssai")


def get_mongo():
    client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=2000)
    db = client[MONGO_DB]
    col = db["parsed_artifacts"]
    # ensure indexes once
    col.create_index([("application_id", ASCENDING)])
    col.create_index([("document_id", ASCENDING)])
    col.create_index([("doc_type", ASCENDING), ("application_id", ASCENDING)])
    return col


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


def get_parsed_for_docs(doc_ids: list[str]) -> list[dict[str, Any]]:
    col = get_mongo()
    return list(col.find({"document_id": {"$in": doc_ids}}, {"chunks": 0}))


def _flatten_artifacts_for_embed(artifact: dict) -> list[dict]:
    """
    Returns [{'text': ..., 'page': n}] from either chunks (pdf/image) or tables (xlsx rows).
    """
    out: list[dict] = []
    chunks = artifact.get("chunks") or []
    for c in chunks:
        t = c.get("text", "")
        if t.strip():
            out.append({"text": t, "page": c.get("page")})
    tables = artifact.get("tables") or []
    for t in tables:
        rows = t.get("rows", [])[:200]  # cap
        for r in rows:
            kv = ", ".join(f"{k}={r[k]}" for k in list(r.keys())[:6])
            out.append({"text": f"{t.get('name','table')} | {kv}", "page": None})
    return out


ALLOWED_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
}


class ApplicationIn(BaseModel):
    applicant_full_name: str = Field(min_length=2)
    applicant_national_id: str | None = None
    household_size: int | None = None
    monthly_income: float | None = None


class ApplicationOut(BaseModel):
    id: str


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/applications", response_model=ApplicationOut)
def create_application(payload: ApplicationIn):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        app_id = str(uuid.uuid4())
        cur.execute(
            """
            insert into applications (id, applicant_full_name, applicant_national_id, household_size, monthly_income)
            values (%s, %s, %s, %s, %s)
            """,
            (
                app_id,
                payload.applicant_full_name,
                payload.applicant_national_id,
                payload.household_size,
                payload.monthly_income,
            ),
        )
        cur.execute(
            "insert into audit_log(application_id, actor, action, payload) values (%s, %s, %s, %s)",
            (app_id, "user", "created_application", None),
        )
        conn.commit()
        cur.close()
        conn.close()
        return {"id": app_id}
    except Exception as e:  # noqa: BLE001
        # ruff B904: preserve original traceback
        raise HTTPException(status_code=500, detail=f"db error: {e}") from e


def _conn():
    return psycopg2.connect(DATABASE_URL)


@app.post("/applications/{app_id}/documents")
def upload_document(
    app_id: str,
    doc_type: str = Form(...),  # noqa: B008  (FastAPI pattern)
    file: UploadFile = File(...),  # noqa: B008  (FastAPI pattern)
):
    # 1) basic sanity
    if not re.fullmatch(r"[0-9a-fA-F-]{36}", app_id):
        raise HTTPException(400, "invalid application_id")
    # ensure application exists
    try:
        conn = _conn()
        cur = conn.cursor()
        cur.execute("select 1 from applications where id=%s", (app_id,))
        if cur.fetchone() is None:
            cur.close()
            conn.close()
            raise HTTPException(404, "application not found")
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"db error: {e}") from e

    # 2) validate content type
    ctype = file.content_type or mimetypes.guess_type(file.filename or "")[0]
    ext = ALLOWED_TYPES.get(ctype)
    if not ext:
        raise HTTPException(400, f"unsupported content-type: {ctype}")

    # 3) pick safe filename and final path
    orig = (file.filename or f"upload{ext}").split("/")[-1]
    orig = SAFE_NAME.sub("_", orig)
    doc_id = str(uuid.uuid4())
    app_dir = Path(STORAGE_ROOT) / app_id
    app_dir.mkdir(parents=True, exist_ok=True)
    dest = app_dir / f"{doc_id}{ext}"

    # 4) write file atomically
    tmp = dest.with_suffix(dest.suffix + ".part")
    with tmp.open("wb") as f:
        while chunk := file.file.read(1024 * 1024):
            f.write(chunk)
    tmp.replace(dest)  # atomic move

    size = dest.stat().st_size

    # 5) register in DB
    try:
        cur = conn.cursor()
        cur.execute(
            """
            insert into documents (id, application_id, doc_type, filename, storage_path, content_type, size_bytes)
            values (%s,%s,%s,%s,%s,%s,%s)
            """,
            (doc_id, app_id, doc_type, orig, str(dest), ctype, size),
        )
        cur.execute(
            "insert into audit_log(application_id, actor, action, payload) values (%s,%s,%s,%s)",
            (
                app_id,
                "user",
                "uploaded_document",
                Json({"document_id": doc_id, "doc_type": doc_type}),
            ),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"db error: {e}") from e

    return {"document_id": doc_id, "path": str(dest), "size": size, "content_type": ctype}


@app.post("/documents/{doc_id}/parse")
def parse_document(doc_id: str):
    # 1) fetch doc metadata from Postgres
    try:
        conn = _conn()
        cur = conn.cursor()
        cur.execute(
            "select application_id, doc_type, storage_path from documents where id=%s",
            (doc_id,),
        )
        row = cur.fetchone()
        if not row:
            cur.close()
            conn.close()
            raise HTTPException(404, "document not found")
        application_id, doc_type, storage_path = row
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"db error: {e}") from e

    path = Path(storage_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        pages = extract_pages_text(path)
        chunks = chunk_pages(pages)
        tables = []
    elif suffix in {".jpg", ".jpeg", ".png"}:
        pages = ocr_image(path)
        chunks = chunk_pages(pages)
        tables = []
    elif suffix in {".xlsx"}:
        # spreadsheets: no page chunks; we store tables
        pages = []
        chunks = []
        tables = parse_xlsx_tables(path)
    else:
        raise HTTPException(400, "only PDF/JPG/PNG/XLSX parsing implemented")

    # 3) write to Mongo
    col = get_mongo()
    doc = {
        "application_id": str(application_id),
        "document_id": doc_id,
        "doc_type": doc_type,
        "chunks": chunks,
        "tables": tables,
    }
    col.delete_many({"document_id": doc_id})  # idempotent rewrite
    col.insert_one(doc)

    # 4) audit
    try:
        cur = conn.cursor()
        cur.execute(
            "insert into audit_log(application_id, actor, action, payload) values (%s,%s,%s,%s::jsonb)",
            (str(application_id), "system", "parsed_document", json.dumps({"document_id": doc_id})),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception:
        # bandit B110: never swallow silently; record & continue
        logger.exception("audit_log insert failed for parsed_document")

    return {
        "document_id": doc_id,
        "pages": len(set([c["page"] for c in chunks])),
        "chunks": len(chunks),
    }


@app.get("/applications/{app_id}/documents")
def list_documents(app_id: str):
    try:
        conn = _conn()
        cur = conn.cursor()
        cur.execute(
            """
            select id, doc_type, filename, content_type, size_bytes
            from documents where application_id=%s order by created_at desc
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
        conn.close()
        return {"documents": items}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"db error: {e}") from e


@app.get("/applications/{app_id}/status")
def application_status(app_id: str):
    conn = _conn()
    app = get_application(conn, app_id)
    if not app:
        conn.close()
        raise HTTPException(404, "application not found")

    docs = list_docs_for_app(conn, app_id)
    conn.close()

    # what’s uploaded vs required
    required = {"eid_front", "eid_back", "bank_statement"}
    have = {d["doc_type"] for d in docs}
    missing = list(required - have)

    return {
        "application_id": app_id,
        "documents": docs,
        "missing_required": missing,
        "ready_for_evaluation": len(missing) == 0 and len(docs) > 0,
    }


@app.post("/applications/{app_id}/evaluate")
def application_evaluate(app_id: str):
    conn = _conn()
    app = get_application(conn, app_id)
    if not app:
        conn.close()
        raise HTTPException(404, "application not found")

    docs = list_docs_for_app(conn, app_id)
    parsed = get_parsed_for_docs([d["id"] for d in docs])
    result = evaluate_rules(app, docs, parsed)

    # audit (non-blocking)
    try:
        cur = conn.cursor()
        cur.execute(
            "insert into audit_log(application_id, actor, action, payload) values (%s,%s,%s,%s::jsonb)",
            (app_id, "system", "evaluated_application", json.dumps(result)),
        )
        conn.commit()
        cur.close()
    except Exception:
        logger.exception("audit_log insert failed for evaluated_application")
    finally:
        conn.close()

    return result


@app.post("/applications/{app_id}/parse_all")
def parse_all(app_id: str):
    conn = _conn()
    docs = list_docs_for_app(conn, app_id)
    conn.close()
    if not docs:
        raise HTTPException(404, "no documents for application")

    results = []
    for d in docs:
        try:
            res = parse_document(d["id"])
            results.append(
                {
                    "id": d["id"],
                    "ok": True,
                    "chunks": res.get("chunks", 0),
                    "pages": res.get("pages", 0),
                }
            )
        except HTTPException as he:
            results.append({"id": d["id"], "ok": False, "error": he.detail})
        except Exception as e:  # noqa: BLE001
            results.append({"id": d["id"], "ok": False, "error": str(e)})

    ok = sum(1 for r in results if r["ok"])
    return {"parsed_ok": ok, "total": len(results), "details": results}


def _collect_evidence(parsed_docs: list[dict], max_chars: int = 1200) -> str:
    # Prefer bank tables rows first; then texts
    lines: list[str] = []
    for p in parsed_docs:
        if p.get("tables"):
            # take first few rows of first table
            t = p["tables"][0]
            rows = t.get("rows", [])[:5]
            for row in rows:
                lines.append(" • " + ", ".join(f"{k}={v}" for k, v in list(row.items())[:6]))
    if not lines:
        # fallback to first few chunks
        for p in parsed_docs:
            chunks = p.get("chunks") or []
            if chunks:
                lines.append(chunks[0].get("text", "")[:400])
    text = "\n".join(lines)
    return text[:max_chars]


@app.post("/applications/{app_id}/justify")
def application_justify(app_id: str):
    conn = _conn()
    app = get_application(conn, app_id)
    if not app:
        conn.close()
        raise HTTPException(404, "application not found")

    docs = list_docs_for_app(conn, app_id)
    parsed = get_parsed_for_docs([d["id"] for d in docs])
    conn.close()

    # deterministic rules & score
    eval_res = evaluate_rules(app, docs, parsed)

    # craft a concise prompt
    failed = [r for r in eval_res["rules"] if not r["passed"]]
    passed = [r for r in eval_res["rules"] if r["passed"]]
    evidence = _collect_evidence(parsed)

    sys_prompt = (
        "You are a senior case reviewer. Explain the eligibility result succinctly for another reviewer. "
        "Be concrete, cite which documents were checked, and keep it under 120 words. "
        "If something is missing, clearly call it out. No fluff."
    )
    user_prompt = f"""
Application:
- name: {app.get('applicant_full_name')}
- household_size: {app.get('household_size')}
- monthly_income: {app.get('monthly_income')}

Rules (passed):
{chr(10).join(f"- {r['id']}: {r['reason']}" for r in passed)}

Rules (failed):
{chr(10).join(f"- {r['id']}: {r['reason']}" for r in failed)}

Evidence (snippets):
{evidence}

Explain the overall status "{eval_res['status']}" (score {eval_res['score']}). Output 2-3 short bullets.
"""

    prompt = f"<system>\n{sys_prompt}\n</system>\n<user>\n{user_prompt}\n</user>"
    try:
        text = llm_generate(prompt, model=DEFAULT_MODEL, temperature=0.2, timeout_s=60)
    except Exception as e:
        raise HTTPException(500, f"llm error: {e}") from e

    # trace (if Langfuse configured)
    trace_llm(
        event="application_justify",
        input_payload={
            "model": DEFAULT_MODEL,
            "status": eval_res["status"],
            "score": eval_res["score"],
        },
        output_text=text,
        tags=["justify", "eval"],
    )

    return {
        "application_id": app_id,
        "status": eval_res["status"],
        "score": eval_res["score"],
        "explanation": text,
    }


@app.post("/documents/{doc_id}/index")
def index_document(doc_id: str):
    # 1) fetch meta
    conn = _conn()
    cur = conn.cursor()
    cur.execute("select application_id, doc_type from documents where id=%s", (doc_id,))
    row = cur.fetchone()
    if not row:
        cur.close()
        conn.close()
        raise HTTPException(404, "document not found")
    application_id, doc_type = row

    # 2) load parsed artifact
    col = get_mongo()
    art = col.find_one({"document_id": doc_id})
    if not art:
        cur.close()
        conn.close()
        raise HTTPException(400, "document not parsed yet")

    # 3) build texts → vectors
    items = _flatten_artifacts_for_embed(art)
    texts = [i["text"] for i in items]
    vectors = embed(texts)
    for i, v in zip(items, vectors, strict=False):
        i["vector"] = v

    # 4) upsert
    n = vs_upsert(
        str(application_id),
        [
            {
                "document_id": doc_id,
                "doc_type": doc_type,
                "page": i["page"],
                "text": i["text"],
                "vector": i["vector"],
            }
            for i in items
        ],
    )

    cur.close()
    conn.close()
    return {"document_id": doc_id, "embedded": n}


@app.post("/applications/{app_id}/reindex")
def reindex_application(app_id: str):
    conn = _conn()
    docs = list_docs_for_app(conn, app_id)
    conn.close()
    if not docs:
        raise HTTPException(404, "no documents to index")
    total = 0
    for d in docs:
        try:
            res = index_document(d["id"])
            total += int(res.get("embedded", 0))
        except HTTPException:
            continue
    return {"application_id": app_id, "embedded": total}


@app.get("/applications/{app_id}/search")
def semantic_search(app_id: str, q: str, k: int = 6):
    vec = embed([q])[0]
    hits = vs_search(app_id, vec, top_k=int(k))
    return {"query": q, "results": hits}


@app.post("/applications/{app_id}/qa")
def qa(app_id: str, q: str):
    # retrieve
    vec = embed([q])[0]
    hits = vs_search(app_id, vec, top_k=6)

    # craft prompt with citations
    context = (
        "\n\n".join(
            [
                f"[{i+1}] ({h.get('doc_type')} p{h.get('page')}) {h.get('text')}"
                for i, h in enumerate(hits)
            ]
        )
        or "No context found."
    )
    sys_prompt = "Answer the user's question using only the CONTEXT. Be concise (<=120 words). Cite sources like [1], [2]. If unknown, say you don't know."
    user_prompt = f"QUESTION:\n{q}\n\nCONTEXT:\n{context}\n\nAnswer with citations."

    prompt = f"<system>\n{sys_prompt}\n</system>\n<user>\n{user_prompt}\n</user>"
    try:
        answer = llm_generate(prompt, temperature=0.1, timeout_s=60)
    except Exception as e:
        raise HTTPException(500, f"llm error: {e}") from e

    trace_llm("qa", {"q": q, "k": len(hits)}, answer, tags=["rag", "qa"])
    return {"answer": answer, "hits": hits}


@app.post("/auth/token")
def login(form: OAuth2PasswordRequestForm = Depends()):  # noqa: B008 (FastAPI)
    if not PWD_HASH:
        raise HTTPException(500, "server not configured: ADMIN_PWD_HASH missing")
    if form.username != USERNAME or not verify_password(form.password, PWD_HASH):
        raise HTTPException(401, "bad credentials")
    tok = create_access_token(sub=USERNAME, minutes=settings.ACCESS_TOKEN_EXPIRE_MIN)
    return {"access_token": tok, "token_type": "bearer"}


@app.post("/applications/{app_id}/agent/run")
def run_agents(app_id: str) -> dict[str, Any]:
    """
    Orchestrate: extract -> validate -> eligibility -> recommend (LangGraph).
    Returns the ReAct-style steps + final artifacts for transparency.
    """
    # quick existence check
    conn = _conn()
    app = get_application(conn, app_id)
    conn.close()
    if not app:
        raise HTTPException(404, "application not found")

    state = agent_runner({"application_id": app_id})
    return {
        "application_id": app_id,
        "status": state.get("status", "ok"),
        "steps": state.get("steps", []),
        "validations": state.get("validations", {}),
        "rules_result": state.get("rules_result", {}),
        "ml_result": state.get("ml_result", {}),
        "recommendation": state.get("recommendation", ""),
    }


@app.post("/ml/train")
def ml_train(payload: dict[str, Any] | None = None):
    """
    Minimal trainer: builds a dataset from existing applications in DB.
    For demo, uses a synthetic label based on income_per_capita + doc completeness.
    """
    conn = pg_conn()
    cur = conn.cursor()
    cur.execute("select id from applications order by created_at desc limit 500")
    app_ids = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()

    if not app_ids:
        raise HTTPException(400, "no applications to build training set")

    try:
        stats = train_from_applications(app_ids)
    except Exception as e:
        raise HTTPException(500, f"ml train error: {e}") from e
    return stats


@app.get("/applications/{app_id}/ml_score")
def ml_score(app_id: str):
    """
    Predict an eligibility probability and expose linear contributions.
    """
    conn = pg_conn()
    app = get_application(conn, app_id)
    conn.close()
    if not app:
        raise HTTPException(404, "application not found")

    try:
        res = predict_with_explanations(app_id)
    except FileNotFoundError:
        raise HTTPException(400, "model not trained; call POST /ml/train first")
    except Exception as e:
        raise HTTPException(500, f"ml predict error: {e}") from e

    return {
        "application_id": app_id,
        **res,
    }
