from __future__ import annotations

import os
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "ssai_chunks")
VECTOR_SIZE = int(os.getenv("EMBED_DIM", "384"))

_client: QdrantClient | None = None


def client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL, timeout=10.0)
    return _client


def ensure_collection() -> None:
    c = client()
    if COLLECTION not in [col.name for col in c.get_collections().collections]:
        c.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def upsert_chunks(application_id: str, items: list[dict[str, Any]]) -> int:
    """
    items: [{"document_id":..., "doc_type":..., "page": int|None, "text": "...", "vector": [...]}, ...]
    """
    if not items:
        return 0
    ensure_collection()
    points = []
    for it in items:
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=it["vector"],
                payload={
                    "application_id": application_id,
                    "document_id": it["document_id"],
                    "doc_type": it.get("doc_type"),
                    "page": it.get("page"),
                    "text": it["text"],
                },
            )
        )
    client().upsert(collection_name=COLLECTION, points=points)
    return len(points)


def search(application_id: str, query_vec: list[float], top_k: int = 6) -> list[dict[str, Any]]:
    ensure_collection()
    flt = Filter(
        must=[FieldCondition(key="application_id", match=MatchValue(value=application_id))]
    )
    res = client().search(
        collection_name=COLLECTION,
        query_vector=query_vec,
        limit=top_k,
        query_filter=flt,
        with_payload=True,
    )
    out = []
    for r in res:
        p = r.payload or {}
        out.append(
            {
                "score": float(r.score),
                "document_id": p.get("document_id"),
                "doc_type": p.get("doc_type"),
                "page": p.get("page"),
                "text": p.get("text", "")[:600],
            }
        )
    return out
