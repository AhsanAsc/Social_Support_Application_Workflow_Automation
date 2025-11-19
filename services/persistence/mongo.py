from __future__ import annotations

import os
from typing import Any

from pymongo import ASCENDING, MongoClient

MONGO_URL = os.getenv("MONGO_URL", "mongodb://mongo:27017")
MONGO_DB = os.getenv("MONGO_DB", "ssai")


def get_mongo():
    client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=2000)
    db = client[MONGO_DB]
    col = db["parsed_artifacts"]
    col.create_index([("application_id", ASCENDING)])
    col.create_index([("document_id", ASCENDING)])
    col.create_index([("doc_type", ASCENDING), ("application_id", ASCENDING)])
    return col


def get_parsed_for_docs(doc_ids: list[str]) -> list[dict[str, Any]]:
    col = get_mongo()
    return list(col.find({"document_id": {"$in": doc_ids}}, {"chunks": 0}))
