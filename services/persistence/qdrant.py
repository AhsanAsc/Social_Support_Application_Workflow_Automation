from qdrant_client import QdrantClient
from functools import lru_cache
from core.config import settings


@lru_cache
def get_qdrant_client() -> QdrantClient:
    """
    Creates a singleton Qdrant client.
    """
    return QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        timeout=60,
    )
