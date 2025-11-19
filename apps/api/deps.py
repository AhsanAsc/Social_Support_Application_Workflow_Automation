from fastapi import Depends, HTTPException, status
from core.security import get_current_user
from core.config import settings
from services.persistence.postgres import get_postgres_pool
from services.persistence.mongo import get_mongo_client
from services.persistence.qdrant import get_qdrant_client
from services.persistence.neo4j import get_neo4j_driver


def get_settings():
    """Provides application settings/config globally."""
    return settings


def get_current_active_user(user=Depends(get_current_user)):
    """
    Placeholder for user validation.
    Later: check DB, disabled accounts, roles, etc.
    """
    return user


def get_pg(=Depends(get_postgres_pool)):
    """Dependency for Postgres connection pool."""
    return get_postgres_pool()


def get_mongo():
    """Dependency for MongoDB client."""
    return get_mongo_client()


def get_qdrant():
    """Dependency for Qdrant."""
    return get_qdrant_client()


def get_neo4j():
    """Dependency for Neo4j driver."""
    return get_neo4j_driver()
