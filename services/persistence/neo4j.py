from neo4j import GraphDatabase
from functools import lru_cache
from core.config import settings


@lru_cache
def get_neo4j_driver():
    return GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )


def upsert_applicant_graph(driver, app_id: int, extracted_data: dict):
    """
    Creates/updates applicant → documents → relationships graph.
    """
    with driver.session() as session:
        session.run(
            """
            MERGE (a:Applicant {app_id: $app_id})
            // TODO: attach employer, bank accounts, family, addresses, etc.
            """,
            app_id=app_id,
        )
