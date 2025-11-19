from typing import List


def embed_job_descriptions(texts: List[str]) -> List[List[float]]:
    """
    Placeholder for job/upskilling text embeddings.
    Later: use sentence-transformers or same model as RAG.
    """
    # TODO: real embeddings
    return [[0.0] * 10 for _ in texts]
