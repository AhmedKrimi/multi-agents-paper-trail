from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _get_embedder():
    from sentence_transformers import SentenceTransformer
    # Create embedder for name matching
    embedder = SentenceTransformer("BAAI/bge-m3")
    return embedder


def embed(text: list[str] | str) -> np.ndarray:
    """Generate normalized embeddings for one or more text inputs"""
    return _get_embedder().encode(text, normalize_embeddings=True)