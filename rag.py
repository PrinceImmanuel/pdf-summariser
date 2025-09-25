"""
rag.py

Retrieval helpers: cosine similarity and top-k selection.

Functions:
- cosine_sim(a: np.ndarray, b: np.ndarray) -> float
- find_top_k_indices(query_vec: Sequence[float], doc_vecs: List[Sequence[float]], top_k: int) -> List[int]
"""

from typing import Sequence, List
import numpy as np


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity with guard for zero vectors."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return -1.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def find_top_k_indices(query_vec: Sequence[float], doc_vecs: List[Sequence[float]], top_k: int) -> List[int]:
    """
    Compute cosine similarity between query_vec and each vector in doc_vecs,
    return the indices of the top_k most similar (highest cosine).
    """
    q = np.array(query_vec, dtype=float)
    sims = []
    for i, v in enumerate(doc_vecs):
        v_arr = np.array(v, dtype=float)
        sims.append((cosine_sim(q, v_arr), i))
    sims.sort(key=lambda x: x[0], reverse=True)
    return [idx for (_, idx) in sims[:top_k]]
