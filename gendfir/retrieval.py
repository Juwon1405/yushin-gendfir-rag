"""
Retrieval and attention.

Reference equations (Loumachi et al., 2024, Sec. III-C):

    Eq. (14)  R = TopK(V_E · v_Q^T,  k)
              — Top-k events by dot product between query and event embeddings.

    Eq. (15)  α_i = exp( v_Q^T · v_RE_i / sqrt(d) )
                  / sum_j exp( v_Q^T · v_RE_j / sqrt(d) )
              — Scaled-dot-product attention over the retrieved set.
              This is the same softmax-attention formulation as Vaswani
              et al. (2017), applied here purely to the post-retrieval
              top-k set so that the final context vector reflects the
              relative relevance of each retrieved event.

    Eq. (16)  c = sum_{i=1..k} α_i · v_RE_i
              — Attention-weighted context vector handed to the LLM.

This module exposes pure functions on numpy arrays, deliberately
decoupled from any embedding backend.
"""
from __future__ import annotations

import math
from typing import List

import numpy as np


def _l2_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x) + 1e-12)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for two 1-D vectors. Used in tests/diagnostics."""
    return float(np.dot(a, b) / (_l2_norm(a) * _l2_norm(b)))


def topk_indices(E: np.ndarray, vq: np.ndarray, k: int) -> List[int]:
    """
    Eq. (14) — Top-k retrieval by cosine similarity.

    Although the paper writes the retrieval as a dot product
    (V_E · v_Q^T), we use cosine similarity to neutralise scale
    differences across embedding norms — a standard practical choice
    that matches what most embedding-backed RAG systems do in
    deployment.

    Parameters
    ----------
    E : np.ndarray, shape (N, D)
        Event embedding matrix.
    vq : np.ndarray, shape (D,)
        Query embedding.
    k : int
        Number of events to retrieve. Capped at N.

    Returns
    -------
    List[int]
        Indices of the top-k most similar events, descending order.
    """
    if E.ndim != 2:
        raise ValueError(f"E must be 2-D, got shape {E.shape}")
    if vq.ndim != 1:
        raise ValueError(f"vq must be 1-D, got shape {vq.shape}")
    if E.shape[1] != vq.shape[0]:
        raise ValueError(
            f"Dimension mismatch: E has D={E.shape[1]}, vq has D={vq.shape[0]}"
        )

    n = E.shape[0]
    if n == 0 or k <= 0:
        return []
    k = min(k, n)

    norms = np.linalg.norm(E, axis=1) * _l2_norm(vq) + 1e-12
    sims = (E @ vq) / norms
    order = np.argsort(-sims)[:k]
    return order.tolist()


def attention_weights(V: np.ndarray, vq: np.ndarray) -> np.ndarray:
    """
    Eq. (15) — Scaled-dot-product softmax attention.

    Computes α_i over the retrieved set V (k × D) given the query vq.

    Parameters
    ----------
    V : np.ndarray, shape (k, D)
        Retrieved event embeddings (top-k subset of V_E).
    vq : np.ndarray, shape (D,)
        Query embedding.

    Returns
    -------
    np.ndarray, shape (k,)
        Attention weights, summing to 1.
    """
    if V.ndim != 2:
        raise ValueError(f"V must be 2-D, got shape {V.shape}")
    if V.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)

    d = V.shape[1]
    logits = (V @ vq) / math.sqrt(d)
    # Numerical-stable softmax
    logits = logits - np.max(logits)
    exps = np.exp(logits)
    return (exps / (np.sum(exps) + 1e-12)).astype(np.float32)


def weighted_context(V: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Eq. (16) — Attention-weighted context vector.

    c = sum_i α_i · v_RE_i

    Returned as a 1-D array of length D. Used downstream as a compact
    representation of the retrieved evidence. (Note: the LLM in this
    pipeline still receives the raw evidence snippets for provenance,
    not just the weighted vector.)
    """
    if V.shape[0] != weights.shape[0]:
        raise ValueError(
            f"Length mismatch: V has {V.shape[0]} rows, "
            f"weights has {weights.shape[0]}"
        )
    return weights @ V
