"""Tests for gendfir.retrieval — Eq. (14)–(16)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from gendfir.retrieval import (
    attention_weights,
    cosine_similarity,
    topk_indices,
    weighted_context,
)


def test_cosine_similarity_basic():
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(1.0)
    c = np.array([0.0, 1.0])
    assert cosine_similarity(a, c) == pytest.approx(0.0)
    d = np.array([-1.0, 0.0])
    assert cosine_similarity(a, d) == pytest.approx(-1.0)


def test_topk_indices_orders_by_similarity():
    """Eq. (14): TopK retrieval returns indices in descending similarity."""
    E = np.array(
        [
            [1.0, 0.0],   # cos(q,e0) = 1
            [0.0, 1.0],   # cos(q,e1) = 0
            [0.7, 0.7],   # cos(q,e2) ≈ 0.707
        ]
    )
    vq = np.array([1.0, 0.0])
    idxs = topk_indices(E, vq, k=2)
    assert idxs == [0, 2]


def test_topk_caps_at_n():
    E = np.eye(3)
    vq = np.array([1.0, 0.0, 0.0])
    assert topk_indices(E, vq, k=10) == [0, 1, 2] or len(topk_indices(E, vq, 10)) == 3


def test_topk_empty_inputs():
    """k=0 or empty E should both return []."""
    # Empty event matrix
    E = np.empty((0, 4))
    vq = np.zeros(4)
    assert topk_indices(E, vq, k=3) == []
    # k=0 with matching dims
    E2 = np.eye(3)
    vq2 = np.zeros(3)
    assert topk_indices(E2, vq2, k=0) == []


def test_topk_dim_mismatch_raises():
    E = np.eye(3)
    vq = np.zeros(2)
    with pytest.raises(ValueError):
        topk_indices(E, vq, k=2)


def test_attention_sums_to_one():
    """Eq. (15): softmax-attention weights are a probability distribution."""
    rng = np.random.default_rng(1405)
    V = rng.standard_normal((6, 8)).astype(np.float32)
    vq = rng.standard_normal(8).astype(np.float32)
    w = attention_weights(V, vq)
    assert w.shape == (6,)
    assert math.isclose(float(np.sum(w)), 1.0, abs_tol=1e-5)
    assert (w >= 0).all()


def test_attention_higher_for_more_similar():
    """Eq. (15): more similar event → higher attention weight."""
    V = np.array(
        [
            [1.0, 0.0],   # very similar to query
            [0.0, 1.0],   # orthogonal to query
        ]
    )
    vq = np.array([1.0, 0.0])
    w = attention_weights(V, vq)
    assert w[0] > w[1]


def test_attention_uniform_for_orthogonal_set():
    """If all events have identical dot products with q, weights are uniform."""
    V = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    vq = np.array([1.0, 0.0])
    w = attention_weights(V, vq)
    assert w == pytest.approx(np.array([1 / 3, 1 / 3, 1 / 3]), abs=1e-5)


def test_weighted_context_shape():
    """Eq. (16): c = sum α_i * v_RE_i has shape (D,)."""
    V = np.array([[1.0, 0.0], [0.0, 1.0]])
    w = np.array([0.7, 0.3])
    c = weighted_context(V, w)
    assert c.shape == (2,)
    assert c == pytest.approx(np.array([0.7, 0.3]))


def test_weighted_context_length_mismatch():
    V = np.eye(3)
    w = np.array([0.5, 0.5])
    with pytest.raises(ValueError):
        weighted_context(V, w)
