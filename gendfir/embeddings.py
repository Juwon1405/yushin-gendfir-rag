"""
Embedding backends.

The paper uses mxbai-embed-large via Ollama (Loumachi et al., 2024,
Sec. IV-B). This module preserves that as the primary backend and adds
a sentence-transformers fallback for environments without Ollama,
useful for CI runners and quick experiments.
"""
from __future__ import annotations

from typing import List

import numpy as np


class Embedder:
    """Abstract base class for embedding backends."""

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts into an (N, D) float32 array.

        Implementations must return one row per input text. Vectors do
        not need to be unit-normalised — downstream code uses cosine
        similarity, which normalises on the fly.
        """
        raise NotImplementedError


class OllamaEmbedder(Embedder):
    """
    Ollama-backed embedding.

    Default model is `mxbai-embed-large` (Mixedbread AI), matching the
    paper's setup. Other compatible models include `nomic-embed-text`
    and `bge-m3`.

    Prerequisites:
        - Ollama daemon running (default: http://127.0.0.1:11434)
        - Model pulled:  `ollama pull mxbai-embed-large`
    """

    def __init__(self, model: str = "mxbai-embed-large"):
        try:
            import ollama
        except ImportError as e:
            raise ImportError(
                "ollama package not installed — install via "
                "`pip install ollama`, or use the fallback "
                "SentenceTransformerEmbedder."
            ) from e
        self._client = ollama
        self.model = model

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = []
        for text in texts:
            response = self._client.embeddings(model=self.model, prompt=text)
            vectors.append(np.array(response["embedding"], dtype=np.float32))
        return np.vstack(vectors)


class SentenceTransformerEmbedder(Embedder):
    """
    Sentence-Transformers fallback (CPU-friendly, no Ollama needed).

    Default model: `sentence-transformers/all-MiniLM-L6-v2` —
    384-dim, ~80 MB, fast on CPU. For closer parity with the paper,
    use `mixedbread-ai/mxbai-embed-large-v1` (1024-dim).
    """

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed — install via "
                "`pip install sentence-transformers`."
            ) from e
        self.model_name = model
        self._model = SentenceTransformer(model)

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        embs = self._model.encode(
            texts, normalize_embeddings=False, convert_to_numpy=True
        )
        return embs.astype(np.float32)


class DeterministicHashEmbedder(Embedder):
    """
    Hash-based deterministic embedder for offline tests.

    NOT for production. Produces a fixed-dim vector per input string by
    hashing token bag-of-words into a sparse vector. Used by the test
    suite when neither Ollama nor sentence-transformers is available.
    """

    def __init__(self, dim: int = 64, seed: int = 1405):
        self.dim = dim
        self.seed = seed

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for token in str(t).lower().split():
                # Stable Python hash via builtin (seed-invariant via str ops)
                h = (hash(f"{self.seed}:{token}") & 0xFFFFFFFF) % self.dim
                out[i, h] += 1.0
        return out


def make_embedder(
    prefer_ollama: bool = True,
    ollama_model: str = "mxbai-embed-large",
    st_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    allow_fallback: bool = True,
) -> Embedder:
    """
    Factory: pick the best available embedder.

    Order of attempts:
        1. OllamaEmbedder      (if `prefer_ollama`)
        2. SentenceTransformerEmbedder
        3. DeterministicHashEmbedder (only if `allow_fallback`)

    Parameters
    ----------
    prefer_ollama : bool
        Try Ollama first. Disable in CI.
    ollama_model : str
        Ollama model tag.
    st_model : str
        sentence-transformers model identifier.
    allow_fallback : bool
        If both Ollama and sentence-transformers fail, fall back to a
        deterministic hash embedder. Useful for unit tests; not for
        meaningful retrieval.
    """
    if prefer_ollama:
        try:
            return OllamaEmbedder(ollama_model)
        except Exception:
            pass
    try:
        return SentenceTransformerEmbedder(st_model)
    except Exception:
        if allow_fallback:
            return DeterministicHashEmbedder()
        raise
