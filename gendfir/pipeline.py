"""
End-to-end GenDFIR pipeline.

Four stages, matching the paper:
    1. Pre-process & chunk    (Sec. III-B  →  gendfir.chunking)
    2. Embed events + query                →  gendfir.embeddings
    3. Top-k retrieve + attention (Sec. III-C → gendfir.retrieval)
    4. Generate timeline analysis           →  gendfir.llm + gendfir.prompts
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from gendfir.chunking import EventDoc, csv_to_event_docs
from gendfir.embeddings import Embedder
from gendfir.llm import LLM
from gendfir.prompts import DFIR_AGENT_SYSTEM_PROMPT, build_user_prompt
from gendfir.retrieval import attention_weights, topk_indices, weighted_context


@dataclass
class PipelineResult:
    """Output of a single GenDFIR pipeline run."""

    report: str                       # LLM-generated timeline analysis
    retrieved_indices: List[int]      # Top-k indices into the original event list
    attention_weights: np.ndarray     # Per-retrieved-event attention scores (sums to 1)
    selected_events: List[EventDoc]   # Retrieved EventDocs in retrieval order
    user_prompt: str                  # Prompt that was fed to the LLM
    system_prompt: str                # System prompt that was fed to the LLM


class GenDFIRPipeline:
    """
    Replication of the GenDFIR pipeline (Loumachi et al., 2024).

    Parameters
    ----------
    embedder : Embedder
        Embedding backend. Use `gendfir.embeddings.make_embedder()` for
        the default (Ollama → sentence-transformers → hash fallback).
    llm : LLM
        LLM backend. Use `gendfir.llm.make_llm()` for the default
        (Ollama → template fallback).
    embed_token_cap : int
        Max tokens TM per chunk (paper: 512).
    chars_per_token : int
        C_avg (paper: 4 chars/token in English).
    system_prompt : Optional[str]
        Override the paper's DFIR system prompt. Defaults to the verbatim
        Listing-2 prompt from the paper.
    """

    def __init__(
        self,
        embedder: Embedder,
        llm: LLM,
        embed_token_cap: int = 512,
        chars_per_token: int = 4,
        system_prompt: Optional[str] = None,
    ):
        self.embedder = embedder
        self.llm = llm
        self.embed_token_cap = embed_token_cap
        self.chars_per_token = chars_per_token
        self.system_prompt = system_prompt or DFIR_AGENT_SYSTEM_PROMPT

    # ─── Stage helpers ──────────────────────────────────────────────

    def load_events(self, csv_path: str) -> List[EventDoc]:
        return csv_to_event_docs(
            csv_path,
            embed_token_cap=self.embed_token_cap,
            chars_per_token=self.chars_per_token,
        )

    def embed_events(self, events: List[EventDoc]) -> np.ndarray:
        return self.embedder.embed([e.text for e in events])

    def embed_query(self, query: str) -> np.ndarray:
        return self.embedder.embed([query])[0]

    # ─── End-to-end ────────────────────────────────────────────────

    def run(
        self,
        csv_path: str,
        query: str,
        topk: int = 10,
        max_tokens: int = 2000,
        temperature: float = 0.1,
        snippet_char_limit: int = 240,
    ) -> PipelineResult:
        """
        Execute the full pipeline.

        Parameters
        ----------
        csv_path : str
            Path to the incident-events CSV.
        query : str
            DFIR analyst query (e.g., "Conduct DFIR timeline analysis on
            the unauthorised-access incident").
        topk : int
            Top-k retrieval depth.
        max_tokens : int
            LLM max_tokens budget.
        temperature : float
            LLM sampling temperature.
        snippet_char_limit : int
            Per-snippet char cap inside the user prompt.
        """
        # 1) Pre-process & chunk
        events = self.load_events(csv_path)

        # 2) Embed events + query
        E = self.embed_events(events)        # (N, D)
        vq = self.embed_query(query)         # (D,)

        # 3) Top-k retrieve  +  scaled-dot-product attention
        k = min(topk, len(events))
        idxs = topk_indices(E, vq, k)
        V = E[idxs, :]                       # (k, D)
        attn = attention_weights(V, vq)      # (k,)
        # Eq. (16) — context vector retained for diagnostic / extension use
        _ = weighted_context(V, attn)

        selected = [events[i] for i in idxs]

        # 4) Generate timeline analysis
        user_prompt = build_user_prompt(
            query=query,
            selected_event_snippets=[e.raw for e in selected],
            snippet_char_limit=snippet_char_limit,
        )
        report = self.llm.generate(
            self.system_prompt,
            user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return PipelineResult(
            report=report,
            retrieved_indices=idxs,
            attention_weights=attn,
            selected_events=selected,
            user_prompt=user_prompt,
            system_prompt=self.system_prompt,
        )
