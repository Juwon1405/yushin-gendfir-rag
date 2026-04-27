"""
GenDFIR-RAG — A study replication of Loumachi, Ghanem & Ferrag (2024).

This package is an unofficial, educational replication of the GenDFIR paper:

    Loumachi, F. Y., Ghanem, M. C., & Ferrag, M. A. (2024).
    GenDFIR: Advancing Cyber Incident Timeline Analysis Through
    Retrieval-Augmented Generation and Large Language Models.
    arXiv preprint.

The official authors' repository is at https://github.com/GenDFIR/GenDFIR.
This package is NOT affiliated with the original authors. It exists
purely for learning, reproducibility study, and extension experiments.

Modules:
    chunking    — Event-length-aware chunking (paper Eq. 1–4)
    embeddings  — Embedding backends (Ollama / sentence-transformers)
    llm         — LLM backends (Ollama / fallback template)
    retrieval   — Top-k retrieval + scaled-dot-product attention (Eq. 14–16)
    pipeline    — End-to-end orchestration
    prompts     — DFIR agent system prompt (verbatim from paper Listing 2)
    metrics     — Accuracy / Relevance evaluation (Eq. 20)
    cli         — Command-line interface
"""

__version__ = "0.2.0"
__author__ = "YuShin (優心 / Bang Juwon)"

from gendfir.pipeline import GenDFIRPipeline
from gendfir.embeddings import Embedder, OllamaEmbedder, SentenceTransformerEmbedder, make_embedder
from gendfir.llm import LLM, OllamaLLM, TemplateFallbackLLM, make_llm
from gendfir.retrieval import topk_indices, attention_weights, weighted_context
from gendfir.chunking import csv_to_event_docs, EventDoc, approx_tokens

__all__ = [
    "GenDFIRPipeline",
    "Embedder",
    "OllamaEmbedder",
    "SentenceTransformerEmbedder",
    "make_embedder",
    "LLM",
    "OllamaLLM",
    "TemplateFallbackLLM",
    "make_llm",
    "topk_indices",
    "attention_weights",
    "weighted_context",
    "csv_to_event_docs",
    "EventDoc",
    "approx_tokens",
]
