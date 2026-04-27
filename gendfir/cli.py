"""
Command-line interface for GenDFIR-RAG.

Examples
--------
# Smoke run (uses fallback LLM if Ollama unavailable, prints report):
python -m gendfir.cli \\
    --csv examples/scenarios/unauthorised_access.csv \\
    --query "Conduct DFIR timeline analysis on the unauthorised access incident"

# Faithful replication (paper baseline — needs Ollama with llama3.1:8b):
python -m gendfir.cli \\
    --csv examples/scenarios/unauthorised_access.csv \\
    --query "Conduct DFIR timeline analysis ..." \\
    --embed-model mxbai-embed-large \\
    --llm-model llama3.1:8b \\
    --topk 15 \\
    --max-tokens 2000 \\
    --temperature 0.1

# Force Sentence-Transformers + template fallback (no Ollama needed):
python -m gendfir.cli --csv ... --query ... --no-ollama
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gendfir import __version__
from gendfir.embeddings import make_embedder
from gendfir.llm import make_llm
from gendfir.pipeline import GenDFIRPipeline


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gendfir",
        description=(
            "GenDFIR-RAG — unofficial study replication of "
            "Loumachi, Ghanem & Ferrag (2024). "
            "RAG + LLM pipeline for cyber-incident timeline analysis."
        ),
    )
    p.add_argument(
        "--csv",
        required=True,
        help="Path to the incident-events CSV (any column schema).",
    )
    p.add_argument(
        "--query",
        required=True,
        help="DFIR analyst query, e.g., "
        "'Conduct DFIR timeline analysis on the unauthorised access incident'.",
    )
    p.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Top-k events to retrieve (default: 10).",
    )
    p.add_argument(
        "--no-ollama",
        action="store_true",
        help="Disable Ollama and use Sentence-Transformers + template fallback.",
    )
    p.add_argument(
        "--embed-model",
        default="mxbai-embed-large",
        help="Ollama embedding model tag (default: mxbai-embed-large).",
    )
    p.add_argument(
        "--llm-model",
        default="llama3.1:8b",
        help="Ollama LLM model tag (default: llama3.1:8b — paper baseline).",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=2000,
        help="LLM max output tokens (default: 2000 — paper setting).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="LLM sampling temperature (default: 0.1 — low for factual output).",
    )
    p.add_argument(
        "--embed-token-cap",
        type=int,
        default=512,
        help="Embedding-model max tokens per chunk TM (default: 512).",
    )
    p.add_argument(
        "--chars-per-token",
        type=int,
        default=4,
        help="C_avg — average characters per token (default: 4, paper).",
    )
    p.add_argument(
        "--snippet-char-limit",
        type=int,
        default=240,
        help="Per-snippet char cap inside the LLM prompt (default: 240).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human-readable text.",
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"gendfir-rag {__version__}",
    )
    return p


def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)

    if not Path(args.csv).is_file():
        print(f"[error] CSV not found: {args.csv}", file=sys.stderr)
        return 2

    embedder = make_embedder(
        prefer_ollama=not args.no_ollama,
        ollama_model=args.embed_model,
    )
    llm = make_llm(
        prefer_ollama=not args.no_ollama,
        model=args.llm_model,
    )

    pipe = GenDFIRPipeline(
        embedder=embedder,
        llm=llm,
        embed_token_cap=args.embed_token_cap,
        chars_per_token=args.chars_per_token,
    )
    result = pipe.run(
        csv_path=args.csv,
        query=args.query,
        topk=args.topk,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        snippet_char_limit=args.snippet_char_limit,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "report": result.report,
                    "retrieved_indices": result.retrieved_indices,
                    "attention_weights": [float(x) for x in result.attention_weights],
                    "embedder": type(embedder).__name__,
                    "llm": type(llm).__name__,
                    "topk": args.topk,
                    "embed_token_cap": args.embed_token_cap,
                    "chars_per_token": args.chars_per_token,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print("=" * 70)
        print(f"GenDFIR-RAG  v{__version__}")
        print(f"Embedder : {type(embedder).__name__}")
        print(f"LLM      : {type(llm).__name__}")
        print(f"Top-k    : {args.topk}")
        print("=" * 70)
        print(result.report)
        print("=" * 70)
        print(
            "Retrieved event indices (descending similarity): "
            + ", ".join(str(i) for i in result.retrieved_indices)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
