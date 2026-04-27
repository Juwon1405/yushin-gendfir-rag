"""End-to-end pipeline test using the deterministic fallback backends."""
from __future__ import annotations

from pathlib import Path

import pytest

from gendfir.embeddings import DeterministicHashEmbedder
from gendfir.llm import TemplateFallbackLLM
from gendfir.pipeline import GenDFIRPipeline


SCENARIOS_DIR = Path(__file__).parent.parent / "examples" / "scenarios"


@pytest.mark.parametrize(
    "csv_name, query",
    [
        ("unauthorised_access.csv", "Reconstruct the unauthorised access incident timeline."),
        ("lateral_movement.csv", "Identify privilege escalation and lateral movement."),
        ("syn_flood.csv", "Determine when the SYN flood started and when service was restored."),
    ],
)
def test_pipeline_runs_end_to_end_offline(csv_name, query):
    """Pipeline runs without Ollama / sentence-transformers and returns a report."""
    csv_path = SCENARIOS_DIR / csv_name
    assert csv_path.is_file(), f"missing fixture: {csv_path}"

    pipe = GenDFIRPipeline(
        embedder=DeterministicHashEmbedder(dim=64),
        llm=TemplateFallbackLLM(),
        embed_token_cap=128,   # smaller than default for test speed
    )
    result = pipe.run(
        csv_path=str(csv_path),
        query=query,
        topk=5,
        max_tokens=500,
    )

    # Structural assertions only — semantic correctness needs a real LLM.
    assert isinstance(result.report, str) and result.report
    assert len(result.retrieved_indices) <= 5
    assert len(result.retrieved_indices) > 0
    assert result.attention_weights.shape == (len(result.retrieved_indices),)
    assert abs(float(result.attention_weights.sum()) - 1.0) < 1e-4
    assert query in result.user_prompt
    assert "DFIR AI assistant" in result.system_prompt


def test_pipeline_rejects_topk_zero(tmp_path):
    """topk = 0 must produce no retrieval."""
    # Build a tiny CSV
    p = tmp_path / "evt.csv"
    p.write_text("a\nfoo\nbar\n", encoding="utf-8")

    pipe = GenDFIRPipeline(
        embedder=DeterministicHashEmbedder(dim=32),
        llm=TemplateFallbackLLM(),
        embed_token_cap=32,
    )
    result = pipe.run(csv_path=str(p), query="anything", topk=0)
    assert result.retrieved_indices == []
    assert result.attention_weights.shape == (0,)
