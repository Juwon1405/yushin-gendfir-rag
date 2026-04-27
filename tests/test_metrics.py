"""Tests for gendfir.metrics — Eq. (20)."""
from __future__ import annotations

import pytest

from gendfir.metrics import (
    EvaluationReport,
    FactCounts,
    accuracy,
    exact_match,
    relevance,
    section_accuracy,
    top_k_recall,
)


def test_accuracy_paper_equation():
    """Eq. (20): correct / (correct + incorrect)."""
    f = FactCounts(kb_correct=8, kb_incorrect=1, insight_correct=4, insight_incorrect=1)
    assert accuracy(f) == pytest.approx(12 / 14)


def test_accuracy_empty_report_is_zero():
    assert accuracy(FactCounts()) == 0.0


def test_section_accuracy():
    assert section_accuracy(3, 1) == 0.75
    assert section_accuracy(0, 0) == 0.0


def test_relevance_paper_value():
    """Paper reports 94.51% on Unauthorised-Access scenario."""
    # Just ensure the function arithmetic matches when given hand-picked counts.
    assert relevance(86, 91) == pytest.approx(86 / 91)


def test_exact_match_normalisation():
    assert exact_match("Logon  attempt failed.", "logon attempt FAILED.")
    assert not exact_match("Logon attempt failed", "Logon failed")


def test_exact_match_strict_mode():
    assert not exact_match("a", "A", normalise=False)
    assert exact_match("a", "a", normalise=False)


def test_top_k_recall():
    retrieved = [3, 5, 7, 9]
    ground_truth = [3, 7, 11]
    # 2 of 3 ground-truth indices are in the retrieved set
    assert top_k_recall(retrieved, ground_truth) == pytest.approx(2 / 3)


def test_top_k_recall_empty_gt():
    assert top_k_recall([1, 2, 3], []) == 0.0


def test_evaluation_report_to_dict_and_text():
    f = FactCounts(kb_correct=10, kb_incorrect=2, insight_correct=3, insight_incorrect=0)
    rep = EvaluationReport(
        scenario="UnauthorisedAccess",
        fact_counts=f,
        retrieved_relevant=14,
        retrieved_total=15,
        retrieval_recall=0.92,
    )
    d = rep.to_dict()
    assert d["scenario"] == "UnauthorisedAccess"
    assert d["accuracy"] == pytest.approx(13 / 15)
    assert "kb_correct" in d["fact_counts"]
    text = rep.render_text()
    assert "Overall Accuracy" in text
    assert "0.8" in text  # ratio renders
