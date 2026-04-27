"""
Evaluation metrics.

Reference (Loumachi et al., 2024, Sec. V-B):

    Eq. (20)  Accuracy = Overall_Correct_Facts /
                          Overall (Correct_Facts + Incorrect_Facts)

The paper reports four context-specific metrics:

    1. Accuracy           — Eq. (20), implemented here.
    2. Relevance          — proportion of generated facts that are
                            on-topic for the incident; in the paper
                            this is computed against ground-truth
                            knowledge-base events.
    3. Exact Match (EM)   — string-level match against ground truth;
                            implemented here as a helper.
    4. Top-k Evidence
       Retrieval          — does the retrieved set include the
                            ground-truth events? Implemented as
                            top_k_recall.

Because reports are free-text, the paper relies on human review to
classify each generated fact as Correct vs Incorrect. This module
provides helpers for *both* (a) human-in-the-loop scoring and
(b) automatic ground-truth comparison when structured ground truth
is available. Only the structural arithmetic is automated; semantic
correctness still needs human judgement, as the paper itself notes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Set


@dataclass
class FactCounts:
    """
    Per-section fact counts from a generated DFIR report.

    The paper splits a report into:
        a) Knowledge-Base Facts Section
           (artefact analysis, timeline, correlations — derived from
            retrieved evidence)
        b) Additional Insights Section
           (mitigation, recommendations — drawn from LLM general
            knowledge applied to the case)

    Each section is tallied separately so analysts can see whether
    errors come from retrieval or from LLM extrapolation.
    """

    kb_correct: int = 0
    kb_incorrect: int = 0
    insight_correct: int = 0
    insight_incorrect: int = 0

    @property
    def total_correct(self) -> int:
        return self.kb_correct + self.insight_correct

    @property
    def total_incorrect(self) -> int:
        return self.kb_incorrect + self.insight_incorrect

    @property
    def total(self) -> int:
        return self.total_correct + self.total_incorrect


def accuracy(facts: FactCounts) -> float:
    """
    Eq. (20) — overall accuracy.

    Returns 0.0 for an empty report (no facts at all). Caller should
    treat this as undefined rather than as a true zero-accuracy score.
    """
    if facts.total == 0:
        return 0.0
    return facts.total_correct / facts.total


def section_accuracy(correct: int, incorrect: int) -> float:
    """Accuracy for a single section (KB or Insights)."""
    total = correct + incorrect
    return 0.0 if total == 0 else correct / total


def relevance(retrieved_relevant: int, retrieved_total: int) -> float:
    """
    Relevance rate over the retrieved evidence set.

    The paper reports this per scenario (e.g., 94.51% on the
    Unauthorised-Access scenario). It is the proportion of retrieved
    events that are judged on-topic relative to the analyst's query.
    """
    if retrieved_total <= 0:
        return 0.0
    return retrieved_relevant / retrieved_total


def exact_match(prediction: str, reference: str, normalise: bool = True) -> bool:
    """
    Token-level Exact Match.

    `normalise=True` lowercases and collapses whitespace before
    comparing — practical for free-text DFIR strings where case and
    spacing are not semantically meaningful.
    """
    if normalise:
        a = " ".join(prediction.lower().split())
        b = " ".join(reference.lower().split())
        return a == b
    return prediction == reference


def top_k_recall(
    retrieved_indices: Iterable[int],
    ground_truth_indices: Iterable[int],
) -> float:
    """
    Top-k retrieval recall.

    Given the indices the retriever returned and a set of ground-truth
    event indices, what fraction of the ground truth was successfully
    retrieved? Useful when the synthetic dataset has labelled
    "anomalous" rows.
    """
    gt: Set[int] = set(ground_truth_indices)
    if not gt:
        return 0.0
    retrieved: Set[int] = set(retrieved_indices)
    hits = len(gt & retrieved)
    return hits / len(gt)


@dataclass
class EvaluationReport:
    """Aggregated evaluation summary for one scenario run."""

    scenario: str
    fact_counts: FactCounts = field(default_factory=FactCounts)
    retrieved_relevant: int = 0
    retrieved_total: int = 0
    retrieval_recall: float = 0.0

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "accuracy": accuracy(self.fact_counts),
            "kb_section_accuracy": section_accuracy(
                self.fact_counts.kb_correct, self.fact_counts.kb_incorrect
            ),
            "insight_section_accuracy": section_accuracy(
                self.fact_counts.insight_correct,
                self.fact_counts.insight_incorrect,
            ),
            "relevance": relevance(self.retrieved_relevant, self.retrieved_total),
            "retrieval_recall_at_k": self.retrieval_recall,
            "fact_counts": {
                "kb_correct": self.fact_counts.kb_correct,
                "kb_incorrect": self.fact_counts.kb_incorrect,
                "insight_correct": self.fact_counts.insight_correct,
                "insight_incorrect": self.fact_counts.insight_incorrect,
                "total": self.fact_counts.total,
            },
        }

    def render_text(self) -> str:
        d = self.to_dict()
        return (
            f"Scenario: {d['scenario']}\n"
            f"  Overall Accuracy        : {d['accuracy']:.4f}\n"
            f"  KB-Section Accuracy     : {d['kb_section_accuracy']:.4f}\n"
            f"  Insight-Section Accuracy: {d['insight_section_accuracy']:.4f}\n"
            f"  Relevance Rate          : {d['relevance']:.4f}\n"
            f"  Top-k Retrieval Recall  : {d['retrieval_recall_at_k']:.4f}\n"
            f"  Fact Counts             : {d['fact_counts']}\n"
        )
