# Evaluation

The paper proposes four context-specific metrics for evaluating
GenDFIR-style outputs (Sec. V-B). Free-text DFIR reports cannot be
auto-graded with conventional NLP metrics — F1/precision/recall fail
to capture evidence-citation correctness. The paper relies on **human
review**, and so does this replication.

## What is automated vs. what is human

| Metric | This package automates? |
|--------|------------------------|
| **Accuracy** (Eq. 20) | Arithmetic only — humans must classify each fact as correct/incorrect first. |
| **Relevance** | Arithmetic only — humans judge on-topic vs off-topic for each retrieved event. |
| **Exact Match** | Yes — `gendfir.metrics.exact_match()`. Useful when a structured ground-truth string exists. |
| **Top-k Retrieval Recall** | Yes — `gendfir.metrics.top_k_recall()`. Requires a labelled set of "anomalous" event indices. |

## Workflow

```python
from gendfir import GenDFIRPipeline
from gendfir.embeddings import make_embedder
from gendfir.llm import make_llm
from gendfir.metrics import EvaluationReport, FactCounts, top_k_recall

pipe = GenDFIRPipeline(embedder=make_embedder(), llm=make_llm())
result = pipe.run(
    csv_path="examples/scenarios/unauthorised_access.csv",
    query="Conduct DFIR timeline analysis on the unauthorised access incident",
    topk=15,
)

# 1. Have a human (or a stronger LLM-as-a-judge) classify each fact
#    in result.report as correct vs incorrect, split by section:
fc = FactCounts(
    kb_correct=18, kb_incorrect=2,
    insight_correct=3, insight_incorrect=1,
)

# 2. Optional: compare retrieved indices against a labelled ground-truth
#    set of "true anomaly" rows.
ground_truth_indices = [0, 1, 2, 7, 9, 14]   # row indices in CSV
recall = top_k_recall(result.retrieved_indices, ground_truth_indices)

# 3. Roll into a report
rep = EvaluationReport(
    scenario="UnauthorisedAccess",
    fact_counts=fc,
    retrieved_relevant=14,
    retrieved_total=15,
    retrieval_recall=recall,
)
print(rep.render_text())
```

## Recommended split

The paper splits report contents into two sections, and we follow that:

- **Knowledge-Base Facts** — anything derived from retrieved evidence
  (timeline, correlations, IOCs that appear in the events).
- **Additional Insights** — anything from the LLM's general knowledge
  applied to the case (mitigation suggestions, ATT&CK mappings,
  policy recommendations).

Tracking these separately lets you tell whether errors are coming from
**retrieval failures** (KB section) or **LLM hallucination**
(Insights section). Different fixes for different failure modes.

## Paper-reported numbers (Sec. V)

For reference, the paper reports on six scenarios:

| Scenario | Overall Accuracy (paper) |
|----------|-------------------------:|
| SYN Flood | reported in Fig. 8 |
| Rhino Hunt | reported in Fig. 8 |
| Phishing Email-1 | reported in Fig. 8 |
| Phishing Email-2 | reported in Fig. 8 |
| DNS Spoof | reported in Fig. 8 |
| Unauthorised Access | reported in Fig. 8 (relevance ≈ 94.51%) |

Reproducing these numbers requires the original paper's exact
synthetic datasets, which are released under the authors' repository
at https://github.com/GenDFIR/GenDFIR. The CSV scenarios in `examples/` here
are written from scratch and cover similar topics but are NOT the
same data — comparison numbers will differ accordingly.
