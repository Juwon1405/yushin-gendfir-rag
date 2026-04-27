# Architecture

This document maps the paper's framework (Loumachi et al., 2024) onto
this codebase, equation by equation, so anyone reading the source can
verify the replication is faithful.

## Pipeline overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   Incident CSV  ──▶  ① Pre-process & Chunk  ──▶   List[EventDoc]    │
│                          (Eq. 1–4)                                  │
│                                                                     │
│                                       ▼                             │
│                              ② Embed events + query                 │
│                                       ▼                             │
│                              Event matrix V_E (N × D)               │
│                              Query vector v_Q (D,)                  │
│                                                                     │
│                                       ▼                             │
│                              ③ Top-k Retrieval                      │
│                                       (Eq. 14)                      │
│                              ▼                                      │
│                                       ▼                             │
│                              ④ Attention + Context vector           │
│                                       (Eq. 15–16)                   │
│                                       ▼                             │
│                              ⑤ LLM (Llama 3.1 8B)                   │
│                                       ▼                             │
│                              DFIR Timeline Report                   │
└─────────────────────────────────────────────────────────────────────┘
```

## Stage-by-stage

### ① Pre-process & Chunk — `gendfir/chunking.py`

| Paper | Code |
|-------|------|
| Eq. (1)  T(E_i) = Σ T(e_ij) | `event_char_length()` |
| Eq. (2)  T_tokens(E_i) ≈ T(E_i) / C_avg | `approx_tokens()` (default `chars_per_token=4`) |
| Eq. (3)  T_tokens(C_k) ≈ T(C_k) / C_avg | same function applied to a chunk |
| Eq. (4)  T_tokens(C_k) ≤ TM | enforced inside `csv_to_event_docs()` via padding/truncation |

One CSV row → one event sentence → one chunk → one embedding. The paper
calls this "events chunking" and emphasises that 1 event = 1 chunk for
retrieval precision.

### ② Embedding — `gendfir/embeddings.py`

The paper uses `mxbai-embed-large` (Mixedbread AI) via Ollama. We
preserve that as the default and add `sentence-transformers` (CPU) and
a deterministic hash fallback for tests.

### ③ Top-k Retrieval — `gendfir/retrieval.py::topk_indices`

| Paper | Code |
|-------|------|
| Eq. (14)  R = TopK( V_E · v_Q^T,  k ) | `topk_indices(E, vq, k)` |

We use **cosine similarity** rather than the paper's bare dot product.
This is a common practical choice that controls for embedding-norm
variance and matches the behaviour of FAISS / Chroma defaults. For
strict paper-faithful runs, swap in a custom dot-product variant —
the difference is small for normalised embedding outputs.

### ④ Attention + Context vector — `gendfir/retrieval.py`

| Paper | Code |
|-------|------|
| Eq. (15)  α_i = softmax( v_Q · v_RE_i / √d ) | `attention_weights(V, vq)` |
| Eq. (16)  c = Σ α_i · v_RE_i | `weighted_context(V, weights)` |

This is **scaled-dot-product attention** (Vaswani et al., 2017) applied
not at training time but at retrieval time, over the top-k set. Its
practical effect is to rank retrieved evidence by intra-set relevance
before the LLM sees it. We compute the context vector `c` for
diagnostic / extension use, and pass the raw retrieved snippets to the
LLM (preserving provenance for evidence citation).

### ⑤ LLM Generation — `gendfir/llm.py` + `gendfir/prompts.py`

Paper baseline: Llama 3.1 8B, zero-shot, max_tokens=2000, temperature
≈ 0.1. We preserve this and provide a deterministic template fallback
for environments without Ollama.

The system prompt in `gendfir.prompts.DFIR_AGENT_SYSTEM_PROMPT` is
**reproduced verbatim** from the paper's Listing 2. Do not modify it
when running replication experiments — fork a new constant instead.

## Data structures

```python
@dataclass
class EventDoc:
    text: str       # Padded/truncated sentence (length = TM × C_avg)
    length: int     # Character count after padding
    tokens: int     # Approximate token count (Eq. 2)
    raw: str        # Pre-padding form, used for evidence snippets
```

`PipelineResult.attention_weights` is a 1-D numpy array of length k
(sums to 1). `PipelineResult.retrieved_indices` are indices into the
original event list (post-chunking, pre-retrieval).

## Where this implementation deviates from the paper

| Item | Paper | Here | Why |
|------|------:|------|-----|
| Retrieval scoring | Bare dot product | Cosine similarity | Norm-invariance; standard practice |
| Embedding default | `mxbai-embed-large` (Ollama) | Same — unchanged | Faithful |
| LLM default | Llama 3.1 8B (Ollama) | Same — unchanged | Faithful |
| LLM fallback | Not provided | `TemplateFallbackLLM` | Lets the package be importable / tested without Ollama |
| Embedding fallback | Not provided | `SentenceTransformer` and hash fallback | Same reason |
| Evaluation | Manual human review | Same; we provide `metrics` helpers but DO NOT auto-judge correctness | The paper itself notes human review is essential |

## Where this implementation could grow

- **Reranker stage** between Top-k and attention (cross-encoder).
- **Per-attribute embedding** instead of flat sentence concatenation, to test whether type-aware retrieval improves recall.
- **MITRE ATT&CK awareness** — annotate retrieved events with technique IDs before the LLM sees them.
- **Long-context LLMs** — the paper uses Llama 3.1 8B (128k ctx); allow larger top-k without truncation.
- **Multi-CSV ingestion** — currently one CSV per run; combining multiple sources (firewall + endpoint + DC logs) is a natural extension.
