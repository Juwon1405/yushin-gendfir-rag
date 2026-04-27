# yushin-gendfir-rag

> **A study replication of the GenDFIR paper** — a modular, testable Python re-implementation of the Retrieval-Augmented Generation pipeline for DFIR cyber-incident timeline analysis proposed by Loumachi, Ghanem & Ferrag (2024).

[![Python](https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-30%20passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-Loumachi%20et%20al.%202024-b31b1b)](#-citation)
[![Status](https://img.shields.io/badge/status-replication%20study-yellow)]()

> ⚠️ **This is an unofficial, educational re-implementation.** The original paper is by Loumachi, Ghanem & Ferrag (London Metropolitan University / University of Liverpool, 2024). The official authors' repository is at <https://github.com/GenDFIR>. This repo is **not affiliated with the original authors**; it exists for learning, reproducibility study, and extension experiments. Please cite the original paper — see [Citation](#-citation).

---

## 🎯 What this is

A clean Python package that walks through the GenDFIR pipeline stage-by-stage, with each paper equation mapped to a verifiable function and a passing unit test:

| Paper stage | This package |
|-------------|--------------|
| **§III-B** Pre-processing & length-aware chunking (Eq. 1–4) | `gendfir/chunking.py` |
| **§III-B/C** Embedding | `gendfir/embeddings.py` |
| **§III-C** Top-k retrieval (Eq. 14) | `gendfir/retrieval.py::topk_indices` |
| **§III-C** Scaled-dot-product attention (Eq. 15–16) | `gendfir/retrieval.py::attention_weights / weighted_context` |
| **§III-C** DFIR agent prompt (Listing 2) | `gendfir/prompts.py::DFIR_AGENT_SYSTEM_PROMPT` (verbatim) |
| **§III-C** Llama 3.1 8B generation | `gendfir/llm.py::OllamaLLM` |
| **§V-B** Accuracy (Eq. 20), Relevance, EM, Top-k recall | `gendfir/metrics.py` |

The package can be exercised offline (deterministic hash embedder + template LLM) or with the paper's actual stack (`mxbai-embed-large` + `llama3.1:8b` via Ollama). 30 unit tests verify the core math and end-to-end behaviour.

---

## 🚀 Quick start

### Offline smoke run (no Ollama, no LLM)

```bash
pip install numpy pandas pytest
git clone https://github.com/Juwon1405/yushin-gendfir-rag.git
cd yushin-gendfir-rag

python -m pytest                     # 30 tests pass

python -m gendfir.cli \
    --csv examples/scenarios/lateral_movement.csv \
    --query "Reconstruct the attacker timeline. Identify privilege escalation and lateral movement." \
    --no-ollama --topk 5
```

### Faithful paper baseline (Ollama + Llama 3.1 8B)

```bash
# 1. Install Ollama and pull the paper's models
ollama pull mxbai-embed-large       # embedder
ollama pull llama3.1:8b             # LLM (paper baseline)

# 2. Install package extras
pip install -e ".[ollama]"

# 3. Run
python -m gendfir.cli \
    --csv examples/scenarios/unauthorised_access.csv \
    --query "Conduct DFIR timeline analysis on the unauthorised access incident" \
    --embed-model mxbai-embed-large \
    --llm-model llama3.1:8b \
    --topk 15 --max-tokens 2000 --temperature 0.1
```

The paper's hardware setup (Sec. IV-B): NVIDIA GeForce RTX 4060, 8 GB VRAM, CUDA 12.6. Llama 3.1 8B in 4-bit quantisation fits comfortably.

---

## 🧮 The core equations, in code

The paper's Section III defines the mathematical framework. Every equation has a corresponding function in this package, accompanied by a unit test that asserts its expected behaviour.

```text
# Eq. (1)  Event character length
T(E_i) = Σⱼ T(e_ij)
└── gendfir.chunking.event_char_length()

# Eq. (2)/(3)  Token approximation
T_tokens ≈ T / C_avg          (paper sets C_avg = 4)
└── gendfir.chunking.approx_tokens()

# Eq. (4)  Chunk-token cap
T_tokens(C_k) ≤ TM            (paper TM = 512 for mxbai-embed-large)
└── enforced inside gendfir.chunking.csv_to_event_docs()

# Eq. (14)  Top-k retrieval
R = TopK(V_E · v_Q^T,  k)
└── gendfir.retrieval.topk_indices()

# Eq. (15)  Scaled-dot-product attention
α_i = exp( v_Q · v_RE_i / √d ) / Σⱼ exp( v_Q · v_RE_j / √d )
└── gendfir.retrieval.attention_weights()

# Eq. (16)  Attention-weighted context vector
c = Σ α_i · v_RE_i
└── gendfir.retrieval.weighted_context()

# Eq. (20)  Accuracy
Accuracy = Correct_Facts / (Correct_Facts + Incorrect_Facts)
└── gendfir.metrics.accuracy()
```

For the full mapping, see [`docs/architecture.md`](docs/architecture.md).

---

## 🧠 The DFIR agent prompt (verbatim from the paper)

```text
You are a DFIR AI assistant, tasked with analysing artefacts,
correlating events, and producing a coherent timeline of the
incident. Base your answer on the provided context and do not
include additional information outside of the context given.
```

Source: Loumachi et al. (2024), Listing 2, p. 10. Lives in `gendfir/prompts.py::DFIR_AGENT_SYSTEM_PROMPT`. **Do not modify** when running replication experiments — fork a new constant if you want to test prompt variations.

---

## 🧪 What was tested

```text
$ python -m pytest -v

tests/test_chunking.py ........... 7 passed   # Eq. 1–4
tests/test_metrics.py  ........... 9 passed   # Eq. 20 + helpers
tests/test_pipeline.py ........... 4 passed   # End-to-end on 3 scenarios
tests/test_retrieval.py .......... 10 passed  # Eq. 14–16

============================== 30 passed ==============================
```

The tests assert:
- Token approximation matches the paper at `C_avg = 4` for representative inputs.
- Padding/truncation enforces the chunk-token cap exactly.
- Top-k retrieval orders by similarity and handles edge cases (empty matrix, k = 0, dim mismatch).
- Attention weights sum to 1 and are higher for events more similar to the query.
- The end-to-end pipeline runs offline on three synthetic scenarios and produces a structured report referencing the paper's system prompt.
- The accuracy equation rejects empty reports, divides correctly, and produces stable section/total breakdowns.

---

## 🗂️ Repo layout

```
yushin-gendfir-rag/
├── gendfir/
│   ├── __init__.py
│   ├── chunking.py         # Eq. 1–4 — event-length-aware chunking
│   ├── embeddings.py       # Ollama / Sentence-Transformers / hash fallback
│   ├── llm.py              # OllamaLLM (paper) / TemplateFallbackLLM
│   ├── retrieval.py        # Eq. 14–16 — top-k + scaled-dot-product attention
│   ├── prompts.py          # DFIR agent prompt (paper Listing 2, verbatim)
│   ├── metrics.py          # Eq. 20 + Relevance + EM + top-k recall
│   ├── pipeline.py         # 4-stage orchestrator
│   └── cli.py              # python -m gendfir.cli ...
├── tests/                  # 30 tests covering the equations + end-to-end
├── examples/scenarios/
│   ├── unauthorised_access.csv   # Mirrors paper Table V (Sec. V-A)
│   ├── lateral_movement.csv      # Added — ATT&CK lateral-movement chain
│   └── syn_flood.csv             # Mirrors paper Table IV
├── docs/
│   ├── architecture.md     # Stage-by-stage equation-to-code mapping
│   └── evaluation.md       # How to run accuracy / relevance evaluation
├── CITATION.cff            # GitHub auto-citation, points at the paper
├── LICENSE                 # MIT + academic-provenance notice
├── pyproject.toml
└── requirements.txt
```

---

## 🧩 What was extended beyond the paper

The paper's reference repository ships a working implementation focused on the experimental scenarios reported in §V. This replication intentionally separates concerns into modules so that each component can be reused or swapped:

1. **Embedder abstraction** with three backends — Ollama (paper baseline), Sentence-Transformers (CPU), and a deterministic hash embedder (for tests / offline CI).
2. **LLM abstraction** with a template fallback so the package can be imported and tested in environments without Ollama.
3. **`metrics` module** that explicitly distinguishes Knowledge-Base facts vs Additional-Insights facts (per the paper's Sec. V-B taxonomy), so failure modes can be attributed to either retrieval or LLM extrapolation.
4. **30 unit tests** that pin down the equations as executable specifications.
5. **Three synthetic scenarios** including an ATT&CK lateral-movement chain (T1078 → T1548 → T1059.001 → T1550/T1558 → T1021.002) not in the paper.

All extensions are clearly labelled in code and docs as "added by this replication".

---

## ⚖️ Honest limitations

- **Synthetic data only.** The example scenarios are fabricated for testing. Performance numbers from this repo do not transfer directly to real incident data.
- **No automatic correctness scoring.** Accuracy and Relevance still require a human (or a strong judge model) to classify facts. The package automates only the arithmetic, exactly like the paper does.
- **Cosine similarity, not bare dot product.** The retrieval scoring uses cosine to be norm-invariant. For strict paper parity, swap in a dot-product variant in `topk_indices`. The behavioural difference is small for sentence-transformer-class embeddings.
- **Single-CSV input.** Multi-source merging (firewall + EDR + DC logs) is left as an extension — see `docs/architecture.md`.

---

## 📚 Citation

If you use this code in academic work, please cite the **original paper** as the primary source:

```bibtex
@misc{loumachi2024gendfir,
  title        = {{GenDFIR}: Advancing Cyber Incident Timeline Analysis Through
                  Retrieval-Augmented Generation and Large Language Models},
  author       = {Loumachi, Fatma Yasmine and Ghanem, Mohamed Chahine and
                  Ferrag, Mohamed Amine},
  year         = {2024},
  howpublished = {\url{https://github.com/GenDFIR}},
  note         = {Cyber Security Research Centre, London Metropolitan University, UK;
                  Cybersecurity Institute, University of Liverpool, UK.}
}
```

If you specifically reference the implementation in this repository, you may additionally cite:

```bibtex
@software{bang2026gendfirrag,
  author       = {Bang, Juwon (YuShin)},
  title        = {{yushin-gendfir-rag}: An unofficial replication of GenDFIR
                  (Loumachi, Ghanem \& Ferrag, 2024)},
  year         = {2026},
  url          = {https://github.com/Juwon1405/yushin-gendfir-rag}
}
```

A `CITATION.cff` is included so GitHub's "Cite this repository" widget renders both references automatically.

---

## 🤝 Companion projects

- **[yushin-mac-artifact-collector](https://github.com/Juwon1405/yushin-mac-artifact-collector)** — collect macOS DFIR artefacts as a single-file bash script.
- **[yushin-mac-forensics-platform](https://github.com/Juwon1405/yushin-mac-forensics-platform)** — Flask-based platform that ingests collector ZIPs and produces searchable evidence tables + DFIR PDF reports.
- **[yushin-dfir](https://github.com/Juwon1405/yushin-dfir)** — autonomous DFIR agent (architecture-first), submission to SANS FIND EVIL! 2026.

---

## 📄 License

MIT — see [LICENSE](LICENSE), which also includes an academic-provenance notice.

## ✍️ Author

**YuShin (優心 / Bang Juwon)** — DFIR practitioner, Tokyo.

> *"Read the paper. Verify the math. Pin the equations as tests. Then extend."*

If this study replication helps your own DFIR + LLM research, a ⭐ on the repo means a lot. Issues / PRs welcome.
