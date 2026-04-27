# Synthetic DFIR Scenarios

These three CSVs are **synthetic** test datasets used to exercise the
`gendfir-rag` pipeline end-to-end. They are *not* lifted from real
incidents — every IP, hostname, and username is fabricated.

| Scenario | File | Events | Mirrors Paper Scenario |
|----------|------|-------:|------------------------|
| Unauthorised Access | `scenarios/unauthorised_access.csv` | 20 | Yes — Table IV / V (extended slightly here for illustration) |
| Lateral Movement | `scenarios/lateral_movement.csv` | 13 | No — added by this replication for ATT&CK lateral-movement coverage |
| SYN Flood | `scenarios/syn_flood.csv` | 14 | Yes — Table IV |

## Why these three

- **Unauthorised Access** is the scenario the paper showcases in full
  (Table V/VI/VII/VIII). Reproducing it exactly is the cleanest way to
  validate the replication.
- **Lateral Movement** extends the paper into a 4624 → 4672 → 4688
  (encoded PowerShell) → 4769 (Kerberos ST) chain that corresponds to
  MITRE ATT&CK techniques T1078, T1548, T1059.001, T1550/T1558, T1021.002
  — useful for ATT&CK-aware retrieval experiments.
- **SYN Flood** stress-tests the pipeline on a highly repetitive,
  high-cardinality event stream (per-second SYN counts).

## Running them

```bash
# Faithful paper baseline (needs Ollama with llama3.1:8b + mxbai-embed-large)
python -m gendfir.cli \
    --csv examples/scenarios/unauthorised_access.csv \
    --query "Conduct DFIR timeline analysis on the unauthorised access incident" \
    --topk 15

# Smoke run with no Ollama — uses sentence-transformers + template fallback
python -m gendfir.cli \
    --csv examples/scenarios/lateral_movement.csv \
    --query "Reconstruct the attacker timeline. Identify privilege escalation and lateral movement." \
    --no-ollama
```

## Defining your own

Any CSV works — the column schema is free-form. Each row becomes one
event sentence (attributes joined with `, `, ending with `.`). Keep
each row under ~2,000 characters for safe embedding (paper default:
512 tokens × 4 chars/token = 2,048).
