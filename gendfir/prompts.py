"""
Prompts.

The DFIR-agent system prompt below is reproduced verbatim from the
GenDFIR paper's Listing 2 (Loumachi et al., 2024, Sec. III-C):

    "You are a DFIR AI assistant, tasked with analysing artefacts,
     correlating events, and producing a coherent timeline of the
     incident. Base your answer on the provided context and do not
     include additional information outside of the context given."

We keep the wording identical so that comparison studies can attribute
behavioural differences to the model / retrieval pipeline rather than
to prompt drift. Any prompt experiments should be added as new
constants — do NOT modify `DFIR_AGENT_SYSTEM_PROMPT`.
"""
from __future__ import annotations

import textwrap
from typing import List


# Paper Listing 2 — DO NOT MODIFY for replication faithfulness.
DFIR_AGENT_SYSTEM_PROMPT: str = (
    "You are a DFIR AI assistant, tasked with analysing artefacts, "
    "correlating events, and producing a coherent timeline of the "
    "incident. Base your answer on the provided context and do not "
    "include additional information outside of the context given."
)


def build_user_prompt(
    query: str,
    selected_event_snippets: List[str],
    snippet_char_limit: int = 240,
) -> str:
    """
    Compose the user-turn prompt fed to the LLM.

    The prompt mirrors the paper's task framing: chronological timeline,
    event correlation, evidence citation, and recommendations — with an
    explicit no-extrapolation constraint.

    Parameters
    ----------
    query : str
        The DFIR analyst's query (e.g.,
        "Conduct DFIR timeline analysis on the unauthorised access incident").
    selected_event_snippets : List[str]
        Raw text of retrieved events, in retrieval (similarity) order.
    snippet_char_limit : int
        Per-snippet character cap to keep prompt size predictable.
    """
    bullets = []
    for snippet in selected_event_snippets:
        clean = snippet.strip().replace("\n", " ")
        if len(clean) > snippet_char_limit:
            clean = clean[:snippet_char_limit] + "…"
        bullets.append(f"- {clean}")
    context_block = "\n".join(bullets) if bullets else "- (no evidence retrieved)"

    return textwrap.dedent(
        f"""\
        [DFIR Query]
        {query}

        [Retrieved Evidence Snippets]
        {context_block}

        [Task]
        1) Reconstruct the incident timeline in chronological order using
           ONLY the evidence above.
        2) Explain correlations between events and flag anomalies.
        3) Cite the supporting evidence snippet (in parentheses) for every
           material claim.
        4) Conclude with investigator recommendations:
           - Immediate (containment) actions
           - Short-term (eradication / hardening) actions
           - Long-term (detection-engineering) improvements
        (Do NOT introduce information beyond the retrieved context.)
        """
    ).strip()
