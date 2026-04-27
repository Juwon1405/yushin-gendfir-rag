"""
Event-length-aware chunking, as defined in the GenDFIR paper.

Reference equations (Loumachi et al., 2024, Sec. III-B):

    Eq. (1)  T(E_i) = sum_{j=1..n} T(e_ij)
             — total characters in event E_i (sum of attribute char-lengths)

    Eq. (2)  T_tokens(E_i) ≈ T(E_i) / C_avg,         C_avg = 4
             — approximate tokens, given ~4 chars/token in English

    Eq. (3)  T_tokens(C_k) ≈ T(C_k) / C_avg
             — same relation for a chunk C_k (one chunk = one event)

    Eq. (4)  T_tokens(C_k) ≤ TM
             — chunk must fit the embedding model's max-token capacity TM
               (e.g., 512 for `mxbai-embed-large`)

In this implementation, each row of the input CSV becomes one event
sentence. Attributes are joined with ", " to form a single natural-language
representation of the event, then padded or truncated to keep the
character count at exactly TM * C_avg, ensuring consistent embedding
dimensionality and stable retrieval scores.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class EventDoc:
    """A single chunked event — one CSV row turned into one document."""

    text: str       # The flattened event sentence (padded/truncated)
    length: int     # Character count after padding/truncation
    tokens: int     # Approximate token count (Eq. 2)
    raw: str        # Pre-padding text, useful for evidence snippets


def approx_tokens(char_len: int, chars_per_token: int = 4) -> int:
    """
    Eq. (2)/(3) — approximate token count.

    The paper picks C_avg = 4 to maximise token capture (see Sec. III-B).
    """
    if chars_per_token < 1:
        raise ValueError("chars_per_token must be >= 1")
    return math.ceil(char_len / chars_per_token)


def event_char_length(values: List[str]) -> int:
    """
    Eq. (1) — sum of character counts across attributes of an event.

    The paper notates this as T(E_i) = sum_{j=1..n} T(e_ij). After we
    join attributes with a separator and a trailing period, the character
    length differs slightly from the pure sum, which we treat as the
    canonical T(E_i) for this implementation. The separator overhead is
    an implementation choice noted by the paper as acceptable.
    """
    return sum(len(str(v)) for v in values)


def csv_to_event_docs(
    csv_path: str,
    embed_token_cap: int = 512,
    chars_per_token: int = 4,
    pad_char: str = " ",
) -> List[EventDoc]:
    """
    Load a CSV of incident events and chunk it according to the paper.

    Parameters
    ----------
    csv_path : str
        Path to the incident-events CSV. Any column schema is allowed —
        all non-empty cells in a row are flattened into one event sentence.
    embed_token_cap : int
        Maximum tokens TM the embedding model can ingest per call.
        Defaults to 512 (mxbai-embed-large, all-MiniLM-L6-v2).
    chars_per_token : int
        C_avg — the assumed average characters per token. Paper default: 4.
    pad_char : str
        Padding character for events shorter than the cap. Default: space.

    Returns
    -------
    List[EventDoc]
        One EventDoc per CSV row, each padded/truncated to the cap.

    Raises
    ------
    FileNotFoundError
        If `csv_path` does not exist.
    ValueError
        If the CSV is empty after stripping.
    """
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")

    max_chars_per_chunk = embed_token_cap * chars_per_token
    docs: List[EventDoc] = []

    for _, row in df.iterrows():
        # Flatten attributes -> sentence
        parts = [str(v).strip() for v in row.values if str(v).strip()]
        if not parts:
            continue
        sentence = ", ".join(parts).strip()
        if not sentence.endswith("."):
            sentence += "."

        raw = sentence
        length = len(sentence)

        # Eq. (4) — enforce the chunk-token cap
        if length < max_chars_per_chunk:
            sentence = sentence.ljust(max_chars_per_chunk, pad_char)
        elif length > max_chars_per_chunk:
            sentence = sentence[:max_chars_per_chunk]

        final_length = len(sentence)
        docs.append(
            EventDoc(
                text=sentence,
                length=final_length,
                tokens=approx_tokens(final_length, chars_per_token),
                raw=raw,
            )
        )

    if not docs:
        raise ValueError(f"No usable events in CSV: {csv_path}")

    return docs
