"""Tests for gendfir.chunking — Eq. (1)–(4)."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from gendfir.chunking import (
    EventDoc,
    approx_tokens,
    csv_to_event_docs,
    event_char_length,
)


def test_approx_tokens_paper_default():
    """Eq. (2): T_tokens ≈ T(E_i) / C_avg, with C_avg = 4."""
    # 12 chars / 4 = 3 tokens
    assert approx_tokens(12, chars_per_token=4) == 3
    # 13 chars / 4 = 3.25 → ceil → 4 tokens
    assert approx_tokens(13, chars_per_token=4) == 4
    # Edge: 0 chars
    assert approx_tokens(0, chars_per_token=4) == 0


def test_approx_tokens_invalid_cpt():
    with pytest.raises(ValueError):
        approx_tokens(100, chars_per_token=0)


def test_event_char_length():
    """Eq. (1): T(E_i) = sum_j T(e_ij)."""
    values = ["abc", "de", "", "fgh"]
    assert event_char_length(values) == 3 + 2 + 0 + 3


def _write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        for r in rows:
            w.writerow(r)


def test_csv_to_event_docs_padding(tmp_path: Path):
    """Short events are padded to embed_token_cap * chars_per_token chars."""
    csv_path = tmp_path / "events.csv"
    _write_csv(
        csv_path,
        [
            ["timestamp", "event"],
            ["2024-01-01", "logon"],
            ["2024-01-02", "logoff"],
        ],
    )
    docs = csv_to_event_docs(
        str(csv_path), embed_token_cap=16, chars_per_token=4
    )
    assert len(docs) == 2
    expected_len = 16 * 4
    for d in docs:
        assert isinstance(d, EventDoc)
        assert d.length == expected_len
        # raw form should still be the unpadded sentence ending in "."
        assert d.raw.endswith(".")
        assert "logon" in d.raw or "logoff" in d.raw
        # tokens should equal cap (padded to capacity)
        assert d.tokens == 16


def test_csv_to_event_docs_truncation(tmp_path: Path):
    """Over-long events are truncated to the cap."""
    csv_path = tmp_path / "events.csv"
    long_value = "x" * 1000
    _write_csv(csv_path, [["a"], [long_value]])
    docs = csv_to_event_docs(
        str(csv_path), embed_token_cap=8, chars_per_token=4
    )
    assert len(docs) == 1
    assert docs[0].length == 8 * 4  # 32
    assert docs[0].tokens == 8


def test_csv_to_event_docs_empty_raises(tmp_path: Path):
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("col1,col2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        csv_to_event_docs(str(csv_path))


def test_csv_to_event_docs_skips_blank_rows(tmp_path: Path):
    csv_path = tmp_path / "events.csv"
    _write_csv(
        csv_path,
        [
            ["a", "b"],
            ["", ""],         # all-blank row → should be skipped
            ["evt1", "src1"],
        ],
    )
    docs = csv_to_event_docs(str(csv_path), embed_token_cap=32)
    assert len(docs) == 1
    assert "evt1" in docs[0].raw
