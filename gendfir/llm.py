"""
LLM backends.

Paper baseline: Llama 3.1 8B (zero-shot) via Ollama
(Loumachi et al., 2024, Sec. IV-B). This module preserves that as the
primary backend and adds a deterministic template fallback that runs
without any LLM — handy for offline tests and as a contract example.
"""
from __future__ import annotations

import textwrap


class LLM:
    """Abstract base class for LLM backends."""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.1,
    ) -> str:
        raise NotImplementedError


class OllamaLLM(LLM):
    """
    Ollama-backed LLM.

    Default model is `llama3.1:8b` to match the paper. Any Ollama-served
    model works (e.g., `qwen2.5:14b-instruct-q4_K_M`, `mistral:7b`).

    The paper uses a low temperature (0.1) for deterministic, factual
    timeline reconstruction.
    """

    def __init__(self, model: str = "llama3.1:8b"):
        try:
            import ollama
        except ImportError as e:
            raise ImportError(
                "ollama package not installed — install via "
                "`pip install ollama`."
            ) from e
        self._client = ollama
        self.model = model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.1,
    ) -> str:
        full = f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}"
        response = self._client.generate(
            model=self.model,
            prompt=full,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )
        return response["response"]


class TemplateFallbackLLM(LLM):
    """
    Deterministic template fallback when no real LLM is available.

    Returns a structured DFIR-report skeleton with the retrieved evidence
    embedded verbatim. This keeps the pipeline contract intact for CI,
    documentation rendering, and dry runs — and gives a paste-ready
    template the user can hand to any external LLM.
    """

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.1,
    ) -> str:
        return textwrap.dedent(
            f"""\
            [TemplateFallbackLLM — no real LLM connected]

            ── System Prompt ──
            {system_prompt}

            ── User Prompt ──
            {user_prompt}

            ── DFIR Timeline-Analysis Report Template ──
            1) Incident Overview
               (Summarise the scenario in 2–3 sentences using the retrieved evidence.)

            2) Reconstructed Event Timeline (chronological)
               (List each retrieved event with its timestamp and identifier.)

            3) Correlations & Anomalies
               (Group events that share a subject/host/credential; flag deviations.)

            4) Evidence Citations
               (For every claim above, cite the exact evidence snippet in parentheses.)

            5) Recommendations
               - Immediate containment actions
               - Short-term eradication / hardening
               - Long-term detection-engineering improvements

            (Note: do not introduce information beyond the retrieved evidence.)
            """
        )


def make_llm(
    prefer_ollama: bool = True,
    model: str = "llama3.1:8b",
) -> LLM:
    """
    Factory: pick the best available LLM.

    Order of attempts:
        1. OllamaLLM (if `prefer_ollama` is True)
        2. TemplateFallbackLLM (always available)
    """
    if prefer_ollama:
        try:
            return OllamaLLM(model)
        except Exception:
            pass
    return TemplateFallbackLLM()
