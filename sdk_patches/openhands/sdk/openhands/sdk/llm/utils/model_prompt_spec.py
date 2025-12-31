"""Utilities for detecting model families and variants.

These helpers allow prompts and other systems to tailor behavior for specific
LLM providers while keeping naming heuristics centralized.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelPromptSpec(BaseModel):
    """Detected prompt metadata for a given model configuration."""

    model_config = ConfigDict(frozen=True)

    family: str | None = None
    variant: str | None = None


_MODEL_FAMILY_PATTERNS: dict[str, tuple[str, ...]] = {
    "openai_gpt": (
        "gpt-",
        "o1",
        "o3",
        "o4",
    ),
    "anthropic_claude": ("claude",),
    "google_gemini": ("gemini",),
    "meta_llama": ("llama",),
    "mistral": ("mistral",),
    "deepseek": ("deepseek",),
    "alibaba_qwen": ("qwen",),
}

# Ordered heuristics to pick the most specific variant available for a family.
_MODEL_VARIANT_PATTERNS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "openai_gpt": (
        ("gpt-5-codex", ("gpt-5-codex", "gpt-5.1-codex")),
        ("gpt-5", ("gpt-5", "gpt-5.1")),
    ),
}


def _normalize(name: str | None) -> str:
    return (name or "").strip().lower()


def _match_family(model_name: str) -> str | None:
    normalized = _normalize(model_name)
    if not normalized:
        return None

    for family, patterns in _MODEL_FAMILY_PATTERNS.items():
        if any(pattern in normalized for pattern in patterns):
            return family
    return None


def _match_variant(
    family: str,
    model_name: str,
    canonical_name: str | None = None,
) -> str | None:
    patterns = _MODEL_VARIANT_PATTERNS.get(family)
    if not patterns:
        return None

    # Choose canonical_name if available, otherwise fall back to model_name
    candidate = _normalize(canonical_name) or _normalize(model_name)
    if not candidate:
        return None

    for variant, substrings in patterns:
        if any(sub in candidate for sub in substrings):
            return variant

    return None


def get_model_prompt_spec(
    model_name: str,
    canonical_name: str | None = None,
) -> ModelPromptSpec:
    """Return family and variant prompt metadata for the given identifiers."""

    family = _match_family(model_name)
    if family is None and canonical_name:
        family = _match_family(canonical_name)

    variant = None
    if family is not None:
        variant = _match_variant(family, model_name, canonical_name)

    return ModelPromptSpec(family=family, variant=variant)


__all__ = ["ModelPromptSpec", "get_model_prompt_spec"]
