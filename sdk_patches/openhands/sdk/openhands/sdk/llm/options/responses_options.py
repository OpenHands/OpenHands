from __future__ import annotations

from typing import Any

from openhands.sdk.llm.options.common import apply_defaults_if_absent
from openhands.sdk.llm.utils.model_features import get_features


def select_responses_options(
    llm,
    user_kwargs: dict[str, Any],
    *,
    include: list[str] | None,
    store: bool | None,
) -> dict[str, Any]:
    """Behavior-preserving extraction of _normalize_responses_kwargs."""
    # Apply defaults for keys that are not forced by policy
    out = apply_defaults_if_absent(
        user_kwargs,
        {
            "max_output_tokens": llm.max_output_tokens,
        },
    )

    # Enforce sampling/tool behavior for Responses path
    out["temperature"] = 1.0
    out["tool_choice"] = "auto"

    # If user didn't set extra_headers, propagate from llm config
    if llm.extra_headers is not None and "extra_headers" not in out:
        out["extra_headers"] = dict(llm.extra_headers)

    # Store defaults to False (stateless) unless explicitly provided
    if store is not None:
        out["store"] = bool(store)
    else:
        out.setdefault("store", False)

    # Include encrypted reasoning only when the user enables it on the LLM,
    # and only for stateless calls (store=False). Respect user choice.
    include_list = list(include) if include is not None else []

    if not out.get("store", False) and llm.enable_encrypted_reasoning:
        if "reasoning.encrypted_content" not in include_list:
            include_list.append("reasoning.encrypted_content")
    if include_list:
        out["include"] = include_list

    # Include reasoning effort only if explicitly set
    if llm.reasoning_effort:
        out["reasoning"] = {"effort": llm.reasoning_effort}
        # Optionally include summary if explicitly set (requires verified org)
        if llm.reasoning_summary:
            out["reasoning"]["summary"] = llm.reasoning_summary

    # Send prompt_cache_retention only if model supports it
    if (
        get_features(llm.model).supports_prompt_cache_retention
        and llm.prompt_cache_retention
    ):
        out["prompt_cache_retention"] = llm.prompt_cache_retention

    # Pass through user-provided extra_body unchanged
    if llm.litellm_extra_body:
        out["extra_body"] = llm.litellm_extra_body

    return out
