from dataclasses import dataclass


def model_matches(model: str, patterns: list[str]) -> bool:
    """Return True if any pattern appears as a substring in the raw model name.

    Matching semantics:
    - Case-insensitive substring search on full raw model string
    """
    raw = (model or "").strip().lower()
    for pat in patterns:
        token = pat.strip().lower()
        if token in raw:
            return True
    return False


def apply_ordered_model_rules(model: str, rules: list[str]) -> bool:
    """Apply ordered include/exclude model rules to determine final support.

    Rules semantics:
    - Each entry is a substring token. '!' prefix marks an exclude rule.
    - Case-insensitive substring matching against the raw model string.
    - Evaluated in order; the last matching rule wins.
    - If no rule matches, returns False.
    """
    raw = (model or "").strip().lower()
    decided: bool | None = None
    for rule in rules:
        token = rule.strip().lower()
        if not token:
            continue
        is_exclude = token.startswith("!")
        core = token[1:] if is_exclude else token
        if core and core in raw:
            decided = not is_exclude
    return bool(decided)


@dataclass(frozen=True)
class ModelFeatures:
    supports_reasoning_effort: bool
    supports_extended_thinking: bool
    supports_prompt_cache: bool
    supports_stop_words: bool
    supports_responses_api: bool
    force_string_serializer: bool
    send_reasoning_content: bool
    supports_prompt_cache_retention: bool


# Model lists capturing current behavior. Keep entries lowercase.

REASONING_EFFORT_MODELS: list[str] = [
    # Mirror main behavior exactly (no unintended expansion)
    "o1-2024-12-17",
    "o1",
    "o3",
    "o3-2025-04-16",
    "o3-mini-2025-01-31",
    "o3-mini",
    "o4-mini",
    "o4-mini-2025-04-16",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    # OpenAI GPT-5 family (includes mini variants)
    "gpt-5",
    # Anthropic Opus 4.5
    "claude-opus-4-5",
    # Nova 2 Lite
    "nova-2-lite",
]

EXTENDED_THINKING_MODELS: list[str] = [
    # Anthropic model family
    # We did not include sonnet 3.7 and 4 here as they don't brings
    # significant performance improvements for agents
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
]

PROMPT_CACHE_MODELS: list[str] = [
    "claude-3-7-sonnet",
    "claude-sonnet-3-7-latest",
    "claude-3-5-sonnet",
    "claude-3-5-haiku",
    "claude-3-haiku-20240307",
    "claude-3-opus-20240229",
    "claude-sonnet-4",
    "claude-opus-4",
    # Anthropic Haiku 4.5 variants (dash only; official IDs use hyphens)
    "claude-haiku-4-5",
    "claude-opus-4-5",
]

# Models that support a top-level prompt_cache_retention parameter
# Source: OpenAI Prompt Caching docs (extended retention), which list:
#   - gpt-5.2
#   - gpt-5.1
#   - gpt-5.1-codex
#   - gpt-5.1-codex-mini
#   - gpt-5.1-chat-latest
#   - gpt-5
#   - gpt-5-codex
#   - gpt-4.1
# Use ordered include/exclude rules (last wins) to naturally express exceptions.
PROMPT_CACHE_RETENTION_MODELS: list[str] = [
    # Broad allow for GPT-5 family and GPT-4.1 (covers gpt-5.2 and variants)
    "gpt-5",
    "gpt-4.1",
    # Exclude all mini variants by default
    "!mini",
    # Re-allow the explicitly documented supported mini variant
    "gpt-5.1-codex-mini",
]

SUPPORTS_STOP_WORDS_FALSE_MODELS: list[str] = [
    # o-series families don't support stop words
    "o1",
    "o3",
    # grok-4 specific model name (basename)
    "grok-4-0709",
    "grok-code-fast-1",
    # DeepSeek R1 family
    "deepseek-r1-0528",
]

# Models that should use the OpenAI Responses API path by default
RESPONSES_API_MODELS: list[str] = [
    # OpenAI GPT-5 family (includes mini variants)
    "gpt-5",
    # OpenAI Codex (uses Responses API)
    "codex-mini-latest",
]

# Models that require string serializer for tool messages
# These models don't support structured content format [{"type":"text","text":"..."}]
# and need plain strings instead
# NOTE: model_matches uses case-insensitive substring matching, not globbing.
#       Keep these entries as bare substrings without wildcards.
FORCE_STRING_SERIALIZER_MODELS: list[str] = [
    "deepseek",  # e.g., DeepSeek-V3.2-Exp
    "glm",  # e.g., GLM-4.5 / GLM-4.6
    # Kimi K2-Instruct requires string serialization only on Groq
    "groq/kimi-k2-instruct",  # explicit provider-prefixed IDs
    # MiniMax-M2 via OpenRouter rejects array content with
    # "Input should be a valid string" for ChatCompletionToolMessage.content
    "openrouter/minimax",
]

# Models that we should send full reasoning content
# in the message input
SEND_REASONING_CONTENT_MODELS: list[str] = [
    "kimi-k2-thinking",
    "openrouter/minimax-m2",  # MiniMax-M2 via OpenRouter (interleaved thinking)
    "deepseek/deepseek-reasoner",
]


def get_features(model: str) -> ModelFeatures:
    """Get model features."""
    return ModelFeatures(
        supports_reasoning_effort=model_matches(model, REASONING_EFFORT_MODELS),
        supports_extended_thinking=model_matches(model, EXTENDED_THINKING_MODELS),
        supports_prompt_cache=model_matches(model, PROMPT_CACHE_MODELS),
        supports_stop_words=not model_matches(model, SUPPORTS_STOP_WORDS_FALSE_MODELS),
        supports_responses_api=model_matches(model, RESPONSES_API_MODELS),
        force_string_serializer=model_matches(model, FORCE_STRING_SERIALIZER_MODELS),
        send_reasoning_content=model_matches(model, SEND_REASONING_CONTENT_MODELS),
        # Extended prompt_cache_retention support follows ordered include/exclude rules.
        supports_prompt_cache_retention=apply_ordered_model_rules(
            model, PROMPT_CACHE_RETENTION_MODELS
        ),
    )


# Default temperature mapping.
# Each entry: (pattern, default_temperature)
DEFAULT_TEMPERATURE_MODELS: list[tuple[str, float]] = [
    ("kimi-k2-thinking", 1.0),
]


def get_default_temperature(model: str) -> float:
    """Return the default temperature for a given model pattern.

    Uses case-insensitive substring matching via model_matches.
    """
    for pattern, value in DEFAULT_TEMPERATURE_MODELS:
        if model_matches(model, [pattern]):
            return value
    return 0.0
