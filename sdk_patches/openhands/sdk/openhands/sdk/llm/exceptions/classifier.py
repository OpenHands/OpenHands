from __future__ import annotations

from litellm.exceptions import BadRequestError, ContextWindowExceededError, OpenAIError

from .types import LLMContextWindowExceedError


# Minimal, provider-agnostic context-window detection
LONG_PROMPT_PATTERNS: list[str] = [
    "contextwindowexceedederror",
    "prompt is too long",
    "input length and `max_tokens` exceed context limit",
    "please reduce the length of",
    "the request exceeds the available context size",
    "context length exceeded",
    "input exceeds the context window",
]


def is_context_window_exceeded(exception: Exception) -> bool:
    if isinstance(exception, (ContextWindowExceededError, LLMContextWindowExceedError)):
        return True

    if not isinstance(exception, (BadRequestError, OpenAIError)):
        return False

    s = str(exception).lower()
    return any(p in s for p in LONG_PROMPT_PATTERNS)


AUTH_PATTERNS: list[str] = [
    "invalid api key",
    "unauthorized",
    "missing api key",
    "invalid authentication",
    "access denied",
]


def looks_like_auth_error(exception: Exception) -> bool:
    if not isinstance(exception, (BadRequestError, OpenAIError)):
        return False
    s = str(exception).lower()
    if any(p in s for p in AUTH_PATTERNS):
        return True
    # Some providers include explicit status codes in message text
    for code in ("status 401", "status 403"):
        if code in s:
            return True
    return False
