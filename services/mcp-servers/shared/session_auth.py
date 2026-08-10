"""Session API key resolution for Findings Service calls."""

from __future__ import annotations

import os

SESSION_API_KEY_ENV = "SESSION_API_KEY"
SESSION_API_KEY_HEADER = "X-Session-API-Key"


class MissingSessionApiKeyError(RuntimeError):
    """Raised when SESSION_API_KEY is unset or empty."""


def get_session_api_key() -> str:
    key = os.environ.get(SESSION_API_KEY_ENV, "").strip()
    if not key:
        raise MissingSessionApiKeyError(
            f"{SESSION_API_KEY_ENV} is required for Findings Service auth"
        )
    return key


def session_auth_headers() -> dict[str, str]:
    return {SESSION_API_KEY_HEADER: get_session_api_key()}
