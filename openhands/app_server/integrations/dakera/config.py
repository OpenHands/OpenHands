"""Dakera integration configuration.

All settings are read from environment variables so they can be supplied at
deployment time without touching code.
"""

from __future__ import annotations

import os


class DakeraConfig:
    """Configuration for the Dakera memory integration.

    Read from environment variables with sensible defaults.  The class is
    intentionally a plain Python class (not a Pydantic model) so it can be
    instantiated without a running event loop and requires no external deps.
    """

    def __init__(
        self,
        *,
        api_url: str | None = None,
        api_key: str | None = None,
        agent_id: str | None = None,
        top_k: int | None = None,
        enabled: bool | None = None,
        timeout: float | None = None,
    ) -> None:
        self.api_url: str = api_url or os.getenv(
            'DAKERA_API_URL', 'http://localhost:3300'
        )
        self.api_key: str | None = api_key or os.getenv('DAKERA_API_KEY') or None
        self.agent_id: str = agent_id or os.getenv('DAKERA_AGENT_ID', 'openhands')
        self.top_k: int = top_k if top_k is not None else int(
            os.getenv('DAKERA_TOP_K', '5')
        )
        self.timeout: float = timeout if timeout is not None else float(
            os.getenv('DAKERA_TIMEOUT', '5.0')
        )

        if enabled is not None:
            self.enabled = enabled
        else:
            self.enabled = os.getenv('DAKERA_ENABLED', 'true').lower() not in (
                'false', '0', 'no',
            )

    @property
    def auth_headers(self) -> dict[str, str]:
        """Return Authorization header dict, or empty dict if no key is set."""
        if self.api_key:
            return {'Authorization': f'Bearer {self.api_key}'}
        return {}
