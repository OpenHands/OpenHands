"""Shared bearer-token auth for local OpenHands demo MCP servers."""

from __future__ import annotations

import os

from fastmcp.server.auth import StaticTokenVerifier

# Default key for local testing. Override with DEMO_MCP_API_KEY env var.
DEMO_API_KEY = os.getenv('DEMO_MCP_API_KEY', 'demo-secret-key')


def demo_auth_provider() -> StaticTokenVerifier:
    """Return a fastmcp auth provider that accepts ``Bearer <DEMO_API_KEY>``."""
    return StaticTokenVerifier(
        tokens={
            DEMO_API_KEY: {'client_id': 'demo', 'scopes': ['mcp']},
        }
    )


def authorization_headers(api_key: str | None = None) -> dict[str, str]:
    key = api_key if api_key is not None else DEMO_API_KEY
    return {'Authorization': f'Bearer {key}'}
