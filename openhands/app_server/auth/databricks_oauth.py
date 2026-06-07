"""Databricks U2M OAuth PKCE helpers for browser login (PWAF).

These helpers now live canonically in the OpenHands SDK
(``openhands.sdk.llm.providers.databricks.pkce``) so the web app, the CLI, and
the SDK share a single implementation. This module re-exports them.

A local fallback is kept for the case where an **older** ``openhands-sdk``
(predating the ``pkce`` module) is installed — the web app must still import
and start in that situation. Once the pinned SDK release contains ``pkce`` the
fallback is dead code and can be removed.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import time
from typing import Any
from urllib.parse import urlencode

import httpx

try:  # Preferred: shared SDK implementation.
    from openhands.sdk.llm.providers.databricks.pkce import (
        async_exchange_code_for_tokens,
        build_authorize_url,
        exchange_code_for_tokens,
        generate_pkce,
    )

    _USING_SDK_PKCE = True
except ImportError:  # Fallback for older openhands-sdk without the pkce module.
    _USING_SDK_PKCE = False

    try:
        from openhands.sdk.llm.providers.databricks.utils import USER_AGENT
    except ImportError:
        USER_AGENT = 'OpenHandsOSS/unknown'

    def generate_pkce() -> tuple[str, str]:
        """Return (verifier, challenge). Challenge is S256 of verifier."""
        verifier = (
            base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b'=').decode()
        )
        digest = hashlib.sha256(verifier.encode()).digest()
        challenge = base64.urlsafe_b64encode(digest).rstrip(b'=').decode()
        return verifier, challenge

    def build_authorize_url(
        host: str, client_id: str, redirect_uri: str, state: str, challenge: str
    ) -> str:
        """Build Databricks OIDC authorize URL with PKCE (S256)."""
        host = host.rstrip('/')
        params = {
            'response_type': 'code',
            'client_id': client_id,
            'redirect_uri': redirect_uri,
            'scope': 'all-apis offline_access',
            'state': state,
            'code_challenge': challenge,
            'code_challenge_method': 'S256',
        }
        return f'{host}/oidc/v1/authorize?{urlencode(params)}'

    def _token_request(
        host: str,
        client_id: str,
        redirect_uri: str,
        code: str,
        verifier: str,
        client_secret: str | None,
    ) -> tuple[str, dict[str, str]]:
        host = host.rstrip('/')
        token_data: dict[str, str] = {
            'grant_type': 'authorization_code',
            'code': code,
            'redirect_uri': redirect_uri,
            'client_id': client_id,
            'code_verifier': verifier,
        }
        if client_secret:
            token_data['client_secret'] = client_secret
        return f'{host}/oidc/v1/token', token_data

    def _shape(data: dict[str, Any], client_id: str, host: str) -> dict[str, Any]:
        return {
            'access_token': data['access_token'],
            'refresh_token': data.get('refresh_token', ''),
            'expires_at': time.time() + data.get('expires_in', 3600),
            'client_id': client_id,
            'host': host.rstrip('/'),
        }

    def exchange_code_for_tokens(
        host: str,
        client_id: str,
        redirect_uri: str,
        code: str,
        verifier: str,
        client_secret: str | None = None,
    ) -> dict[str, Any]:
        """Synchronous code → tokens exchange (PWAF User-Agent)."""
        url, data = _token_request(
            host, client_id, redirect_uri, code, verifier, client_secret
        )
        resp = httpx.post(
            url, data=data, headers={'User-Agent': USER_AGENT}, timeout=15.0
        )
        resp.raise_for_status()
        return _shape(resp.json(), client_id, host)

    async def async_exchange_code_for_tokens(
        host: str,
        client_id: str,
        redirect_uri: str,
        code: str,
        verifier: str,
        client_secret: str | None = None,
    ) -> dict[str, Any]:
        """Async code → tokens exchange — does not block the event loop."""
        url, data = _token_request(
            host, client_id, redirect_uri, code, verifier, client_secret
        )
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, data=data, headers={'User-Agent': USER_AGENT})
        resp.raise_for_status()
        return _shape(resp.json(), client_id, host)


__all__ = [
    'generate_pkce',
    'build_authorize_url',
    'exchange_code_for_tokens',
    'async_exchange_code_for_tokens',
]
