"""Databricks U2M OAuth PKCE helpers for browser login (PWAF).

Avoids importing ``openhands.sdk.llm.providers.databricks`` at module import time
so the web app can start when an older ``openhands-sdk`` without the native provider
is installed. Token exchange returns a plain dict for session storage.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import time
from typing import Any

import httpx


def generate_pkce() -> tuple[str, str]:
    """Return (verifier, challenge). Challenge is S256 of verifier."""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b'=').decode()
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b'=').decode()
    return verifier, challenge


def build_authorize_url(
    host: str, client_id: str, redirect_uri: str, state: str, challenge: str
) -> str:
    """Build Databricks OIDC authorize URL with PKCE (S256)."""
    from urllib.parse import urlencode

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


def exchange_code_for_tokens(
    host: str,
    client_id: str,
    redirect_uri: str,
    code: str,
    verifier: str,
    client_secret: str | None = None,
) -> dict[str, Any]:
    """Exchange authorization code for tokens. Sends PWAF User-Agent.

    ``client_secret`` is required for **confidential** OAuth apps (apps that have
    a secret registered in Databricks App connections). Public apps omit it.
    Omitting it for a confidential app returns ``{"error": "invalid_client"}``.

    Returns a dict compatible with ``StoredU2MTokens`` / ``model_validate``.
    """
    try:
        from openhands.sdk.llm.providers.databricks.utils import USER_AGENT
    except ImportError:
        USER_AGENT = 'OpenHandsOSS/unknown'

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

    resp = httpx.post(
        f'{host}/oidc/v1/token',
        data=token_data,
        headers={'User-Agent': USER_AGENT},
        timeout=15.0,
    )
    resp.raise_for_status()
    data = resp.json()
    return {
        'access_token': data['access_token'],
        'refresh_token': data['refresh_token'],
        'expires_at': time.time() + data.get('expires_in', 3600),
        'client_id': client_id,
        'host': host,
    }
