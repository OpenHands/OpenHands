"""Helpers for storing ChatGPT subscription OAuth tokens in user LLM settings."""

from __future__ import annotations

import json
from typing import Any

from pydantic import SecretStr

CHATGPT_OAUTH_JSON_KEY = 'chatgpt'


def encode_chatgpt_token_bundle(access_token: str, refresh_token: str) -> str:
    """Serialize OAuth tokens for storage in llm_api_key."""
    payload: dict[str, Any] = {
        CHATGPT_OAUTH_JSON_KEY: {
            'access_token': access_token,
            'refresh_token': refresh_token,
        }
    }
    return json.dumps(payload, separators=(',', ':'))


def is_chatgpt_oauth_bundle(secret: SecretStr | None) -> bool:
    if not secret:
        return False
    raw = secret.get_secret_value().strip()
    if not raw.startswith('{'):
        return False
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return False
    cg = data.get(CHATGPT_OAUTH_JSON_KEY)
    return isinstance(cg, dict) and bool(cg.get('access_token'))


def decode_chatgpt_access_token_for_llm(
    secret: SecretStr | None, model: str
) -> SecretStr | None:
    """Return the bearer token for LiteLLM when using chatgpt/* models."""
    if not model.startswith('chatgpt/'):
        return secret
    if not secret:
        return None
    raw = secret.get_secret_value().strip()
    if not raw.startswith('{'):
        return secret
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return secret
    cg = data.get(CHATGPT_OAUTH_JSON_KEY)
    if isinstance(cg, dict) and cg.get('access_token'):
        return SecretStr(str(cg['access_token']))
    return secret
