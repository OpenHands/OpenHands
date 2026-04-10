"""ChatGPT subscription device OAuth (same endpoints as LiteLLM / Codex CLI)."""

from __future__ import annotations

import time
from typing import Any

import httpx

CHATGPT_AUTH_BASE = 'https://auth.openai.com'
CHATGPT_CLIENT_ID = 'app_EMoamEEZ73f0CkXaXp7hrann'
CHATGPT_DEVICE_VERIFY_URL = f'{CHATGPT_AUTH_BASE}/codex/device'
CHATGPT_DEVICE_CODE_URL = f'{CHATGPT_AUTH_BASE}/api/accounts/deviceauth/usercode'
CHATGPT_DEVICE_TOKEN_URL = f'{CHATGPT_AUTH_BASE}/api/accounts/deviceauth/token'
CHATGPT_OAUTH_TOKEN_URL = f'{CHATGPT_AUTH_BASE}/oauth/token'
DEVICE_CODE_TIMEOUT_SECONDS = 15 * 60
DEVICE_CODE_POLL_SLEEP_SECONDS = 5


def request_device_code() -> dict[str, str]:
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            CHATGPT_DEVICE_CODE_URL,
            json={'client_id': CHATGPT_CLIENT_ID},
        )
        resp.raise_for_status()
        data = resp.json()
    device_auth_id = data.get('device_auth_id')
    user_code = data.get('user_code') or data.get('usercode')
    interval = data.get('interval')
    if not device_auth_id or not user_code:
        raise ValueError(f'Invalid device code response: {data!r}')
    return {
        'device_auth_id': str(device_auth_id),
        'user_code': str(user_code),
        'interval': str(interval or '5'),
    }


def poll_once(device_auth_id: str, user_code: str) -> dict[str, Any] | None:
    """Returns authorization payload when ready, None if still pending."""
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            CHATGPT_DEVICE_TOKEN_URL,
            json={
                'device_auth_id': device_auth_id,
                'user_code': user_code,
            },
        )
    if resp.status_code in (403, 404):
        return None
    resp.raise_for_status()
    data = resp.json()
    if all(
        k in data
        for k in ('authorization_code', 'code_challenge', 'code_verifier')
    ):
        return data
    return None


def exchange_authorization_code(code_data: dict[str, str]) -> dict[str, str]:
    redirect_uri = f'{CHATGPT_AUTH_BASE}/deviceauth/callback'
    body = (
        'grant_type=authorization_code'
        f"&code={code_data['authorization_code']}"
        f'&redirect_uri={redirect_uri}'
        f'&client_id={CHATGPT_CLIENT_ID}'
        f"&code_verifier={code_data['code_verifier']}"
    )
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            CHATGPT_OAUTH_TOKEN_URL,
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            content=body,
        )
        resp.raise_for_status()
        data = resp.json()
    if not all(k in data for k in ('access_token', 'refresh_token', 'id_token')):
        raise ValueError(f'Token response missing fields: {data!r}')
    return {
        'access_token': data['access_token'],
        'refresh_token': data['refresh_token'],
        'id_token': data['id_token'],
    }


def poll_until_authorized(
    device_auth_id: str,
    user_code: str,
    interval: int,
) -> dict[str, str]:
    start = time.time()
    sleep_s = max(interval, DEVICE_CODE_POLL_SLEEP_SECONDS)
    while time.time() - start < DEVICE_CODE_TIMEOUT_SECONDS:
        payload = poll_once(device_auth_id, user_code)
        if payload is not None:
            return exchange_authorization_code(payload)
        time.sleep(sleep_s)
    raise TimeoutError('Timed out waiting for ChatGPT device authorization')
