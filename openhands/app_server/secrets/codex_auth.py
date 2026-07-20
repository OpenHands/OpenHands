import json
import os
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import httpx
from fastapi import HTTPException, status

CODEX_AUTH_ROUTE_PREFIX = '/api/internal/conversations'
CODEX_AUTH_ROUTE = '/{conversation_id}/codex-auth'
CODEX_REFRESH_CLIENT_ID = 'app_EMoamEEZ73f0CkXaXp7hrann'
_REFRESH_TOKEN_URL = 'https://auth.openai.com/oauth/token'


def codex_auth_path(conversation_id: UUID) -> str:
    return f'{CODEX_AUTH_ROUTE_PREFIX}{CODEX_AUTH_ROUTE.format(conversation_id=conversation_id)}'


def is_chatgpt_codex_auth(value: str) -> bool:
    try:
        document = json.loads(value)
    except (TypeError, ValueError):
        return False
    if not isinstance(document, dict):
        return False
    if document.get('auth_mode') not in (None, 'chatgpt'):
        return False
    tokens = document.get('tokens')
    return (
        isinstance(tokens, dict)
        and isinstance(tokens.get('refresh_token'), str)
        and bool(tokens['refresh_token'])
    )


def codex_token_payload(value: str) -> dict[str, str]:
    try:
        tokens = json.loads(value)['tokens']
    except (KeyError, TypeError, ValueError):
        tokens = None
    if not isinstance(tokens, dict):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail='Stored Codex authentication needs to be refreshed',
        )
    payload = {
        key: token
        for key in ('id_token', 'access_token', 'refresh_token')
        if isinstance((token := tokens.get(key)), str) and token
    }
    if 'refresh_token' not in payload:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail='Stored Codex authentication needs to be refreshed',
        )
    return payload


def merge_codex_refresh(value: str, refresh: dict[str, Any]) -> str:
    if not isinstance(refresh.get('access_token'), str) or not refresh['access_token']:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            detail='Codex credential refresh returned an invalid response',
        )
    document = json.loads(value)
    tokens = document['tokens']
    for key in ('id_token', 'access_token', 'refresh_token'):
        token = refresh.get(key)
        if isinstance(token, str) and token:
            tokens[key] = token
    document['last_refresh'] = datetime.now(UTC).isoformat().replace('+00:00', 'Z')
    updated = json.dumps(document, separators=(',', ':'))
    if not is_chatgpt_codex_auth(updated):
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            detail='Codex credential refresh returned invalid authentication',
        )
    return updated


async def request_codex_token_refresh(refresh_token: str) -> httpx.Response:
    url = os.getenv('OPENHANDS_CODEX_REFRESH_TOKEN_URL', _REFRESH_TOKEN_URL)
    async with httpx.AsyncClient(timeout=30.0) as client:
        return await client.post(
            url,
            json={
                'client_id': CODEX_REFRESH_CLIENT_ID,
                'grant_type': 'refresh_token',
                'refresh_token': refresh_token,
            },
        )


def codex_refresh_error(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
        error = payload.get('error')
        code = error.get('code') if isinstance(error, dict) else None
    except (AttributeError, TypeError, ValueError):
        code = None
    return {'error': {'code': code or 'credential_refresh_failed'}}
