import base64
import hashlib
import hmac
import json
import os
import re
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import httpx
import jwt
from fastapi import APIRouter, Header, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from storage.codex_auth_store import CodexAuthStore
from storage.redis import get_redis_client_async, redis_exceptions

from openhands.app_server.config import depends_jwt_service
from openhands.app_server.constants import MAX_API_SECRET_VALUE_LENGTH
from openhands.app_server.sandbox.session_auth import (
    validate_session_key,
    validate_teardown_session_key,
)
from openhands.app_server.secrets.codex_auth import (
    CODEX_AUTH_ROUTE,
    CODEX_AUTH_ROUTE_PREFIX,
    is_chatgpt_codex_auth,
)
from openhands.app_server.services.jwt_service import JwtService

router = APIRouter(prefix=CODEX_AUTH_ROUTE_PREFIX)
jwt_service_dependency = depends_jwt_service()

_SECRET_NAME = 'CODEX_AUTH_JSON'
_REFRESH_CLIENT_ID = 'app_EMoamEEZ73f0CkXaXp7hrann'
_REFRESH_LOCK_TTL_SECONDS = 120
_REFRESH_LOCK_WAIT_SECONDS = 30
_REFRESH_TOKEN_URL = 'https://auth.openai.com/oauth/token'
_DIGEST_PATTERN = re.compile(r'^[0-9a-f]{64}$')


@dataclass(frozen=True)
class _CodexAuthScope:
    user_id: str
    org_id: UUID
    sandbox_id: str
    conversation_id: UUID
    token_digest: str
    expires_at: int

    @property
    def refresh_lock_key(self) -> str:
        scope = f'{self.user_id}\0{self.org_id}'
        scope_digest = hashlib.sha256(scope.encode()).hexdigest()
        return f'codex-auth-refresh:{scope_digest}'

    @property
    def revocation_key(self) -> str:
        return f'codex-auth-revoked:{self.token_digest}'


def _token_digest(token: str) -> str:
    encoded_signature = token.rsplit('.', 1)[-1]
    padding = '=' * (-len(encoded_signature) % 4)
    signature = base64.urlsafe_b64decode(f'{encoded_signature}{padding}')
    return hashlib.sha256(signature).hexdigest()


async def _authorize(
    conversation_id: UUID,
    session_api_key: str | None,
    codex_auth_token: str | None,
    jwt_service: JwtService,
    *,
    allow_revoked: bool = False,
    allow_paused_teardown: bool = False,
) -> _CodexAuthScope:
    if not codex_auth_token:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail='X-OH-Codex header is required',
        )
    try:
        sandbox = await validate_session_key(session_api_key)
    except HTTPException as exc:
        if not allow_paused_teardown or exc.status_code != status.HTTP_401_UNAUTHORIZED:
            raise
        sandbox = await validate_teardown_session_key(session_api_key)
    try:
        claims = jwt_service.verify_jws_token(codex_auth_token)
        org_id = UUID(str(claims['org_id']))
        user_id = str(claims['user_id'])
        sandbox_id = str(claims['sandbox_id'])
        scoped_conversation_id = UUID(str(claims['conversation_id']))
        expires_at = int(claims['exp'])
        token_digest = _token_digest(codex_auth_token)
    except (
        KeyError,
        TypeError,
        ValueError,
        jwt.InvalidTokenError,
    ) as exc:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, detail='Invalid Codex auth token'
        ) from exc
    if (
        claims.get('purpose') != 'codex-auth'
        or claims.get('secret_name') != _SECRET_NAME
        or sandbox.created_by_user_id != user_id
        or sandbox.id != sandbox_id
        or scoped_conversation_id != conversation_id
    ):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, detail='Codex auth token scope mismatch'
        )
    scope = _CodexAuthScope(
        user_id=user_id,
        org_id=org_id,
        sandbox_id=sandbox_id,
        conversation_id=conversation_id,
        token_digest=token_digest,
        expires_at=expires_at,
    )
    if not allow_revoked:
        redis: Any = get_redis_client_async()
        try:
            revoked = await redis.get(scope.revocation_key)
        except redis_exceptions.RedisError as exc:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                detail='Codex credential authorization is unavailable',
            ) from exc
        if revoked is not None:
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                detail='Codex auth token has been revoked',
            )
    return scope


async def _revoke(scope: _CodexAuthScope) -> None:
    redis: Any = get_redis_client_async()
    try:
        await redis.set(
            scope.revocation_key,
            '1',
            ex=max(1, scope.expires_at - int(time.time())),
            nx=False,
        )
    except redis_exceptions.RedisError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail='Codex credential authorization is unavailable',
        ) from exc


@asynccontextmanager
async def _credential_lock(scope: _CodexAuthScope) -> AsyncIterator[None]:
    redis: Any = get_redis_client_async()
    try:
        async with redis.lock(
            scope.refresh_lock_key,
            timeout=_REFRESH_LOCK_TTL_SECONDS,
            blocking_timeout=_REFRESH_LOCK_WAIT_SECONDS,
            sleep=0.1,
            raise_on_release_error=False,
        ):
            yield
    except redis_exceptions.LockError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail='Codex credential refresh is busy',
        ) from exc
    except redis_exceptions.RedisError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail='Codex credential refresh is unavailable',
        ) from exc


async def _get_store(scope: _CodexAuthScope) -> CodexAuthStore:
    return await CodexAuthStore.get_instance(scope.user_id, scope.org_id)


async def _parse_update(request: Request) -> tuple[str, str]:
    body = await request.body()
    try:
        payload = json.loads(body)
        expected_digest = payload['expected_digest']
        value = payload['value']
    except (KeyError, TypeError, ValueError):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail='Invalid Codex credential update',
        ) from None
    try:
        value_size = len(value.encode()) if isinstance(value, str) else None
    except UnicodeError:
        value_size = None
    if not (
        isinstance(expected_digest, str)
        and _DIGEST_PATTERN.fullmatch(expected_digest) is not None
        and isinstance(value, str)
        and value_size is not None
        and value_size <= MAX_API_SECRET_VALUE_LENGTH
        and is_chatgpt_codex_auth(value)
    ):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail='Invalid Codex credential update',
        )
    return expected_digest, value


def _decode_refresh_authorization(value: str | None) -> tuple[str, str]:
    try:
        scheme, encoded = value.split(' ', 1) if value else ('', '')
        decoded = base64.b64decode(encoded, validate=True).decode()
        session_api_key, separator, codex_auth_token = decoded.partition(':')
    except (UnicodeError, ValueError):
        scheme = ''
        session_api_key = ''
        separator = ''
        codex_auth_token = ''
    if (
        scheme.lower() != 'basic'
        or not separator
        or not session_api_key
        or not codex_auth_token
    ):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail='Invalid Codex refresh authorization',
        )
    return session_api_key, codex_auth_token


async def _parse_refresh_request(request: Request) -> str:
    body = await request.body()
    try:
        payload = json.loads(body)
        client_id = payload['client_id']
        grant_type = payload['grant_type']
        refresh_token = payload['refresh_token']
    except (KeyError, TypeError, UnicodeError, ValueError):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail='Invalid Codex credential refresh',
        ) from None
    if (
        client_id != _REFRESH_CLIENT_ID
        or grant_type != 'refresh_token'
        or not isinstance(refresh_token, str)
        or not refresh_token
    ):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail='Invalid Codex credential refresh',
        )
    return refresh_token


def _token_payload(value: str) -> dict[str, str]:
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


def _merge_refresh(value: str, refresh: dict[str, Any]) -> str:
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


async def _request_token_refresh(refresh_token: str) -> httpx.Response:
    url = os.getenv('OPENHANDS_CODEX_REFRESH_TOKEN_URL', _REFRESH_TOKEN_URL)
    async with httpx.AsyncClient(timeout=30.0) as client:
        return await client.post(
            url,
            json={
                'client_id': _REFRESH_CLIENT_ID,
                'grant_type': 'refresh_token',
                'refresh_token': refresh_token,
            },
        )


def _refresh_error(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
        error = payload.get('error')
        code = error.get('code') if isinstance(error, dict) else None
    except (AttributeError, TypeError, ValueError):
        code = None
    return {'error': {'code': code or 'credential_refresh_failed'}}


@router.get(CODEX_AUTH_ROUTE, include_in_schema=False)
async def get_codex_auth(
    conversation_id: UUID,
    x_oh_sandbox: str | None = Header(None),
    x_oh_codex: str | None = Header(None),
    jwt_service: JwtService = jwt_service_dependency,
):
    scope = await _authorize(
        conversation_id,
        x_oh_sandbox,
        x_oh_codex,
        jwt_service,
        allow_paused_teardown=True,
    )
    store = await _get_store(scope)
    value = await store.get_value()
    if value is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, detail='Codex credentials were not found'
        )
    if not is_chatgpt_codex_auth(value):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail='Stored Codex authentication needs to be refreshed',
        )
    return Response(
        content=value,
        media_type='application/json',
        headers={'Cache-Control': 'no-store'},
    )


@router.head(CODEX_AUTH_ROUTE, include_in_schema=False)
async def touch_codex_auth(
    conversation_id: UUID,
    x_oh_sandbox: str | None = Header(None),
    x_oh_codex: str | None = Header(None),
    jwt_service: JwtService = jwt_service_dependency,
):
    scope = await _authorize(
        conversation_id,
        x_oh_sandbox,
        x_oh_codex,
        jwt_service,
        allow_paused_teardown=True,
    )
    store = await _get_store(scope)
    value = await store.get_value()
    if value is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, detail='Codex credentials were not found'
        )
    digest = hashlib.sha256(value.encode()).hexdigest()
    return Response(
        status_code=status.HTTP_204_NO_CONTENT,
        headers={'X-Codex-Auth-Digest': digest, 'Cache-Control': 'no-store'},
    )


@router.put(CODEX_AUTH_ROUTE, include_in_schema=False)
async def update_codex_auth(
    conversation_id: UUID,
    request: Request,
    x_oh_sandbox: str | None = Header(None),
    x_oh_codex: str | None = Header(None),
    jwt_service: JwtService = jwt_service_dependency,
):
    scope = await _authorize(
        conversation_id,
        x_oh_sandbox,
        x_oh_codex,
        jwt_service,
        allow_paused_teardown=True,
    )
    expected_digest, value = await _parse_update(request)
    store = await _get_store(scope)
    try:
        updated = await store.compare_and_swap(expected_digest, value)
    except KeyError as exc:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, detail='Codex credentials were not found'
        ) from exc
    if not updated:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail='Codex credentials changed in another session.',
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(f'{CODEX_AUTH_ROUTE}/refresh', include_in_schema=False)
async def refresh_codex_auth(
    conversation_id: UUID,
    request: Request,
    authorization: str | None = Header(None),
    jwt_service: JwtService = jwt_service_dependency,
):
    session_api_key, codex_auth_token = _decode_refresh_authorization(authorization)
    scope = await _authorize(
        conversation_id, session_api_key, codex_auth_token, jwt_service
    )
    submitted_refresh_token = await _parse_refresh_request(request)
    async with _credential_lock(scope):
        store = await _get_store(scope)
        current = await store.get_value()
        if current is None:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, detail='Codex credentials were not found'
            )
        current_tokens = _token_payload(current)
        current_refresh_token = current_tokens['refresh_token']
        if not hmac.compare_digest(
            submitted_refresh_token.encode(errors='surrogatepass'),
            current_refresh_token.encode(errors='surrogatepass'),
        ):
            return JSONResponse(
                current_tokens,
                headers={'Cache-Control': 'no-store'},
            )
        try:
            response = await _request_token_refresh(current_refresh_token)
        except httpx.HTTPError as exc:
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY,
                detail='Codex credential refresh is unavailable',
            ) from exc
        if not response.is_success:
            response_status = (
                response.status_code
                if 400 <= response.status_code < 500
                else status.HTTP_502_BAD_GATEWAY
            )
            return JSONResponse(
                _refresh_error(response),
                status_code=response_status,
                headers={'Cache-Control': 'no-store'},
            )
        try:
            refresh = response.json()
        except ValueError as exc:
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY,
                detail='Codex credential refresh returned an invalid response',
            ) from exc
        if not isinstance(refresh, dict):
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY,
                detail='Codex credential refresh returned an invalid response',
            )
        updated_value = _merge_refresh(current, refresh)
        current_digest = hashlib.sha256(current.encode()).hexdigest()
        try:
            updated = await store.compare_and_swap(current_digest, updated_value)
        except KeyError as exc:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND, detail='Codex credentials were not found'
            ) from exc
        if not updated:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                detail='Codex credentials changed during refresh',
            )
        return JSONResponse(
            _token_payload(updated_value),
            headers={'Cache-Control': 'no-store'},
        )


@router.delete(CODEX_AUTH_ROUTE, include_in_schema=False)
async def release_codex_auth(
    conversation_id: UUID,
    x_oh_sandbox: str | None = Header(None),
    x_oh_codex: str | None = Header(None),
    jwt_service: JwtService = jwt_service_dependency,
):
    scope = await _authorize(
        conversation_id,
        x_oh_sandbox,
        x_oh_codex,
        jwt_service,
        allow_revoked=True,
        allow_paused_teardown=True,
    )
    await _revoke(scope)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
