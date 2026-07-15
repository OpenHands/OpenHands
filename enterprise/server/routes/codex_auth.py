import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import jwt
from fastapi import APIRouter, Header, HTTPException, Request, Response, status
from storage.redis import get_redis_client_async, redis_exceptions
from storage.saas_secrets_store import SaasSecretsStore

from openhands.app_server.config import depends_jwt_service, get_sandbox_service
from openhands.app_server.constants import MAX_API_SECRET_VALUE_LENGTH
from openhands.app_server.sandbox.sandbox_models import SandboxStatus
from openhands.app_server.sandbox.session_auth import validate_session_key
from openhands.app_server.secrets.codex_auth import is_chatgpt_codex_auth
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.services.jwt_service import JwtService
from openhands.app_server.user.specifiy_user_context import ADMIN, USER_CONTEXT_ATTR

router = APIRouter(prefix='/api/internal/conversations')
jwt_service_dependency = depends_jwt_service()

_SECRET_NAME = 'CODEX_AUTH_JSON'
_LEASE_TTL_SECONDS = 14 * 86400
_DIGEST_PATTERN = re.compile(r'^[0-9a-f]{64}$')
_MAX_REQUEST_BYTES = MAX_API_SECRET_VALUE_LENGTH + 4096
_REPLACE_OWNER_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    redis.call('set', KEYS[1], ARGV[2], 'EX', ARGV[3])
    return 1
end
return 0
"""
_TOUCH_OWNER_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('expire', KEYS[1], ARGV[2])
end
return 0
"""
_RELEASE_OWNER_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
end
return 0
"""


@dataclass(frozen=True)
class _CodexAuthScope:
    user_id: str
    org_id: UUID
    sandbox_id: str
    conversation_id: UUID

    @property
    def owner(self) -> str:
        return json.dumps(
            {
                'conversation_id': str(self.conversation_id),
                'sandbox_id': self.sandbox_id,
            },
            separators=(',', ':'),
            sort_keys=True,
        )

    @property
    def lease_key(self) -> str:
        user_digest = hashlib.sha256(self.user_id.encode()).hexdigest()
        return f'codex-auth-owner:{user_digest}'


async def _authorize(
    conversation_id: UUID,
    session_api_key: str | None,
    codex_auth_token: str | None,
    jwt_service: JwtService,
) -> _CodexAuthScope:
    if not codex_auth_token:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail='X-Codex-Auth-Token header is required',
        )
    sandbox = await validate_session_key(session_api_key)
    try:
        claims = jwt_service.verify_jws_token(codex_auth_token)
        org_id = UUID(str(claims['org_id']))
        user_id = str(claims['user_id'])
        sandbox_id = str(claims['sandbox_id'])
        scoped_conversation_id = UUID(str(claims['conversation_id']))
    except (KeyError, TypeError, ValueError, jwt.InvalidTokenError) as exc:
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
    return _CodexAuthScope(
        user_id=user_id,
        org_id=org_id,
        sandbox_id=sandbox_id,
        conversation_id=conversation_id,
    )


async def _sandbox_is_running(sandbox_id: str) -> bool:
    state = InjectorState()
    setattr(state, USER_CONTEXT_ATTR, ADMIN)
    async with get_sandbox_service(state) as sandbox_service:
        sandbox = await sandbox_service.get_sandbox(sandbox_id)
    return sandbox is not None and sandbox.status == SandboxStatus.RUNNING


async def _replace_owner(redis, key: str, current: str, owner: str) -> bool:
    result = await redis.eval(
        _REPLACE_OWNER_SCRIPT,
        1,
        key,
        current,
        owner,
        str(_LEASE_TTL_SECONDS),
    )
    return bool(result)


async def _acquire_lease(scope: _CodexAuthScope) -> None:
    redis: Any = get_redis_client_async()
    try:
        for _ in range(3):
            acquired = await redis.set(
                scope.lease_key,
                scope.owner,
                ex=_LEASE_TTL_SECONDS,
                nx=True,
            )
            if acquired:
                return
            raw_owner = await redis.get(scope.lease_key)
            if raw_owner is None:
                continue
            current = raw_owner.decode() if isinstance(raw_owner, bytes) else raw_owner
            existing = json.loads(current)
            existing_conversation_id = UUID(existing['conversation_id'])
            existing_sandbox_id = str(existing['sandbox_id'])
            if existing_conversation_id != scope.conversation_id and (
                await _sandbox_is_running(existing_sandbox_id)
            ):
                raise HTTPException(
                    status.HTTP_409_CONFLICT,
                    detail=(
                        'ChatGPT subscription credentials are already in use by '
                        'another Codex conversation. Stop it and try again.'
                    ),
                )
            if await _replace_owner(redis, scope.lease_key, current, scope.owner):
                return
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail='Codex credential ownership changed; try again.',
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail='Codex credential ownership is unavailable',
        ) from exc


async def _require_lease(scope: _CodexAuthScope) -> None:
    redis: Any = get_redis_client_async()
    try:
        touched = await redis.eval(
            _TOUCH_OWNER_SCRIPT,
            1,
            scope.lease_key,
            scope.owner,
            str(_LEASE_TTL_SECONDS),
        )
    except redis_exceptions.RedisError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail='Codex credential ownership is unavailable',
        ) from exc
    if not touched:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail='Codex credential ownership expired or moved to another session.',
        )


async def _release_lease(scope: _CodexAuthScope) -> None:
    redis: Any = get_redis_client_async()
    try:
        await redis.eval(_RELEASE_OWNER_SCRIPT, 1, scope.lease_key, scope.owner)
    except redis_exceptions.RedisError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail='Codex credential ownership is unavailable',
        ) from exc


async def _get_store(scope: _CodexAuthScope) -> SaasSecretsStore:
    return await SaasSecretsStore.get_instance(
        scope.user_id, effective_org_id=scope.org_id
    )


async def _parse_update(request: Request) -> tuple[str, str]:
    body = await request.body()
    if len(body) > _MAX_REQUEST_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail='Codex credential update is too large',
        )
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


@router.get('/{conversation_id}/codex-auth', include_in_schema=False)
async def get_codex_auth(
    conversation_id: UUID,
    x_session_api_key: str | None = Header(None),
    x_codex_auth_token: str | None = Header(None),
    jwt_service: JwtService = jwt_service_dependency,
):
    scope = await _authorize(
        conversation_id, x_session_api_key, x_codex_auth_token, jwt_service
    )
    store = await _get_store(scope)
    value = await store.get_custom_secret_value(_SECRET_NAME)
    if value is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, detail='Codex credentials were not found'
        )
    if not is_chatgpt_codex_auth(value):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail='Stored Codex authentication needs to be refreshed',
        )
    await _acquire_lease(scope)
    return Response(
        content=value,
        media_type='application/json',
        headers={'Cache-Control': 'no-store'},
    )


@router.head('/{conversation_id}/codex-auth', include_in_schema=False)
async def touch_codex_auth(
    conversation_id: UUID,
    x_session_api_key: str | None = Header(None),
    x_codex_auth_token: str | None = Header(None),
    jwt_service: JwtService = jwt_service_dependency,
):
    scope = await _authorize(
        conversation_id, x_session_api_key, x_codex_auth_token, jwt_service
    )
    await _require_lease(scope)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.put('/{conversation_id}/codex-auth', include_in_schema=False)
async def update_codex_auth(
    conversation_id: UUID,
    request: Request,
    x_session_api_key: str | None = Header(None),
    x_codex_auth_token: str | None = Header(None),
    jwt_service: JwtService = jwt_service_dependency,
):
    scope = await _authorize(
        conversation_id, x_session_api_key, x_codex_auth_token, jwt_service
    )
    expected_digest, value = await _parse_update(request)
    await _require_lease(scope)
    store = await _get_store(scope)
    try:
        updated = await store.compare_and_swap_custom_secret(
            _SECRET_NAME, expected_digest, value
        )
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


@router.delete('/{conversation_id}/codex-auth', include_in_schema=False)
async def release_codex_auth(
    conversation_id: UUID,
    x_session_api_key: str | None = Header(None),
    x_codex_auth_token: str | None = Header(None),
    jwt_service: JwtService = jwt_service_dependency,
):
    scope = await _authorize(
        conversation_id, x_session_api_key, x_codex_auth_token, jwt_service
    )
    await _release_lease(scope)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
