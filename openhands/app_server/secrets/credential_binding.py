from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from uuid import UUID

import jwt
from fastapi import APIRouter, Header, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from openhands.app_server.config import depends_jwt_service
from openhands.app_server.constants import MAX_API_SECRET_VALUE_LENGTH
from openhands.app_server.sandbox.session_auth import validate_session_key
from openhands.app_server.secrets.credential_binding_models import (
    CODEX_AUTH_SECRET_NAME,
    CREDENTIAL_BINDING_RENEWAL_ROUTE,
    CREDENTIAL_BINDING_ROUTE,
    CREDENTIAL_BINDING_ROUTE_PREFIX,
    MAX_CREDENTIAL_BINDING_TOKEN_TIMEOUT_SECONDS,
    is_valid_codex_auth,
)
from openhands.app_server.secrets.secrets_store import CredentialVersionConflict
from openhands.app_server.services.jwt_service import JwtService
from openhands.app_server.user_auth.user_auth import get_for_user

router = APIRouter(prefix=CREDENTIAL_BINDING_ROUTE_PREFIX)
jwt_service_dependency = depends_jwt_service()


class CredentialReplacement(BaseModel):
    model_config = ConfigDict(extra='forbid')

    expected_version: str = Field(min_length=1, max_length=256)
    value: str = Field(max_length=MAX_API_SECRET_VALUE_LENGTH)


@dataclass(frozen=True)
class CredentialBindingScope:
    user_id: str
    organization_id: UUID | None
    conversation_id: UUID
    runtime_id: str
    secret_name: str
    renewal_ttl_seconds: int | None


def _authorize(
    authorization: str | None,
    jwt_service: JwtService,
    conversation_id: UUID,
    secret_name: str,
    action: str,
    *,
    allow_expired: bool = False,
) -> CredentialBindingScope:
    scheme, _, token = (authorization or '').partition(' ')
    if scheme.lower() != 'bearer' or not token:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail='Credential binding token is required',
        )
    try:
        claims = jwt_service.verify_jws_token(
            token,
            verify_expiration=not allow_expired,
        )
        user_id = claims['user_id']
        organization_claim = claims['organization_id']
        if organization_claim is not None and not isinstance(organization_claim, str):
            raise TypeError
        organization_id = UUID(organization_claim) if organization_claim else None
        conversation_claim = claims['conversation_id']
        if not isinstance(conversation_claim, str):
            raise TypeError
        scoped_conversation_id = UUID(conversation_claim)
        runtime_id = claims['runtime_id']
        scoped_secret_name = claims['secret_name']
        actions = claims['actions']
        renewal_ttl_seconds = claims.get('renewal_ttl_seconds')
    except (KeyError, TypeError, ValueError, jwt.InvalidTokenError) as exc:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail='Invalid credential binding token',
        ) from exc
    if (
        claims.get('purpose') != 'credential-binding'
        or not isinstance(user_id, str)
        or not user_id
        or not isinstance(runtime_id, str)
        or not runtime_id
        or scoped_conversation_id != conversation_id
        or not isinstance(scoped_secret_name, str)
        or scoped_secret_name != secret_name
        or not isinstance(actions, list)
        or len(actions) != 2
        or not all(isinstance(candidate, str) for candidate in actions)
        or set(actions) != {'load', 'replace'}
        or action not in actions
        or (
            renewal_ttl_seconds is not None
            and (
                isinstance(renewal_ttl_seconds, bool)
                or not isinstance(renewal_ttl_seconds, int)
                or renewal_ttl_seconds <= 0
                or renewal_ttl_seconds > MAX_CREDENTIAL_BINDING_TOKEN_TIMEOUT_SECONDS
            )
        )
    ):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail='Credential binding token scope mismatch',
        )
    return CredentialBindingScope(
        user_id=user_id,
        organization_id=organization_id,
        conversation_id=conversation_id,
        runtime_id=runtime_id,
        secret_name=secret_name,
        renewal_ttl_seconds=renewal_ttl_seconds,
    )


def _token_payload(scope: CredentialBindingScope) -> dict[str, object]:
    assert scope.renewal_ttl_seconds is not None
    return {
        'purpose': 'credential-binding',
        'user_id': scope.user_id,
        'organization_id': (
            str(scope.organization_id) if scope.organization_id is not None else None
        ),
        'conversation_id': str(scope.conversation_id),
        'runtime_id': scope.runtime_id,
        'secret_name': scope.secret_name,
        'actions': ['load', 'replace'],
        'renewal_ttl_seconds': scope.renewal_ttl_seconds,
    }


async def _store(scope: CredentialBindingScope):
    user_auth = await get_for_user(scope.user_id)
    return await user_auth.get_secrets_store()


def _validate_value(secret_name: str, value: str) -> None:
    try:
        size = len(value.encode())
    except UnicodeError:
        size = MAX_API_SECRET_VALUE_LENGTH + 1
    if size > MAX_API_SECRET_VALUE_LENGTH or (
        secret_name == CODEX_AUTH_SECRET_NAME and not is_valid_codex_auth(value)
    ):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail='Credential value is invalid',
        )


@router.get(CREDENTIAL_BINDING_ROUTE, include_in_schema=False)
async def load_credential(
    conversation_id: UUID,
    secret_name: str,
    authorization: str | None = Header(None),
    jwt_service: JwtService = jwt_service_dependency,
):
    if secret_name != CODEX_AUTH_SECRET_NAME:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    scope = _authorize(
        authorization,
        jwt_service,
        conversation_id,
        secret_name,
        'load',
    )
    store = await _store(scope)
    try:
        value, version = await store.load_versioned(
            secret_name,
            scope.organization_id,
        )
    except KeyError as exc:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail='Credential was not found',
        ) from exc
    except NotImplementedError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail='Credential binding is unavailable',
        ) from exc
    _validate_value(secret_name, value)
    return JSONResponse(
        {'value': value, 'version': version},
        headers={'Cache-Control': 'no-store'},
    )


@router.put(CREDENTIAL_BINDING_ROUTE, include_in_schema=False)
async def replace_credential(
    conversation_id: UUID,
    secret_name: str,
    replacement: CredentialReplacement,
    authorization: str | None = Header(None),
    jwt_service: JwtService = jwt_service_dependency,
):
    if secret_name != CODEX_AUTH_SECRET_NAME:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    scope = _authorize(
        authorization,
        jwt_service,
        conversation_id,
        secret_name,
        'replace',
    )
    _validate_value(secret_name, replacement.value)
    store = await _store(scope)
    try:
        version = await store.replace_versioned(
            secret_name,
            replacement.expected_version,
            replacement.value,
            scope.organization_id,
        )
    except KeyError as exc:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail='Credential was not found',
        ) from exc
    except CredentialVersionConflict as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail='Credential changed in another runtime',
        ) from exc
    except NotImplementedError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail='Credential binding is unavailable',
        ) from exc
    return JSONResponse(
        {'version': version},
        headers={'Cache-Control': 'no-store'},
    )


@router.post(CREDENTIAL_BINDING_RENEWAL_ROUTE, include_in_schema=False)
async def renew_credential_binding(
    conversation_id: UUID,
    secret_name: str,
    authorization: str | None = Header(None),
    x_session_api_key: str | None = Header(None),
    jwt_service: JwtService = jwt_service_dependency,
):
    if secret_name != CODEX_AUTH_SECRET_NAME:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    scope = _authorize(
        authorization,
        jwt_service,
        conversation_id,
        secret_name,
        'load',
        allow_expired=True,
    )
    if scope.renewal_ttl_seconds is None:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail='Credential binding token cannot be renewed',
        )
    sandbox = await validate_session_key(x_session_api_key)
    if sandbox.id != scope.runtime_id or (
        sandbox.created_by_user_id is not None
        and sandbox.created_by_user_id != scope.user_id
    ):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail='Credential binding runtime scope mismatch',
        )
    token = jwt_service.create_jws_token(
        _token_payload(scope),
        expires_in=timedelta(seconds=scope.renewal_ttl_seconds),
    )
    return JSONResponse(
        {
            'authorization': f'Bearer {token}',
            'authorization_expires_in_seconds': scope.renewal_ttl_seconds,
        },
        headers={'Cache-Control': 'no-store'},
    )
