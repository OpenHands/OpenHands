from __future__ import annotations

import time
from dataclasses import dataclass
from uuid import UUID

import jwt
from fastapi import APIRouter, Header, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from openhands.app_server.config import (
    depends_jwt_service,
    get_app_conversation_info_service,
    get_sandbox_service,
)
from openhands.app_server.constants import MAX_API_SECRET_VALUE_LENGTH
from openhands.app_server.sandbox.sandbox_models import SandboxStatus
from openhands.app_server.secrets.credential_binding_models import (
    CREDENTIAL_BINDING_ROUTE,
    CREDENTIAL_BINDING_ROUTE_PREFIX,
    is_runtime_managed_credential,
    is_valid_runtime_managed_credential,
)
from openhands.app_server.secrets.secrets_store import (
    CredentialVersionConflict,
    SecretsStore,
)
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.services.jwt_service import JwtService
from openhands.app_server.user.specifiy_user_context import ADMIN, USER_CONTEXT_ATTR
from openhands.app_server.user_auth.user_auth import get_for_user

router = APIRouter(prefix=CREDENTIAL_BINDING_ROUTE_PREFIX)
jwt_service_dependency = depends_jwt_service()
_CONVERSATION_START_GRACE_SECONDS = 300


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
    issued_at: int


def _authorize(
    authorization: str | None,
    jwt_service: JwtService,
    conversation_id: UUID,
    secret_name: str,
    action: str,
) -> CredentialBindingScope:
    scheme, _, token = (authorization or '').partition(' ')
    if scheme.lower() != 'bearer' or not token:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail='Credential binding token is required',
        )
    try:
        claims = jwt_service.verify_jws_token(token)
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
        issued_at = claims['iat']
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
        or not isinstance(issued_at, int)
        or isinstance(issued_at, bool)
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
        issued_at=issued_at,
    )


async def _store(scope: CredentialBindingScope) -> SecretsStore:
    user_auth = await get_for_user(scope.user_id)
    return await user_auth.get_secrets_store()


async def _validate_active_binding(scope: CredentialBindingScope) -> None:
    state = InjectorState()
    setattr(state, USER_CONTEXT_ATTR, ADMIN)
    async with (
        get_app_conversation_info_service(state) as conversation_service,
        get_sandbox_service(state) as sandbox_service,
    ):
        conversation = await conversation_service.get_app_conversation_info(
            scope.conversation_id
        )
        sandbox = await sandbox_service.get_sandbox(scope.runtime_id)
    within_start_grace = (
        time.time() - scope.issued_at <= _CONVERSATION_START_GRACE_SECONDS
    )
    if (
        sandbox is None
        or sandbox.status != SandboxStatus.RUNNING
        or sandbox.created_by_user_id not in (None, scope.user_id)
        or (conversation is None and not within_start_grace)
        or (
            conversation is not None
            and conversation.created_by_user_id not in (None, scope.user_id)
        )
        or (
            conversation is not None
            and conversation.sandbox_id != scope.runtime_id
            and not within_start_grace
        )
    ):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail='Credential binding is no longer active',
        )


def _validate_value(secret_name: str, value: str) -> None:
    try:
        size = len(value.encode())
    except UnicodeError:
        size = MAX_API_SECRET_VALUE_LENGTH + 1
    if size > MAX_API_SECRET_VALUE_LENGTH or (
        not is_valid_runtime_managed_credential(secret_name, value)
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
    if not is_runtime_managed_credential(secret_name):
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    scope = _authorize(
        authorization,
        jwt_service,
        conversation_id,
        secret_name,
        'load',
    )
    await _validate_active_binding(scope)
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
    if not is_runtime_managed_credential(secret_name):
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    scope = _authorize(
        authorization,
        jwt_service,
        conversation_id,
        secret_name,
        'replace',
    )
    await _validate_active_binding(scope)
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
