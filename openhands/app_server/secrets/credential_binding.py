from __future__ import annotations

from dataclasses import dataclass
from typing import TypeGuard
from uuid import UUID

import jwt
from fastapi import APIRouter, Header, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationInfo,
    AppConversationStartTask,
    AppConversationStartTaskStatus,
)
from openhands.app_server.config import (
    depends_jwt_service,
    get_app_conversation_info_service,
    get_app_conversation_start_task_service,
    get_sandbox_service,
)
from openhands.app_server.constants import MAX_API_SECRET_VALUE_LENGTH
from openhands.app_server.sandbox.sandbox_models import SandboxInfo, SandboxStatus
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


class CredentialReplacement(BaseModel):
    model_config = ConfigDict(extra='forbid')

    expected_version: str = Field(min_length=1, max_length=256)
    value: str = Field(max_length=MAX_API_SECRET_VALUE_LENGTH)


class CredentialBindingClaims(BaseModel):
    model_config = ConfigDict(extra='ignore', strict=True)

    purpose: str
    user_id: str
    organization_id: UUID | None
    conversation_id: UUID
    runtime_id: str
    start_task_id: UUID
    secret_name: str
    actions: list[str]

    @field_validator('organization_id', mode='before')
    @classmethod
    def parse_optional_uuid_claim(cls, value: object) -> UUID | None:
        if value is None or value == '':
            return None
        if not isinstance(value, str):
            raise ValueError
        return UUID(value)

    @field_validator('conversation_id', 'start_task_id', mode='before')
    @classmethod
    def parse_uuid_claim(cls, value: object) -> UUID:
        if not isinstance(value, str):
            raise ValueError
        return UUID(value)


@dataclass(frozen=True)
class CredentialBindingScope:
    user_id: str
    organization_id: UUID | None
    conversation_id: UUID
    runtime_id: str
    start_task_id: UUID
    secret_name: str


def _claims_match_scope(
    claims: CredentialBindingClaims,
    conversation_id: UUID,
    secret_name: str,
    action: str,
) -> bool:
    return (
        claims.purpose == 'credential-binding'
        and bool(claims.user_id)
        and bool(claims.runtime_id)
        and claims.conversation_id == conversation_id
        and claims.secret_name == secret_name
        and len(claims.actions) == 2
        and set(claims.actions) == {'load', 'replace'}
        and action in claims.actions
    )


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
        claims = CredentialBindingClaims.model_validate(
            jwt_service.verify_jws_token(token)
        )
    except (TypeError, ValueError, jwt.InvalidTokenError) as exc:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail='Invalid credential binding token',
        ) from exc
    if not _claims_match_scope(claims, conversation_id, secret_name, action):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail='Credential binding token scope mismatch',
        )
    return CredentialBindingScope(
        user_id=claims.user_id,
        organization_id=claims.organization_id,
        conversation_id=conversation_id,
        runtime_id=claims.runtime_id,
        start_task_id=claims.start_task_id,
        secret_name=secret_name,
    )


async def _store(scope: CredentialBindingScope) -> SecretsStore:
    user_auth = await get_for_user(scope.user_id)
    return await user_auth.get_secrets_store()


def _is_current_start_task(
    task: AppConversationStartTask | None,
    scope: CredentialBindingScope,
) -> TypeGuard[AppConversationStartTask]:
    return (
        task is not None
        and task.id == scope.start_task_id
        and task.app_conversation_id == scope.conversation_id
        and task.sandbox_id == scope.runtime_id
        and task.created_by_user_id in (None, scope.user_id)
        and task.status
        in (
            AppConversationStartTaskStatus.STARTING_CONVERSATION,
            AppConversationStartTaskStatus.READY,
        )
    )


def _is_live_owned_runtime(
    sandbox: SandboxInfo | None,
    scope: CredentialBindingScope,
) -> bool:
    return (
        sandbox is not None
        and sandbox.status == SandboxStatus.RUNNING
        and sandbox.created_by_user_id in (None, scope.user_id)
    )


def _is_valid_conversation_mapping(
    conversation: AppConversationInfo | None,
    task_status: AppConversationStartTaskStatus,
    scope: CredentialBindingScope,
) -> bool:
    if conversation is not None and conversation.created_by_user_id not in (
        None,
        scope.user_id,
    ):
        return False
    return task_status != AppConversationStartTaskStatus.READY or (
        conversation is not None and conversation.sandbox_id == scope.runtime_id
    )


async def _validate_active_binding(scope: CredentialBindingScope) -> None:
    state = InjectorState()
    setattr(state, USER_CONTEXT_ATTR, ADMIN)
    async with (
        get_app_conversation_info_service(state) as conversation_service,
        get_app_conversation_start_task_service(state) as start_task_service,
        get_sandbox_service(state) as sandbox_service,
    ):
        task = None
        page_id = None
        while task is None:
            task_page = await start_task_service.search_app_conversation_start_tasks(
                conversation_id__eq=scope.conversation_id,
                page_id=page_id,
                limit=100,
            )
            task = next(
                (
                    candidate
                    for candidate in task_page.items
                    if candidate.status
                    in (
                        AppConversationStartTaskStatus.STARTING_CONVERSATION,
                        AppConversationStartTaskStatus.READY,
                    )
                ),
                None,
            )
            page_id = task_page.next_page_id
            if page_id is None:
                break
        if not _is_current_start_task(task, scope):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                detail='Credential binding is no longer active',
            )
        conversation = await conversation_service.get_app_conversation_info(
            scope.conversation_id
        )
        sandbox = await sandbox_service.get_sandbox_for_authorization(scope.runtime_id)
    if not _is_live_owned_runtime(sandbox, scope) or not _is_valid_conversation_mapping(
        conversation, task.status, scope
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
            status.HTTP_501_NOT_IMPLEMENTED,
            detail='Credential binding is unsupported',
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
            status.HTTP_501_NOT_IMPLEMENTED,
            detail='Credential binding is unsupported',
        ) from exc
    return JSONResponse(
        {'version': version},
        headers={'Cache-Control': 'no-store'},
    )
