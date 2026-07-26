from __future__ import annotations

import os
from typing import Literal
from uuid import UUID

import httpx
from pydantic import BaseModel, ConfigDict, Field

from openhands.app_server.constants import MAX_API_SECRET_VALUE_LENGTH
from openhands.app_server.errors import SandboxError
from openhands.app_server.services.jwt_service import JwtService
from openhands.sdk.agent.acp_file_credentials import (
    CODEX_AUTH_SECRET_NAME,
    is_valid_codex_auth,
)

CREDENTIAL_BINDING_CONTEXT_HEADER = 'X-Credential-Binding-Context'
CREDENTIAL_BINDING_CAPABILITIES = frozenset(
    {
        'credential_binding_v1',
        'credential_binding_readiness_probe_v1',
        'credential_binding_activation_guard_v1',
    }
)


class CredentialBindingContext(BaseModel):
    model_config = ConfigDict(extra='forbid')

    purpose: Literal['codex-credential-binding']
    iat: int
    user_id: str
    organization_id: str | None
    sandbox_id: str
    conversation_id: str
    start_task_id: str | None
    secret_name: Literal['CODEX_AUTH_JSON']
    actions: tuple[Literal['load'], Literal['replace']]


class CredentialReplacement(BaseModel):
    model_config = ConfigDict(extra='forbid')

    expected_version: str = Field(min_length=1, max_length=256)
    value: str = Field(max_length=MAX_API_SECRET_VALUE_LENGTH)


def codex_credential_sync_enabled() -> bool:
    return os.getenv('OH_ENABLE_CODEX_CREDENTIAL_SYNC', '').lower() in {
        '1',
        'true',
        'yes',
    }


def managed_credential_marker(organization_id: UUID | None) -> str:
    return f'org:{organization_id}' if organization_id else 'personal'


def marker_organization_id(marker: str | None) -> UUID | None:
    if marker == 'personal':
        return None
    if marker and marker.startswith('org:'):
        return UUID(marker[4:])
    raise ValueError('Invalid managed credential marker')


def create_binding_context(
    jwt_service: JwtService,
    *,
    user_id: str,
    organization_id: UUID | None,
    sandbox_id: str,
    conversation_id: UUID,
    start_task_id: UUID | None,
) -> str:
    return jwt_service.create_jwe_token(
        {
            'purpose': 'codex-credential-binding',
            'user_id': user_id,
            'organization_id': (
                str(organization_id) if organization_id is not None else None
            ),
            'sandbox_id': sandbox_id,
            'conversation_id': str(conversation_id),
            'start_task_id': str(start_task_id) if start_task_id else None,
            'secret_name': CODEX_AUTH_SECRET_NAME,
            'actions': ['load', 'replace'],
        }
    )


def decode_binding_context(
    jwt_service: JwtService, token: str
) -> CredentialBindingContext:
    return CredentialBindingContext.model_validate(jwt_service.decrypt_jwe_token(token))


def credential_binding_callback_path(sandbox_id: str, conversation_id: UUID) -> str:
    return (
        f'/api/v1/sandboxes/{sandbox_id}/credential-bindings/'
        f'{conversation_id}/{CODEX_AUTH_SECRET_NAME}'
    )


async def agent_server_supports_credential_binding(
    httpx_client: httpx.AsyncClient,
    agent_server_url: str,
    session_api_key: str,
) -> bool:
    response = await httpx_client.get(
        f'{agent_server_url}/server_info',
        headers={'X-Session-API-Key': session_api_key},
        timeout=30.0,
    )
    response.raise_for_status()
    body = response.json()
    capabilities = body.get('capabilities') if isinstance(body, dict) else None
    if capabilities is None:
        return False
    if not isinstance(capabilities, list) or not all(
        isinstance(capability, str) for capability in capabilities
    ):
        raise SandboxError('Agent Server returned invalid capability metadata')
    return CREDENTIAL_BINDING_CAPABILITIES.issubset(capabilities)


async def activate_codex_credential_binding(
    httpx_client: httpx.AsyncClient,
    jwt_service: JwtService,
    *,
    agent_server_url: str,
    callback_url: str,
    session_api_key: str,
    user_id: str,
    organization_id: UUID | None,
    sandbox_id: str,
    conversation_id: UUID,
    start_task_id: UUID | None,
    required: bool,
) -> bool:
    try:
        supported = await agent_server_supports_credential_binding(
            httpx_client, agent_server_url, session_api_key
        )
    except httpx.HTTPStatusError as exc:
        if not required and exc.response.status_code in (404, 501):
            return False
        raise SandboxError(
            'Could not verify Agent Server credential binding support'
        ) from exc
    except Exception as exc:
        raise SandboxError(
            'Could not verify Agent Server credential binding support'
        ) from exc
    if not supported:
        if required:
            raise SandboxError('Agent Server lacks managed credential support')
        return False

    context = create_binding_context(
        jwt_service,
        user_id=user_id,
        organization_id=organization_id,
        sandbox_id=sandbox_id,
        conversation_id=conversation_id,
        start_task_id=start_task_id,
    )
    response = await httpx_client.put(
        (
            f'{agent_server_url}/api/conversations/{conversation_id}/'
            f'credential-bindings/{CODEX_AUTH_SECRET_NAME}'
        ),
        json={
            'url': callback_url,
            'headers': {
                'X-Session-API-Key': session_api_key,
                CREDENTIAL_BINDING_CONTEXT_HEADER: context,
            },
        },
        headers={'X-Session-API-Key': session_api_key},
        timeout=30.0,
    )
    if response.status_code == 501 and not required:
        return False
    try:
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise SandboxError(
            'Agent Server rejected managed credential activation'
        ) from exc
    if response.status_code != 204:
        raise SandboxError('Agent Server returned an invalid activation response')
    return True


def validate_codex_credential(value: str) -> None:
    if len(value.encode()) > MAX_API_SECRET_VALUE_LENGTH or not is_valid_codex_auth(
        value
    ):
        raise ValueError('Invalid Codex credential')
