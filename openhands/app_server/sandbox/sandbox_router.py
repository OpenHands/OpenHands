"""Runtime Containers router for OpenHands App Server."""

import logging
from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.security import APIKeyHeader

from openhands.agent_server.models import Success
from openhands.app_server.config import depends_sandbox_service
from openhands.app_server.sandbox.sandbox_models import (
    SandboxInfo,
    SandboxPage,
    SecretNameItem,
    SecretNamesResponse,
)
from openhands.app_server.sandbox.sandbox_service import (
    SandboxService,
)
from openhands.app_server.sandbox.session_auth import validate_session_key
from openhands.app_server.user.auth_user_context import AuthUserContext
from openhands.server.dependencies import get_dependencies
from openhands.server.user_auth.user_auth import (
    get_for_user as get_user_auth_for_user,
)

_logger = logging.getLogger(__name__)

# We use the get_dependencies method here to signal to the OpenAPI docs that this endpoint
# is protected. The actual protection is provided by SetAuthCookieMiddleware
router = APIRouter(
    prefix='/sandboxes', tags=['Sandbox'], dependencies=get_dependencies()
)
sandbox_service_dependency = depends_sandbox_service()

# Read methods


@router.get('/search')
async def search_sandboxes(
    page_id: Annotated[
        str | None,
        Query(title='Optional next_page_id from the previously returned page'),
    ] = None,
    limit: Annotated[
        int,
        Query(title='The max number of results in the page', gt=0, lte=100),
    ] = 100,
    sandbox_service: SandboxService = sandbox_service_dependency,
) -> SandboxPage:
    """Search / list sandboxes owned by the current user."""
    assert limit > 0
    assert limit <= 100
    return await sandbox_service.search_sandboxes(page_id=page_id, limit=limit)


@router.get('')
async def batch_get_sandboxes(
    id: Annotated[list[str], Query()],
    sandbox_service: SandboxService = sandbox_service_dependency,
) -> list[SandboxInfo | None]:
    """Get a batch of sandboxes given their ids, returning null for any missing."""
    assert len(id) < 100
    sandboxes = await sandbox_service.batch_get_sandboxes(id)
    return sandboxes


# Write Methods


@router.post('')
async def start_sandbox(
    sandbox_spec_id: str | None = None,
    sandbox_service: SandboxService = sandbox_service_dependency,
) -> SandboxInfo:
    info = await sandbox_service.start_sandbox(sandbox_spec_id)
    return info


@router.post('/{sandbox_id}/pause', responses={404: {'description': 'Item not found'}})
async def pause_sandbox(
    sandbox_id: str,
    sandbox_service: SandboxService = sandbox_service_dependency,
) -> Success:
    exists = await sandbox_service.pause_sandbox(sandbox_id)
    if not exists:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    return Success()


@router.post('/{sandbox_id}/resume', responses={404: {'description': 'Item not found'}})
async def resume_sandbox(
    sandbox_id: str,
    sandbox_service: SandboxService = sandbox_service_dependency,
) -> Success:
    exists = await sandbox_service.resume_sandbox(sandbox_id)
    if not exists:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    return Success()


@router.delete('/{id}', responses={404: {'description': 'Item not found'}})
async def delete_sandbox(
    sandbox_id: str,
    sandbox_service: SandboxService = sandbox_service_dependency,
) -> Success:
    exists = await sandbox_service.delete_sandbox(sandbox_id)
    if not exists:
        raise HTTPException(status.HTTP_404_NOT_FOUND)
    return Success()


# ---------------------------------------------------------------------------
# Sandbox-scoped secrets (authenticated via X-Session-API-Key)
# ---------------------------------------------------------------------------


async def _valid_sandbox_from_session_key(
    request: Request,
    sandbox_id: str,
    session_api_key: str = Depends(
        APIKeyHeader(name='X-Session-API-Key', auto_error=False)
    ),
) -> SandboxInfo:
    """Authenticate via ``X-Session-API-Key`` and verify sandbox ownership."""
    sandbox_info = await validate_session_key(session_api_key)

    if sandbox_info.id != sandbox_id:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail='Session API key does not match sandbox',
        )

    return sandbox_info


async def _get_user_context(sandbox_info: SandboxInfo) -> AuthUserContext:
    """Build an ``AuthUserContext`` for the sandbox owner."""
    if not sandbox_info.created_by_user_id:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail='Sandbox has no associated user',
        )
    user_auth = await get_user_auth_for_user(sandbox_info.created_by_user_id)
    return AuthUserContext(user_auth=user_auth)


@router.get('/{sandbox_id}/settings/secrets')
async def list_secret_names(
    sandbox_info: SandboxInfo = Depends(_valid_sandbox_from_session_key),
) -> SecretNamesResponse:
    """List available secret names (no raw values).

    Includes both custom secrets and provider tokens (e.g. github_token).
    """
    user_context = await _get_user_context(sandbox_info)

    items: list[SecretNameItem] = []

    # Custom secrets
    secret_sources = await user_context.get_secrets()
    for name, source in secret_sources.items():
        items.append(SecretNameItem(name=name, description=source.description))

    # Provider tokens (github_token, gitlab_token, etc.)
    provider_env_vars = cast(
        dict[str, str] | None,
        await user_context.get_provider_tokens(as_env_vars=True),
    )
    if provider_env_vars:
        for env_key in provider_env_vars:
            items.append(
                SecretNameItem(name=env_key, description=f'{env_key} provider token')
            )

    return SecretNamesResponse(secrets=items)


@router.get('/{sandbox_id}/settings/secrets/{secret_name}')
async def get_secret_value(
    secret_name: str,
    sandbox_info: SandboxInfo = Depends(_valid_sandbox_from_session_key),
) -> Response:
    """Return a single secret value as plain text.

    Called by ``LookupSecret`` inside the sandbox. Checks custom secrets
    first, then falls back to provider tokens — always resolving the
    latest token at call time.
    """
    user_context = await _get_user_context(sandbox_info)

    # Check custom secrets first
    secret_sources = await user_context.get_secrets()
    source = secret_sources.get(secret_name)
    if source is not None:
        value = source.get_value()
        if value is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail='Secret has no value')
        return Response(content=value, media_type='text/plain')

    # Fall back to provider tokens (resolved fresh per request)
    provider_env_vars = cast(
        dict[str, str] | None,
        await user_context.get_provider_tokens(as_env_vars=True),
    )
    if provider_env_vars:
        token_value = provider_env_vars.get(secret_name)
        if token_value is not None:
            return Response(content=token_value, media_type='text/plain')

    raise HTTPException(status.HTTP_404_NOT_FOUND, detail='Secret not found')
