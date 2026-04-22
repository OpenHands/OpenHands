"""SAAS-specific extensions for the /api/v1/users endpoints.

This module provides SAAS-specific implementations that extend the OSS
user endpoints with organization context (org_id, org_name, role, permissions).
"""

import logging
from types import MappingProxyType
from typing import Any, cast

from fastapi import APIRouter, FastAPI, Header, HTTPException, Query, status
from fastapi.responses import JSONResponse
from server.auth.saas_user_auth import SaasUserAuth
from server.models.user_models import GitOrganizationsResponse, SaasUserInfo

from openhands.app_server.config import (
    depends_user_context,
    resolve_provider_llm_base_url,
)
from openhands.app_server.sandbox.session_auth import validate_session_key_ownership
from openhands.app_server.user.auth_user_context import AuthUserContext
from openhands.app_server.user.user_context import UserContext
from openhands.app_server.utils.dependencies import get_dependencies
from openhands.integrations.provider import ProviderHandler
from openhands.integrations.service_types import ProviderType

_logger = logging.getLogger(__name__)

saas_users_v1_router = APIRouter(
    prefix='/api/v1/users', tags=['User'], dependencies=get_dependencies()
)
user_dependency = depends_user_context()


def _inject_sdk_compat_fields(
    content: dict[str, Any], *, include_api_key: bool
) -> None:
    """Inject flat top-level convenience fields for the SDK.

    The SDK's ``get_llm()`` and ``get_mcp_config()`` read ``llm_model``,
    ``llm_api_key``, ``llm_base_url``, and ``mcp_config`` from the top
    level of the ``/api/v1/users/me`` response. These values live inside
    the nested ``agent_settings`` structure, so we mirror them at the top
    level for backward compatibility.

    The canonical representation is ``agent_settings``; these flat fields
    exist solely for SDK backward compatibility.
    """
    agent_settings = content.get('agent_settings') or {}
    llm = agent_settings.get('llm') or {}
    model = llm.get('model')
    content['llm_model'] = model
    content['llm_base_url'] = resolve_provider_llm_base_url(model, llm.get('base_url'))
    if include_api_key:
        content['llm_api_key'] = llm.get('api_key')
    content['mcp_config'] = agent_settings.get('mcp_config')


@saas_users_v1_router.get('/me')
async def get_current_user_saas(
    user_context: UserContext = user_dependency,
    expose_secrets: bool = Query(
        default=False,
        description='If true, return unmasked secret values (e.g. llm_api_key). '
        'Requires a valid X-Session-API-Key header for an active sandbox '
        'owned by the authenticated user.',
    ),
    x_session_api_key: str | None = Header(default=None),
) -> SaasUserInfo:
    """Get the current authenticated user with SAAS-specific org info.

    Returns user settings along with organization context:
    - org_id: Current organization ID
    - org_name: Current organization name
    - role: User's role in the organization
    - permissions: List of permission strings for the role
    """
    # Get base user info from the context
    base_user_info = await user_context.get_user_info()
    if base_user_info is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail='Not authenticated')

    # Build SAAS user info from base settings
    user_info_data = base_user_info.model_dump(
        mode='json', context={'expose_secrets': True}
    )

    # Add org info if available (from SaasUserAuth)
    org_info = await _get_org_info_from_context(user_context)
    if org_info:
        user_info_data.update(org_info)

    user_info = SaasUserInfo(**user_info_data)

    if expose_secrets:
        await validate_session_key_ownership(user_context, x_session_api_key)
        content = user_info.model_dump(mode='json', context={'expose_secrets': True})
        _inject_sdk_compat_fields(content, include_api_key=True)
        return JSONResponse(content=content)  # type: ignore[return-value]

    content = user_info.model_dump(mode='json')
    _inject_sdk_compat_fields(content, include_api_key=False)
    return JSONResponse(content=content)  # type: ignore[return-value]


@saas_users_v1_router.get('/git-organizations')
async def get_current_user_git_organizations(
    user_context: UserContext = user_dependency,
) -> GitOrganizationsResponse:
    """Return the Git organizations, groups, or workspaces the user belongs to
    on their active provider.

    In SAAS mode users sign in with one provider at a time, so the response
    reflects that single provider.
    """
    provider_tokens = await user_context.get_provider_tokens()
    if not provider_tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Git provider token required.',
        )

    user_id = await user_context.get_user_id()
    client = ProviderHandler(
        provider_tokens=MappingProxyType(provider_tokens),  # type: ignore[arg-type]
        external_auth_id=user_id,
    )

    provider = cast(ProviderType, next(iter(provider_tokens)))
    if provider == ProviderType.GITHUB:
        orgs = await client.get_github_organizations()
    elif provider == ProviderType.GITLAB:
        orgs = await client.get_gitlab_groups()
    elif provider == ProviderType.BITBUCKET:
        orgs = await client.get_bitbucket_workspaces()
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Provider {provider.value} doesn't support git organizations",
        )

    return GitOrganizationsResponse(provider=provider, organizations=orgs)


async def _get_org_info_from_context(user_context: UserContext) -> dict | None:
    """Extract org info from the user context if available.

    This works by checking if the underlying user_auth is a SaasUserAuth
    instance that has the get_org_info method.
    """
    # Check if this is an AuthUserContext with a SaasUserAuth
    if isinstance(user_context, AuthUserContext):
        user_auth = user_context.user_auth
        if isinstance(user_auth, SaasUserAuth):
            return await user_auth.get_org_info()
    return None


def override_users_me_endpoint(app: FastAPI) -> None:
    """Override the OSS /api/v1/users/me endpoint with SAAS version.

    This removes the base OSS endpoint and registers the SAAS version
    which includes organization context (org_id, org_name, role, permissions).

    Must be called after the app is created in saas_server.py.
    """
    # Find and remove the OSS /api/v1/users/me route
    routes_to_remove = []
    for route in app.routes:
        if hasattr(route, 'path') and route.path == '/api/v1/users/me':
            routes_to_remove.append(route)

    for route in routes_to_remove:
        app.routes.remove(route)
        _logger.debug('Removed OSS route: %s', route.path)

    # Add the SAAS version
    app.include_router(saas_users_v1_router)
    _logger.debug('Added SAAS /api/v1/users/me endpoint')
