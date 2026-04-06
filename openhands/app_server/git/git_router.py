"""Git router for OpenHands App Server V1 API.

This module provides V1 API endpoints for Git operations (installations, repositories)
with pagination support. These endpoints are designed to replace the legacy V0 endpoints
in openhands/server/routes/git.py.
"""

from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, HTTPException, Query, status

from openhands.app_server.config import depends_user_context, get_global_config
from openhands.app_server.git.git_models import (
    InstallationPage,
    RepositoryPage,
)
from openhands.app_server.user.user_context import UserContext
from openhands.app_server.utils.dependencies import get_dependencies
from openhands.app_server.utils.paging_utils import (
    decode_page_id,
    encode_page_id,
    paginate_results,
)
from openhands.integrations.provider import ProviderHandler
from openhands.integrations.service_types import ProviderType

if TYPE_CHECKING:
    from openhands.integrations.provider import PROVIDER_TOKEN_TYPE

# We use the get_dependencies method here to signal to the OpenAPI docs that this endpoint
# is protected. The actual protection is provided by SetAuthCookieMiddleware
router = APIRouter(
    prefix='/git',
    tags=['Git'],
    dependencies=get_dependencies(),
)
user_context_dependency = depends_user_context()


@router.get('/installations')
async def get_user_installations(
    provider: ProviderType,
    page_id: Annotated[
        str | None,
        Query(title='Optional next_page_id from the previously returned page'),
    ] = None,
    limit: Annotated[
        int,
        Query(title='The max number of results in the page', gt=0, le=100),
    ] = 100,
    user_context: UserContext = user_context_dependency,
) -> InstallationPage:
    """Get user installations (GitHub apps) or equivalent for other providers.

    Returns a paginated list of installation IDs or workspace IDs depending on the provider.
    """
    # Get provider tokens from user context
    provider_tokens = await user_context.get_provider_tokens()
    if not provider_tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Git provider token required (such as GitHub).',
        )

    user_id = await user_context.get_user_id()
    # Wrap in MappingProxyType as required by ProviderHandler
    # Type ignore: we validated provider_tokens exists, but mypy can't narrow the union type
    client = ProviderHandler(
        provider_tokens=MappingProxyType(provider_tokens),  # type: ignore[arg-type]
        external_auth_id=user_id,
    )

    if provider == ProviderType.GITHUB:
        installations = await client.get_github_installations()
    elif provider == ProviderType.BITBUCKET:
        installations = await client.get_bitbucket_workspaces()
    elif provider == ProviderType.BITBUCKET_DATA_CENTER:
        installations = await client.get_bitbucket_dc_projects()
    elif provider == ProviderType.AZURE_DEVOPS:
        installations = await client.get_azure_devops_organizations()
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Provider {provider} doesn't support installations",
        )

    items, next_page_id = paginate_results(installations, page_id, limit)
    return InstallationPage(items=items, next_page_id=next_page_id)


@router.get('/repositories')
async def get_user_repositories(
    provider: ProviderType,
    sort: str = 'pushed',
    installation_id: str | None = None,
    page_id: Annotated[
        str | None,
        Query(title='Optional next_page_id from the previously returned page'),
    ] = None,
    limit: Annotated[
        int,
        Query(title='The max number of results in the page', gt=0, le=100),
    ] = 100,
    user_context: UserContext = user_context_dependency,
) -> RepositoryPage:
    """Get user repositories.

    Returns a paginated list of repositories for the authenticated user.
    """
    # Get provider tokens from user context
    provider_tokens = await user_context.get_provider_tokens()
    if not provider_tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Git provider token required (such as GitHub).',
        )

    user_id = await user_context.get_user_id()
    # Cast to the expected type since we validated provider_tokens exists
    typed_provider_tokens: PROVIDER_TOKEN_TYPE = provider_tokens  # type: ignore[assignment]
    client = ProviderHandler(
        provider_tokens=MappingProxyType(typed_provider_tokens),
        external_auth_id=user_id,
    )

    page = 1
    decoded_page_id = decode_page_id(page_id)
    if decoded_page_id is not None:
        page = decoded_page_id

    # Get repositories - we'll handle pagination ourselves
    items = await client.get_repositories(
        sort=sort,
        app_mode=get_global_config().app_mode,
        selected_provider=provider,
        page=page,
        per_page=limit + 1,  # We'll handle pagination ourselves
        installation_id=installation_id,
    )

    next_page_id = None
    if len(items) > limit:
        items = items[:-1]
        next_page_id = encode_page_id(page + 1)

    return RepositoryPage(items=items, next_page_id=next_page_id)
