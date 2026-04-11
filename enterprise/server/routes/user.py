from typing import Any

from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import JSONResponse
from pydantic import SecretStr
from server.auth.token_manager import TokenManager
from storage.user_store import UserStore
from utils.identity import resolve_display_name

from openhands.app_server.utils.dependencies import get_dependencies
from openhands.integrations.provider import (
    PROVIDER_TOKEN_TYPE,
    ProviderHandler,
)
from openhands.integrations.service_types import (
    Branch,
    PaginatedBranchesResponse,
    ProviderType,
    Repository,
    SuggestedTask,
    User,
)
from openhands.microagent.types import (
    MicroagentContentResponse,
    MicroagentResponse,
)
from openhands.server.routes.git import (
    get_repository_branches,
    get_repository_microagent_content,
    get_repository_microagents,
    get_suggested_tasks,
    get_user,
    get_user_installations,
    get_user_repositories,
    search_branches,
    search_repositories,
)
from openhands.server.user_auth import (
    get_access_token,
    get_provider_tokens,
    get_user_id,
)

saas_user_router = APIRouter(prefix='/api/user', dependencies=get_dependencies())
token_manager = TokenManager()


@saas_user_router.get(
    '/installations',
    response_model=list[str],
    deprecated=True,
    description='Deprecated: Use `/api/v1/git/installations` instead.',
)
async def saas_get_user_installations(
    provider: ProviderType,
    provider_tokens: PROVIDER_TOKEN_TYPE | None = Depends(get_provider_tokens),
    access_token: SecretStr | None = Depends(get_access_token),
    user_id: str | None = Depends(get_user_id),
):
    if not provider_tokens:
        retval = await _check_idp(
            access_token=access_token,
            default_value=[],
        )
        if retval is not None:
            return retval

    return await get_user_installations(
        provider=provider,
        provider_tokens=provider_tokens,
        access_token=access_token,
        user_id=user_id,
    )


@saas_user_router.get('/git-organizations')
async def saas_get_user_git_organizations(
    provider_tokens: PROVIDER_TOKEN_TYPE | None = Depends(get_provider_tokens),
    access_token: SecretStr | None = Depends(get_access_token),
    user_id: str | None = Depends(get_user_id),
):
    if not provider_tokens:
        retval = await _check_idp(
            access_token=access_token,
            default_value={},
        )
        if retval is not None:
            return retval
        # _check_idp returned None (tokens refreshed on Keycloak side),
        # but provider_tokens is still None for this request.
        return JSONResponse(
            content='Git provider token required.',
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    client = ProviderHandler(
        provider_tokens=provider_tokens,
        external_auth_token=access_token,
        external_auth_id=user_id,
    )

    # SaaS users sign in with one provider at a time
    provider = next(iter(provider_tokens))

    if provider == ProviderType.GITHUB:
        orgs = await client.get_github_organizations()
    elif provider == ProviderType.GITLAB:
        orgs = await client.get_gitlab_groups()
    elif provider == ProviderType.BITBUCKET:
        orgs = await client.get_bitbucket_workspaces()
    else:
        return JSONResponse(
            content=f"Provider {provider.value} doesn't support git organizations",
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    return {
        'provider': provider.value,
        'organizations': orgs,
    }


@saas_user_router.get(
    '/repositories',
    response_model=list[Repository],
    deprecated=True,
    description='Deprecated: Use `/api/v1/git/repositories` instead.',
)
async def saas_get_user_repositories(
    sort: str = 'pushed',
    selected_provider: ProviderType | None = None,
    page: int | None = None,
    per_page: int | None = None,
    installation_id: str | None = None,
    provider_tokens: PROVIDER_TOKEN_TYPE | None = Depends(get_provider_tokens),
    access_token: SecretStr | None = Depends(get_access_token),
    user_id: str | None = Depends(get_user_id),
) -> list[Repository] | JSONResponse:
    if not provider_tokens:
        retval = await _check_idp(
            access_token=access_token,
            default_value=[],
        )
        if retval is not None:
            return retval

    return await get_user_repositories(
        sort=sort,
        selected_provider=selected_provider,
        page=page,
        per_page=per_page,
        installation_id=installation_id,
        provider_tokens=provider_tokens,
        access_token=access_token,
        user_id=user_id,
    )


@saas_user_router.get('/info', response_model=User, deprecated=True)
async def saas_get_user(
    provider_tokens: PROVIDER_TOKEN_TYPE | None = Depends(get_provider_tokens),
    access_token: SecretStr | None = Depends(get_access_token),
    user_id: str | None = Depends(get_user_id),
) -> User | JSONResponse:
    """Get the current user git info. Use GET /api/v1/users/git-info instead"""
    if not provider_tokens:
        if not access_token:
            return JSONResponse(
                content='User is not authenticated.',
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
        user_info = await token_manager.get_user_info(access_token.get_secret_value())
        # Prefer email from DB; fall back to Keycloak if not yet persisted
        email = user_info.email
        sub = user_info.sub
        if sub:
            db_user = await UserStore.get_user_by_id(sub)
            if db_user and db_user.email is not None:
                email = db_user.email

        user_info_dict = user_info.model_dump(exclude_none=True)
        retval = await _check_idp(
            access_token=access_token,
            default_value=User(
                id=sub,
                login=user_info.preferred_username or '',
                avatar_url='',
                email=email,
                name=resolve_display_name(user_info_dict),
                company=user_info.company,
            ),
            user_info=user_info_dict,
        )
        if retval is not None:
            return retval

    return await get_user(
        provider_tokens=provider_tokens, access_token=access_token, user_id=user_id
    )


@saas_user_router.get('/search/repositories', response_model=list[Repository])
async def saas_search_repositories(
    query: str,
    per_page: int = 5,
    sort: str = 'stars',
    order: str = 'desc',
    selected_provider: ProviderType | None = None,
    provider_tokens: PROVIDER_TOKEN_TYPE | None = Depends(get_provider_tokens),
    access_token: SecretStr | None = Depends(get_access_token),
    user_id: str | None = Depends(get_user_id),
) -> list[Repository] | JSONResponse:
    if not provider_tokens:
        retval = await _check_idp(
            access_token=access_token,
            default_value=[],
        )
        if retval is not None:
            return retval

    return await search_repositories(
        query=query,
        per_page=per_page,
        sort=sort,
        order=order,
        selected_provider=selected_provider,
        provider_tokens=provider_tokens,
        access_token=access_token,
        user_id=user_id,
    )


@saas_user_router.get('/suggested-tasks', response_model=list[SuggestedTask])
async def saas_get_suggested_tasks(
    provider_tokens: PROVIDER_TOKEN_TYPE | None = Depends(get_provider_tokens),
    access_token: SecretStr | None = Depends(get_access_token),
    user_id: str | None = Depends(get_user_id),
) -> list[SuggestedTask] | JSONResponse:
    """Get suggested tasks for the authenticated user across their most recently pushed repositories.

    Returns:
    - PRs owned by the user
    - Issues assigned to the user.
    """
    if not provider_tokens:
        retval = await _check_idp(
            access_token=access_token,
            default_value=[],
        )
        if retval is not None:
            return retval

    return await get_suggested_tasks(
        provider_tokens=provider_tokens, access_token=access_token, user_id=user_id
    )


@saas_user_router.get('/repository/branches', response_model=PaginatedBranchesResponse)
async def saas_get_repository_branches(
    repository: str,
    page: int = 1,
    per_page: int = 30,
    provider_tokens: PROVIDER_TOKEN_TYPE | None = Depends(get_provider_tokens),
    access_token: SecretStr | None = Depends(get_access_token),
    user_id: str | None = Depends(get_user_id),
) -> PaginatedBranchesResponse | JSONResponse:
    """Get branches for a repository.

    Args:
        repository: The repository name in the format 'owner/repo'

    Returns:
        A list of branches for the repository
    """
    if not provider_tokens:
        retval = await _check_idp(
            access_token=access_token,
            default_value=[],
        )
        if retval is not None:
            return retval

    return await get_repository_branches(
        repository=repository,
        page=page,
        per_page=per_page,
        provider_tokens=provider_tokens,
        access_token=access_token,
        user_id=user_id,
    )


@saas_user_router.get('/search/branches', response_model=list[Branch])
async def saas_search_branches(
    repository: str,
    query: str,
    per_page: int = 30,
    selected_provider: ProviderType | None = None,
    provider_tokens: PROVIDER_TOKEN_TYPE | None = Depends(get_provider_tokens),
    access_token: SecretStr | None = Depends(get_access_token),
    user_id: str | None = Depends(get_user_id),
) -> list[Branch] | JSONResponse:
    if not provider_tokens:
        retval = await _check_idp(
            access_token=access_token,
            default_value=[],
        )
        if retval is not None:
            return retval

    return await search_branches(
        repository=repository,
        query=query,
        per_page=per_page,
        selected_provider=selected_provider,
        provider_tokens=provider_tokens,
        access_token=access_token,
        user_id=user_id,
    )


@saas_user_router.get(
    '/repository/{repository_name:path}/microagents',
    response_model=list[MicroagentResponse],
)
async def saas_get_repository_microagents(
    repository_name: str,
    provider_tokens: PROVIDER_TOKEN_TYPE | None = Depends(get_provider_tokens),
    access_token: SecretStr | None = Depends(get_access_token),
    user_id: str | None = Depends(get_user_id),
) -> list[MicroagentResponse] | JSONResponse:
    """Scan the microagents directory of a repository and return the list of microagents.

    The microagents directory location depends on the git provider and actual repository name:
    - If git provider is not GitLab and actual repository name is ".openhands": scans "microagents" folder
    - If git provider is GitLab and actual repository name is "openhands-config": scans "microagents" folder
    - Otherwise: scans ".openhands/microagents" folder

    Note: This API returns microagent metadata without content for performance.
    Use the separate content API to fetch individual microagent content.

    Args:
        repository_name: Repository name in the format 'owner/repo' or 'domain/owner/repo'
        provider_tokens: Provider tokens for authentication
        access_token: Access token for external authentication
        user_id: User ID for authentication

    Returns:
        List of microagents found in the repository's microagents directory (without content)
    """
    if not provider_tokens:
        retval = await _check_idp(
            access_token=access_token,
            default_value=[],
        )
        if retval is not None:
            return retval

    return await get_repository_microagents(
        repository_name=repository_name,
        provider_tokens=provider_tokens,
        access_token=access_token,
        user_id=user_id,
    )


@saas_user_router.get(
    '/repository/{repository_name:path}/microagents/content',
    response_model=MicroagentContentResponse,
)
async def saas_get_repository_microagent_content(
    repository_name: str,
    file_path: str = Query(
        ..., description='Path to the microagent file within the repository'
    ),
    provider_tokens: PROVIDER_TOKEN_TYPE | None = Depends(get_provider_tokens),
    access_token: SecretStr | None = Depends(get_access_token),
    user_id: str | None = Depends(get_user_id),
) -> MicroagentContentResponse | JSONResponse:
    """Fetch the content of a specific microagent file from a repository.

    Args:
        repository_name: Repository name in the format 'owner/repo' or 'domain/owner/repo'
        file_path: Query parameter - Path to the microagent file within the repository
        provider_tokens: Provider tokens for authentication
        access_token: Access token for external authentication
        user_id: User ID for authentication

    Returns:
        Microagent file content and metadata

    Example:
        GET /api/user/repository/owner/repo/microagents/content?file_path=.openhands/microagents/my-agent.md
    """
    if not provider_tokens:
        retval = await _check_idp(
            access_token=access_token,
            default_value=MicroagentContentResponse(content='', path=''),
        )
        if retval is not None:
            return retval

    return await get_repository_microagent_content(
        repository_name=repository_name,
        file_path=file_path,
        provider_tokens=provider_tokens,
        access_token=access_token,
        user_id=user_id,
    )


async def _check_idp(
    access_token: SecretStr | None,
    default_value: Any,
    user_info: dict | None = None,
):
    if not access_token:
        return JSONResponse(
            content='User is not authenticated.',
            status_code=status.HTTP_401_UNAUTHORIZED,
        )
    if user_info is None:
        user_info_model = await token_manager.get_user_info(
            access_token.get_secret_value()
        )
        user_info = user_info_model.model_dump(exclude_none=True)
    idp: str | None = user_info.get('identity_provider')
    if not idp:
        return JSONResponse(
            content='IDP not found.',
            status_code=status.HTTP_401_UNAUTHORIZED,
        )
    if ':' in idp:
        idp, _ = idp.rsplit(':', 1)

    # Will return empty dict if IDP doesn't support provider tokens
    if not await token_manager.get_idp_tokens_from_keycloak(
        access_token.get_secret_value(), ProviderType(idp)
    ):
        return default_value
    return None
