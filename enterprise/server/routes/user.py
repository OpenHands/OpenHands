from typing import Any

from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import JSONResponse
from pydantic import SecretStr
from server.auth.token_manager import TokenManager
from utils.identity import resolve_display_name

from openhands.integrations.provider import (
    PROVIDER_TOKEN_TYPE,
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
from openhands.server.dependencies import get_dependencies
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


@saas_user_router.get('/installations', response_model=list[str])
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


@saas_user_router.get('/repositories', response_model=list[Repository])
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


@saas_user_router.get('/info', response_model=User)
async def saas_get_user(
    provider_tokens: PROVIDER_TOKEN_TYPE | None = Depends(get_provider_tokens),
    access_token: SecretStr | None = Depends(get_access_token),
    user_id: str | None = Depends(get_user_id),
) -> User | JSONResponse:
    if not provider_tokens:
        if not access_token:
            return JSONResponse(
                content='User is not authenticated.',
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
        user_info = await token_manager.get_user_info(access_token.get_secret_value())
        if not user_info:
            return JSONResponse(
                content='Failed to retrieve user_info.',
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
        retval = await _check_idp(
            access_token=access_token,
            default_value=User(
                id=(user_info.get('sub') if user_info else '') or '',
                login=(user_info.get('preferred_username') if user_info else '') or '',
                avatar_url='',
                email=user_info.get('email') if user_info else None,
                name=resolve_display_name(user_info) if user_info else None,
                company=user_info.get('company') if user_info else None,
            ),
            user_info=user_info,
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
    user_info = (
        user_info
        if user_info
        else await token_manager.get_user_info(access_token.get_secret_value())
    )
    if not user_info:
        return JSONResponse(
            content='Failed to retrieve user_info.',
            status_code=status.HTTP_401_UNAUTHORIZED,
        )
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
