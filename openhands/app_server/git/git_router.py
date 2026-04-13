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
    BranchPage,
    InstallationPage,
    RepositoryPage,
    SortOrder,
    SuggestedTaskPage,
)
from openhands.app_server.user.user_context import UserContext
from openhands.app_server.utils.dependencies import get_dependencies
from openhands.app_server.utils.paging_utils import (
    decode_page_id,
    encode_page_id,
    paginate_results,
)
from openhands.integrations.provider import ProviderHandler
from openhands.integrations.service_types import (
    Branch,
    ProviderType,
    Repository,
    SuggestedTask,
)

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


@router.get('/installations/search')
async def search_user_installations(
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


@router.get('/repositories/search')
async def search_repositories(
    provider: ProviderType,
    query: Annotated[
        str | None,
        Query(
            title='Search query for finding repositories. If not provided, returns user repositories.'
        ),
    ] = None,
    installation_id: Annotated[
        str | None,
        Query(title='Filter by installation/app ID'),
    ] = None,
    page_id: Annotated[
        str | None,
        Query(title='Optional next_page_id from the previously returned page'),
    ] = None,
    limit: Annotated[
        int,
        Query(title='The max number of results in the page', gt=0, le=100),
    ] = 100,
    sort_order: Annotated[
        SortOrder | None,
        Query(title='Sort order for search results (e.g., stars-desc, forks-asc)'),
    ] = None,
    user_context: UserContext = user_context_dependency,
) -> RepositoryPage:
    """Get or search repositories.

    If query is provided, searches repositories across the git provider.
    If query is not provided, returns a paginated list of the authenticated user's repositories.
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

    # If query is provided, use search; otherwise get user's repositories
    if query:
        # Parse sort_order into sort and order components (if provided)
        if sort_order:
            search_sort, order = sort_order.value.rsplit('-', 1)
        else:
            search_sort = 'stars'
            order = 'desc'

        repos: list[Repository] = await client.search_repositories(
            selected_provider=provider,
            query=query,
            per_page=limit + 1,
            sort=search_sort,
            order=order,
            app_mode=get_global_config().app_mode,
        )
    else:
        if sort_order:
            # TODO: This is a temporary state until we refactor the underlying API.
            # The get_repositories method does not support sorting in the same way as
            # the search method - those should be merged into a single paginated
            # method
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='sort_order is not supported when listing user repositories. It will be supported after API refactoring.',
            )
        # TODO: The underlying API needs refactoring.
        repos = await client.get_repositories(
            sort='pushed',
            app_mode=get_global_config().app_mode,
            selected_provider=provider,
            page=page,
            per_page=limit + 1,
            installation_id=installation_id,
        )

    next_page_id = None
    if len(repos) > limit:
        repos = repos[:-1]
        next_page_id = encode_page_id(page + 1)

    return RepositoryPage(items=repos, next_page_id=next_page_id)


@router.get('/branches/search')
async def search_branches(
    provider: ProviderType,
    repository: Annotated[
        str,
        Query(title='Repository name in format owner/repo'),
    ],
    query: Annotated[
        str,
        Query(title='Branch name search query'),
    ],
    page_id: Annotated[
        str | None,
        Query(title='Optional next_page_id from the previously returned page'),
    ] = None,
    limit: Annotated[
        int,
        Query(title='The max number of results in the page', gt=0, le=100),
    ] = 30,
    user_context: UserContext = user_context_dependency,
) -> BranchPage:
    """Search branches in a repository.

    Returns a paginated list of branches matching the search query.
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

    if query:
        if page != 1:
            # TODO(#13883): Support pagination for branch search after refactoring.
            # The search_branches method does not support paging in the same way as
            # get_branches - those should be merged into a single paginated method
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Pagination not yet supported for branch search queries. Use empty query to list all branches with pagination.',
            )
        # Get search results - we'll handle pagination ourselves
        branches: list[Branch] = await client.search_branches(
            selected_provider=provider,
            repository=repository,
            query=query,
            per_page=limit + 1,
        )
    else:
        current_page = await client.get_branches(
            repository=repository,
            specified_provider=provider,
            page=page,
            per_page=limit + 1,
        )
        branches = current_page.branches

    next_page_id = None
    if len(branches) > limit:
        branches = branches[:-1]
        next_page_id = encode_page_id(page + 1)

    return BranchPage(items=branches, next_page_id=next_page_id)


@router.get('/suggested-tasks/search')
async def search_suggested_tasks(
    page_id: Annotated[
        str | None,
        Query(title='Optional next_page_id from the previously returned page'),
    ] = None,
    limit: Annotated[
        int,
        Query(title='The max number of results in the page', gt=0, le=100),
    ] = 30,
    user_context: UserContext = user_context_dependency,
) -> SuggestedTaskPage:
    """Get suggested tasks for the user.

    Returns a paginated list of suggested tasks including:
    - PRs owned by the user
    - Issues assigned to the user
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

    # Get suggested tasks - we'll handle pagination ourselves
    # The underlying method doesn't have pagination, so we'll fetch all and paginate
    all_tasks: list[SuggestedTask] = await client.get_suggested_tasks()

    # Paginate results
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    paginated_tasks = all_tasks[start_idx:end_idx]

    next_page_id = None
    if end_idx < len(all_tasks):
        next_page_id = encode_page_id(page + 1)

    return SuggestedTaskPage(items=paginated_tasks, next_page_id=next_page_id)
