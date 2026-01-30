# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
# This module belongs to the old V0 web server. The V1 application server lives under openhands/app_server/.
from types import MappingProxyType
from typing import Annotated, cast

from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import JSONResponse
from pydantic import SecretStr

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.provider import (
    PROVIDER_TOKEN_TYPE,
    ProviderHandler,
)
from openhands.integrations.service_types import (
    AuthenticationError,
    Branch,
    PaginatedBranchesResponse,
    ProviderType,
    Repository,
    SuggestedTask,
    UnknownException,
    User,
)
from openhands.microagent.types import (
    MicroagentContentResponse,
    MicroagentResponse,
)
from openhands.server.dependencies import get_dependencies
from openhands.server.shared import server_config
from openhands.server.user_auth import (
    get_access_token,
    get_provider_tokens,
    get_user_id,
)

app = APIRouter(prefix='/api/user', dependencies=get_dependencies())


@app.get('/installations', response_model=list[str])
async def get_user_installations(
    provider: ProviderType,
    provider_tokens: PROVIDER_TOKEN_TYPE | None = Depends(get_provider_tokens),
    access_token: SecretStr | None = Depends(get_access_token),
    user_id: str | None = Depends(get_user_id),
):
    if provider_tokens:
        client = ProviderHandler(
            provider_tokens=provider_tokens,
            external_auth_token=access_token,
            external_auth_id=user_id,
        )

        if provider == ProviderType.GITHUB:
            return await client.get_github_installations()
        elif provider == ProviderType.BITBUCKET:
            return await client.get_bitbucket_workspaces()
        elif provider == ProviderType.AZURE_DEVOPS:
            return await client.get_azure_devops_organizations()
        else:
            return JSONResponse(
                content=f"Provider {provider} doesn't support installations",
                status_code=status.HTTP_400_BAD_REQUEST,
            )

    raise AuthenticationError('Git provider token required. (such as GitHub).')


@app.get('/repositories', response_model=list[Repository])
async def get_user_repositories(
    sort: str = 'pushed',
    selected_provider: Annotated[ProviderType | None, Query()] = None,
    page: int | None = None,
    per_page: int | None = None,
    installation_id: str | None = None,
    provider_tokens: PROVIDER_TOKEN_TYPE | None = Depends(get_provider_tokens),
    access_token: SecretStr | None = Depends(get_access_token),
    user_id: str | None = Depends(get_user_id),
) -> list[Repository] | JSONResponse:
    if provider_tokens:
        client = ProviderHandler(
            provider_tokens=provider_tokens,
            external_auth_token=access_token,
            external_auth_id=user_id,
        )

        try:
            return await client.get_repositories(
                sort,
                server_config.app_mode,
                selected_provider,
                page,
                per_page,
                installation_id,
            )

        except UnknownException as e:
            return JSONResponse(
                content=str(e),
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    logger.info(
        f'Returning 401 Unauthorized - Git provider token required for user_id: {user_id}'
    )
    raise AuthenticationError('Git provider token required. (such as GitHub).')


@app.get('/info', response_model=User)
async def get_user(
    provider_tokens: PROVIDER_TOKEN_TYPE | None = Depends(get_provider_tokens),
    access_token: SecretStr | None = Depends(get_access_token),
    user_id: str | None = Depends(get_user_id),
) -> User | JSONResponse:
    if provider_tokens:
        client = ProviderHandler(
            provider_tokens=provider_tokens, external_auth_token=access_token
        )

        try:
            user: User = await client.get_user()
            return user

        except UnknownException as e:
            return JSONResponse(
                content=str(e),
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    logger.info(
        f'Returning 401 Unauthorized - Git provider token required for user_id: {user_id}'
    )
    raise AuthenticationError('Git provider token required. (such as GitHub).')


@app.get('/search/repositories', response_model=list[Repository])
async def search_repositories(
    query: str,
    per_page: int = 5,
    sort: str = 'stars',
    order: str = 'desc',
    selected_provider: Annotated[ProviderType | None, Query()] = None,
    provider_tokens: PROVIDER_TOKEN_TYPE | None = Depends(get_provider_tokens),
    access_token: SecretStr | None = Depends(get_access_token),
    user_id: str | None = Depends(get_user_id),
) -> list[Repository] | JSONResponse:
    if provider_tokens:
        client = ProviderHandler(
            provider_tokens=provider_tokens,
            external_auth_token=access_token,
            external_auth_id=user_id,
        )
        try:
            repos: list[Repository] = await client.search_repositories(
                selected_provider, query, per_page, sort, order, server_config.app_mode
            )
            return repos

        except UnknownException as e:
            return JSONResponse(
                content=str(e),
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    logger.info(
        f'Returning 401 Unauthorized - Git provider token required for user_id: {user_id}'
    )
    raise AuthenticationError('Git provider token required.')


@app.get('/search/branches', response_model=list[Branch])
async def search_branches(
    repository: str,
    query: str,
    per_page: int = 30,
    selected_provider: Annotated[ProviderType | None, Query()] = None,
    provider_tokens: PROVIDER_TOKEN_TYPE | None = Depends(get_provider_tokens),
    access_token: SecretStr | None = Depends(get_access_token),
    user_id: str | None = Depends(get_user_id),
) -> list[Branch] | JSONResponse:
    if provider_tokens:
        client = ProviderHandler(
            provider_tokens=provider_tokens,
            external_auth_token=access_token,
            external_auth_id=user_id,
        )
        try:
            branches: list[Branch] = await client.search_branches(
                selected_provider, repository, query, per_page
            )
            return branches

        except AuthenticationError as e:
            return JSONResponse(
                content=str(e),
                status_code=status.HTTP_401_UNAUTHORIZED,
            )

        except UnknownException as e:
            return JSONResponse(
                content=str(e),
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    logger.info(
        f'Returning 401 Unauthorized - Git provider token required for user_id: {user_id}'
    )
    return JSONResponse(
        content='Git provider token required.',
        status_code=status.HTTP_401_UNAUTHORIZED,
    )


@app.get('/suggested-tasks', response_model=list[SuggestedTask])
async def get_suggested_tasks(
    provider_tokens: PROVIDER_TOKEN_TYPE | None = Depends(get_provider_tokens),
    access_token: SecretStr | None = Depends(get_access_token),
    user_id: str | None = Depends(get_user_id),
) -> list[SuggestedTask] | JSONResponse:
    """Get suggested tasks for the authenticated user across their most recently pushed repositories.

    Returns:
    - PRs owned by the user
    - Issues assigned to the user.
    """
    if provider_tokens:
        client = ProviderHandler(
            provider_tokens=provider_tokens, external_auth_token=access_token
        )
        try:
            tasks: list[SuggestedTask] = await client.get_suggested_tasks()
            return tasks

        except UnknownException as e:
            return JSONResponse(
                content=str(e),
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
    logger.info(f'Returning 401 Unauthorized - No providers set for user_id: {user_id}')
    raise AuthenticationError('No providers set.')


@app.get('/repository/branches', response_model=PaginatedBranchesResponse)
async def get_repository_branches(
    repository: str,
    page: int = 1,
    per_page: int = 30,
    selected_provider: Annotated[ProviderType | None, Query()] = None,
    provider_tokens: PROVIDER_TOKEN_TYPE | None = Depends(get_provider_tokens),
    access_token: SecretStr | None = Depends(get_access_token),
    user_id: str | None = Depends(get_user_id),
) -> PaginatedBranchesResponse | JSONResponse:
    """Get branches for a repository.

    Args:
        repository: The repository name in the format 'owner/repo'
        page: Page number for pagination (default: 1)
        per_page: Number of branches per page (default: 30)
        selected_provider: Optional provider hint to avoid trying other providers

    Returns:
        A paginated response with branches for the repository
    """
    if provider_tokens:
        client = ProviderHandler(
            provider_tokens=provider_tokens, external_auth_token=access_token
        )
        try:
            branches_response: PaginatedBranchesResponse = await client.get_branches(
                repository,
                specified_provider=selected_provider,
                page=page,
                per_page=per_page,
            )
            return branches_response

        except UnknownException as e:
            return JSONResponse(
                content=str(e),
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    logger.info(
        f'Returning 401 Unauthorized - Git provider token required for user_id: {user_id}'
    )
    raise AuthenticationError('Git provider token required. (such as GitHub).')


def _extract_repo_name(repository_name: str) -> str:
    """Extract the actual repository name from the full repository path.

    Args:
        repository_name: Repository name in format 'owner/repo' or 'domain/owner/repo'

    Returns:
        The actual repository name (last part after the last '/')
    """
    return repository_name.split('/')[-1]


@app.get(
    '/repository/{repository_name:path}/microagents',
    response_model=list[MicroagentResponse],
)
async def get_repository_microagents(
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
    try:
        # Create provider handler for API authentication
        provider_handler = ProviderHandler(
            provider_tokens=provider_tokens
            or cast(PROVIDER_TOKEN_TYPE, MappingProxyType({})),
            external_auth_token=access_token,
            external_auth_id=user_id,
        )

        # Fetch microagents using the provider handler
        microagents = await provider_handler.get_microagents(repository_name)

        logger.info(f'Found {len(microagents)} microagents in {repository_name}')
        return microagents

    except AuthenticationError:
        raise

    except RuntimeError as e:
        return JSONResponse(
            content=str(e),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    except Exception as e:
        logger.error(
            f'Error scanning repository {repository_name}: {str(e)}', exc_info=True
        )
        return JSONResponse(
            content=f'Error scanning repository: {str(e)}',
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@app.get(
    '/repository/{repository_name:path}/microagents/content',
    response_model=MicroagentContentResponse,
)
async def get_repository_microagent_content(
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
    try:
        # Create provider handler for API authentication
        provider_handler = ProviderHandler(
            provider_tokens=provider_tokens
            or cast(PROVIDER_TOKEN_TYPE, MappingProxyType({})),
            external_auth_token=access_token,
            external_auth_id=user_id,
        )

        # Fetch file content using the provider handler
        response = await provider_handler.get_microagent_content(
            repository_name, file_path
        )

        logger.info(
            f'Retrieved content for microagent {file_path} from {repository_name}'
        )

        return response

    except AuthenticationError:
        raise

    except RuntimeError as e:
        return JSONResponse(
            content=str(e),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    except Exception as e:
        logger.error(
            f'Error fetching microagent content from {repository_name}/{file_path}: {str(e)}',
            exc_info=True,
        )
        return JSONResponse(
            content=f'Error fetching microagent content: {str(e)}',
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
