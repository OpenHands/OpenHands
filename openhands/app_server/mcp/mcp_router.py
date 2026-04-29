import os
import re
from typing import Annotated
from uuid import UUID

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_http_request
from pydantic import Field

from openhands.app_server.config import (
    get_app_conversation_info_service,
    get_global_config,
)
from openhands.app_server.integrations.azure_devops.azure_devops_service import (
    AzureDevOpsServiceImpl,
)
from openhands.app_server.integrations.bitbucket.bitbucket_service import (
    BitBucketServiceImpl,
)
from openhands.app_server.integrations.bitbucket_data_center.bitbucket_dc_service import (
    BitbucketDCServiceImpl,
)
from openhands.app_server.integrations.github.github_service import GithubServiceImpl
from openhands.app_server.integrations.gitlab.gitlab_service import GitLabServiceImpl
from openhands.app_server.integrations.provider import ProviderToken
from openhands.app_server.integrations.service_types import GitService, ProviderType
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.user.specifiy_user_context import (
    USER_CONTEXT_ATTR,
    SpecifyUserContext,
)
from openhands.app_server.user_auth import (
    get_access_token,
    get_provider_tokens,
    get_user_id,
)
from openhands.core.logger import openhands_logger as logger
from openhands.server.types import AppMode

mcp_server = FastMCP('mcp', mask_error_details=True)

HOST = f'https://{os.getenv("WEB_HOST", "app.all-hands.dev").strip()}'
CONVERSATION_URL = HOST + '/conversations/{}'


async def get_conversation_link(
    service: GitService, conversation_id: str | None, body: str
) -> str:
    """Appends a followup link, in the PR body, to the OpenHands conversation that opened the PR"""
    if get_global_config().app_mode != AppMode.SAAS:
        return body

    if not conversation_id:
        return body

    user = await service.get_user()
    username = user.login
    conversation_url = CONVERSATION_URL.format(conversation_id)
    conversation_link = (
        f'@{username} can click here to [continue refining the PR]({conversation_url})'
    )
    body += f'\n\n{conversation_link}'
    return body


async def save_pr_metadata(
    user_id: str | None, conversation_id: str, tool_result: str
) -> None:
    # Manually construct state for background operation (no request context available)
    state = InjectorState()
    setattr(state, USER_CONTEXT_ATTR, SpecifyUserContext(user_id))
    async with get_app_conversation_info_service(
        state
    ) as app_conversation_info_service:
        app_conversation_info = (
            await app_conversation_info_service.get_app_conversation_info(
                UUID(conversation_id)
            )
        )
        if not app_conversation_info:
            raise ToolError(f'No such conversation {conversation_id}')

        pull_pattern = r'pull/(\d+)'
        merge_request_pattern = r'merge_requests/(\d+)'
        pull_requests_pattern = r'pull-requests/(\d+)'

        # Check if the tool_result contains the PR number
        pr_number = None
        match_pull = re.search(pull_pattern, tool_result)
        match_merge_request = re.search(merge_request_pattern, tool_result)
        match_pull_requests = re.search(pull_requests_pattern, tool_result)

        if match_pull:
            pr_number = int(match_pull.group(1))
        elif match_merge_request:
            pr_number = int(match_merge_request.group(1))
        elif match_pull_requests:
            pr_number = int(match_pull_requests.group(1))

        if pr_number:
            logger.info(
                f'Saving PR number: {pr_number} for conversation {conversation_id}'
            )
            app_conversation_info.pr_number.append(pr_number)
        else:
            logger.warning(
                f'Failed to extract PR number for conversation {conversation_id}'
            )

        await app_conversation_info_service.save_app_conversation_info(
            app_conversation_info
        )


@mcp_server.tool()
async def create_pr(
    repo_name: Annotated[
        str, Field(description='GitHub repository ({{owner}}/{{repo}})')
    ],
    source_branch: Annotated[str, Field(description='Source branch on repo')],
    target_branch: Annotated[str, Field(description='Target branch on repo')],
    title: Annotated[str, Field(description='PR Title')],
    body: Annotated[str | None, Field(description='PR body')],
    draft: Annotated[bool, Field(description='Whether PR opened is a draft')] = True,
    labels: Annotated[
        list[str] | None,
        Field(
            description='Optional labels to apply to the PR. If labels are provided, they must be selected from the repository’s existing labels. Do not invent new ones. If the repository’s labels are not known, fetch them first.'
        ),
    ] = None,
) -> str:
    """Open a PR in GitHub"""
    logger.info('Calling OpenHands MCP create_pr')

    request = get_http_request()
    headers = request.headers
    conversation_id = headers.get('X-OpenHands-ServerConversation-ID', None)

    provider_tokens = await get_provider_tokens(request)
    access_token = await get_access_token(request)
    user_id = await get_user_id(request)

    github_token = (
        provider_tokens.get(ProviderType.GITHUB, ProviderToken())
        if provider_tokens
        else ProviderToken()
    )

    github_service = GithubServiceImpl(
        user_id=github_token.user_id,
        external_auth_id=user_id,
        external_auth_token=access_token,
        token=github_token.token,
        base_domain=github_token.host,
    )

    try:
        body = await get_conversation_link(github_service, conversation_id, body or '')
    except Exception as e:
        logger.warning(f'Failed to append conversation link: {e}')

    try:
        response = await github_service.create_pr(
            repo_name=repo_name,
            source_branch=source_branch,
            target_branch=target_branch,
            title=title,
            body=body,
            draft=draft,
            labels=labels,
        )

        if conversation_id:
            await save_pr_metadata(user_id, conversation_id, response)

    except Exception as e:
        error = f'Error creating pull request: {e}'
        raise ToolError(str(error))

    return response


@mcp_server.tool()
async def create_mr(
    id: Annotated[
        int | str,
        Field(description='GitLab repository (ID or URL-encoded path of the project)'),
    ],
    source_branch: Annotated[str, Field(description='Source branch on repo')],
    target_branch: Annotated[str, Field(description='Target branch on repo')],
    title: Annotated[
        str,
        Field(
            description='MR Title. Start title with `DRAFT:` or `WIP:` if applicable.'
        ),
    ],
    description: Annotated[str | None, Field(description='MR description')],
    labels: Annotated[
        list[str] | None,
        Field(
            description='Optional labels to apply to the MR. If labels are provided, they must be selected from the repository’s existing labels. Do not invent new ones. If the repository’s labels are not known, fetch them first.'
        ),
    ] = None,
) -> str:
    """Open a MR in GitLab"""
    logger.info('Calling OpenHands MCP create_mr')

    request = get_http_request()
    headers = request.headers
    conversation_id = headers.get('X-OpenHands-ServerConversation-ID', None)

    provider_tokens = await get_provider_tokens(request)
    access_token = await get_access_token(request)
    user_id = await get_user_id(request)

    github_token = (
        provider_tokens.get(ProviderType.GITLAB, ProviderToken())
        if provider_tokens
        else ProviderToken()
    )

    gitlab_service = GitLabServiceImpl(
        user_id=github_token.user_id,
        external_auth_id=user_id,
        external_auth_token=access_token,
        token=github_token.token,
        base_domain=github_token.host,
    )

    try:
        description = await get_conversation_link(
            gitlab_service, conversation_id, description or ''
        )
    except Exception as e:
        logger.warning(f'Failed to append conversation link: {e}')

    try:
        response = await gitlab_service.create_mr(
            id=id,
            source_branch=source_branch,
            target_branch=target_branch,
            title=title,
            description=description,
            labels=labels,
        )

        if conversation_id:
            await save_pr_metadata(user_id, conversation_id, response)

    except Exception as e:
        error = f'Error creating merge request: {e}'
        raise ToolError(str(error))

    return response


@mcp_server.tool()
async def create_bitbucket_pr(
    repo_name: Annotated[
        str, Field(description='Bitbucket repository (workspace/repo_slug)')
    ],
    source_branch: Annotated[str, Field(description='Source branch on repo')],
    target_branch: Annotated[str, Field(description='Target branch on repo')],
    title: Annotated[
        str,
        Field(
            description='PR Title. Start title with `DRAFT:` or `WIP:` if applicable.'
        ),
    ],
    description: Annotated[str | None, Field(description='PR description')],
) -> str:
    """Open a PR in Bitbucket"""
    logger.info('Calling OpenHands MCP create_bitbucket_pr')

    request = get_http_request()
    headers = request.headers
    conversation_id = headers.get('X-OpenHands-ServerConversation-ID', None)

    provider_tokens = await get_provider_tokens(request)
    access_token = await get_access_token(request)
    user_id = await get_user_id(request)

    bitbucket_token = (
        provider_tokens.get(ProviderType.BITBUCKET, ProviderToken())
        if provider_tokens
        else ProviderToken()
    )

    bitbucket_service = BitBucketServiceImpl(
        user_id=bitbucket_token.user_id,
        external_auth_id=user_id,
        external_auth_token=access_token,
        token=bitbucket_token.token,
        base_domain=bitbucket_token.host,
    )

    try:
        description = await get_conversation_link(
            bitbucket_service, conversation_id, description or ''
        )
    except Exception as e:
        logger.warning(f'Failed to append conversation link: {e}')

    try:
        response = await bitbucket_service.create_pr(
            repo_name=repo_name,
            source_branch=source_branch,
            target_branch=target_branch,
            title=title,
            body=description,
        )

        if conversation_id:
            await save_pr_metadata(user_id, conversation_id, response)

    except Exception as e:
        error = f'Error creating pull request: {e}'
        logger.error(error)
        raise ToolError(str(error))

    return response


@mcp_server.tool()
async def create_bitbucket_data_center_pr(
    repo_name: Annotated[
        str, Field(description='Bitbucket Data Center repository (PROJECT/repo_slug)')
    ],
    source_branch: Annotated[str, Field(description='Source branch on repo')],
    target_branch: Annotated[str, Field(description='Target branch on repo')],
    title: Annotated[
        str,
        Field(
            description='PR Title. Start title with `DRAFT:` or `WIP:` if applicable.'
        ),
    ],
    description: Annotated[str | None, Field(description='PR description')],
) -> str:
    """Open a PR in Bitbucket Data Center"""
    logger.info('Calling OpenHands MCP create_bitbucket_data_center_pr')

    request = get_http_request()
    headers = request.headers
    conversation_id = headers.get('X-OpenHands-ServerConversation-ID', None)

    provider_tokens = await get_provider_tokens(request)
    access_token = await get_access_token(request)
    user_id = await get_user_id(request)

    bitbucket_dc_token = (
        provider_tokens.get(ProviderType.BITBUCKET_DATA_CENTER, ProviderToken())
        if provider_tokens
        else ProviderToken()
    )

    bitbucket_dc_service = BitbucketDCServiceImpl(
        user_id=bitbucket_dc_token.user_id,
        external_auth_id=user_id,
        external_auth_token=access_token,
        token=bitbucket_dc_token.token,
        base_domain=bitbucket_dc_token.host,
    )

    try:
        description = await get_conversation_link(
            bitbucket_dc_service, conversation_id, description or ''
        )
    except Exception as e:
        logger.warning(f'Failed to append conversation link: {e}')

    try:
        response = await bitbucket_dc_service.create_pr(
            repo_name=repo_name,
            source_branch=source_branch,
            target_branch=target_branch,
            title=title,
            body=description,
        )

        if conversation_id:
            await save_pr_metadata(user_id, conversation_id, response)

    except Exception as e:
        error = f'Error creating pull request: {e}'
        logger.error(error)
        raise ToolError(str(error))

    return response


@mcp_server.tool()
async def create_azure_devops_pr(
    repo_name: Annotated[
        str, Field(description='Azure DevOps repository (organization/project/repo)')
    ],
    source_branch: Annotated[str, Field(description='Source branch on repo')],
    target_branch: Annotated[str, Field(description='Target branch on repo')],
    title: Annotated[
        str,
        Field(
            description='PR Title. Start title with `DRAFT:` or `WIP:` if applicable.'
        ),
    ],
    description: Annotated[str | None, Field(description='PR description')],
) -> str:
    """Open a PR in Azure DevOps"""
    logger.info('Calling OpenHands MCP create_azure_devops_pr')

    request = get_http_request()
    headers = request.headers
    conversation_id = headers.get('X-OpenHands-ServerConversation-ID', None)

    provider_tokens = await get_provider_tokens(request)
    access_token = await get_access_token(request)
    user_id = await get_user_id(request)

    azure_devops_token = (
        provider_tokens.get(ProviderType.AZURE_DEVOPS, ProviderToken())
        if provider_tokens
        else ProviderToken()
    )

    azure_devops_service = AzureDevOpsServiceImpl(
        user_id=azure_devops_token.user_id,
        external_auth_id=user_id,
        external_auth_token=access_token,
        token=azure_devops_token.token,
        base_domain=azure_devops_token.host,
    )

    try:
        description = await get_conversation_link(
            azure_devops_service, conversation_id, description or ''
        )
    except Exception as e:
        logger.warning(f'Failed to append conversation link: {e}')

    try:
        response = await azure_devops_service.create_pr(
            repo_name=repo_name,
            source_branch=source_branch,
            target_branch=target_branch,
            title=title,
            body=description,
        )

        if conversation_id and user_id:
            await save_pr_metadata(user_id, conversation_id, response)

    except Exception as e:
        error = f'Error creating pull request: {e}'
        logger.error(error)
        raise ToolError(str(error))

    return response
