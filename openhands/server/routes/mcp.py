import os
import re
from typing import Annotated

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_http_request
from pydantic import Field

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.azure_devops.azure_devops_service import (
    AzureDevOpsServiceImpl,
)
from openhands.integrations.bitbucket.bitbucket_service import BitBucketServiceImpl
from openhands.integrations.github.github_service import GithubServiceImpl
from openhands.integrations.gitlab.gitlab_service import GitLabServiceImpl
from openhands.integrations.provider import ProviderToken
from openhands.integrations.service_types import GitService, ProviderType
from openhands.server.shared import ConversationStoreImpl, config, server_config
from openhands.server.types import AppMode
from openhands.server.user_auth import (
    get_access_token,
    get_provider_tokens,
    get_user_id,
)
from openhands.storage.data_models.conversation_metadata import ConversationMetadata

mcp_server = FastMCP('mcp', mask_error_details=True)

HOST = f'https://{os.getenv("WEB_HOST", "app.all-hands.dev").strip()}'
CONVERSATION_URL = HOST + '/conversations/{}'


async def get_conversation_link(
    service: GitService, conversation_id: str | None, body: str
) -> str:
    """Appends a followup link, in the PR body, to the OpenHands conversation that opened the PR"""
    if server_config.app_mode != AppMode.SAAS:
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
    conversation_store = await ConversationStoreImpl.get_instance(config, user_id)
    conversation: ConversationMetadata = await conversation_store.get_metadata(
        conversation_id
    )

    pull_pattern = r'pull/(\d+)'
    merge_request_pattern = r'merge_requests/(\d+)'

    # Check if the tool_result contains the PR number
    pr_number = None
    match_pull = re.search(pull_pattern, tool_result)
    match_merge_request = re.search(merge_request_pattern, tool_result)

    if match_pull:
        pr_number = int(match_pull.group(1))
    elif match_merge_request:
        pr_number = int(match_merge_request.group(1))

    if pr_number:
        logger.info(f'Saving PR number: {pr_number} for conversation {conversation_id}')
        conversation.pr_number.append(pr_number)
    else:
        logger.warning(
            f'Failed to extract PR number for conversation {conversation_id}'
        )

    await conversation_store.save_metadata(conversation)


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
