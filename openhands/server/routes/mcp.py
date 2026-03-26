import asyncio
import json
import os
import re
from typing import Annotated

import httpx
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
    service: GitService, conversation_id: str, body: str
) -> str:
    """Appends a followup link, in the PR body, to the OpenHands conversation that opened the PR"""
    if server_config.app_mode != AppMode.SAAS:
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


def _get_app_server_base_url(request) -> str:
    """Derive the app server base URL from the incoming MCP request.

    Tries the request's Host header first, then falls back to environment.
    """
    host_header = request.headers.get('host')
    scheme = request.url.scheme if hasattr(request.url, 'scheme') else 'http'
    if host_header:
        return f'{scheme}://{host_header}'
    # Fallback for local dev
    port = os.getenv('UVICORN_PORT', '3000')
    return f'http://localhost:{port}'


def _build_forwarding_headers(request) -> dict[str, str]:
    """Build headers to forward from the MCP request to the app server API."""
    headers: dict[str, str] = {}
    for key in ('authorization', 'cookie', 'x-session-api-key'):
        val = request.headers.get(key)
        if val:
            headers[key] = val
    return headers


@mcp_server.tool()
async def create_conversation(
    initial_message: Annotated[
        str,
        Field(
            description='The initial message/task to send to the new conversation. '
            'This is the prompt that the new conversation will start working on.'
        ),
    ],
    title: Annotated[
        str | None,
        Field(description='Optional title for the new conversation'),
    ] = None,
    selected_repository: Annotated[
        str | None,
        Field(
            description='Optional git repository to connect the conversation to (e.g. "owner/repo"). '
            "If not provided, the conversation inherits the parent's repository."
        ),
    ] = None,
    selected_branch: Annotated[
        str | None,
        Field(
            description='Optional git branch to use. Only applies when selected_repository is set.'
        ),
    ] = None,
    environment_url: Annotated[
        str | None,
        Field(
            description='Optional URL of a remote environment to connect to instead of a repository. '
            'Mutually exclusive with selected_repository.'
        ),
    ] = None,
    system_message_suffix: Annotated[
        str | None,
        Field(
            description='Optional additional system prompt text appended to the default system message. '
            'Use this to give the sub-conversation specific instructions, constraints, or context '
            'beyond what is in the initial_message.'
        ),
    ] = None,
    wait_for_completion: Annotated[
        bool,
        Field(
            description='If true, wait for the new conversation to finish and return its result. '
            'If false, return immediately with the conversation ID (fire-and-forget).'
        ),
    ] = False,
    wait_timeout_seconds: Annotated[
        int,
        Field(
            description='Maximum seconds to wait for completion when wait_for_completion is true. '
            'Ignored if wait_for_completion is false.',
            ge=10,
            le=3600,
        ),
    ] = 300,
) -> str:
    """Launch a new sub-conversation from the current conversation.

    Creates a new conversation that inherits LLM configuration from the current
    conversation. By default it shares the parent's sandbox, but you can optionally
    connect it to a specific repository or a remote environment instead.

    Use this when you want to delegate a task to a separate conversation, for example:
    - Running a parallel investigation
    - Performing a sub-task that should have its own conversation history
    - Spawning a background task while continuing the current work

    Returns a JSON object with the conversation ID and status information.
    """
    logger.info('Calling OpenHands MCP create_conversation')

    if selected_repository and environment_url:
        raise ToolError(
            'Cannot specify both selected_repository and environment_url. '
            'Choose one: connect to a repository or to an environment.'
        )

    request = get_http_request()
    parent_conversation_id = request.headers.get(
        'X-OpenHands-ServerConversation-ID', None
    )

    if not parent_conversation_id:
        raise ToolError(
            'Cannot create sub-conversation: no parent conversation ID found. '
            'This tool must be called from within an active conversation.'
        )

    base_url = _get_app_server_base_url(request)
    fwd_headers = _build_forwarding_headers(request)
    fwd_headers['Content-Type'] = 'application/json'

    # Build the start request payload
    payload: dict = {
        'parent_conversation_id': parent_conversation_id,
        'initial_message': {
            'role': 'user',
            'content': [{'type': 'text', 'text': initial_message}],
        },
    }
    if title:
        payload['title'] = title
    if selected_repository:
        payload['selected_repository'] = selected_repository
        if selected_branch:
            payload['selected_branch'] = selected_branch
    if environment_url:
        payload['environment_url'] = environment_url
    if system_message_suffix:
        payload['system_message_suffix'] = system_message_suffix

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Start the conversation
            resp = await client.post(
                f'{base_url}/api/v1/app-conversations',
                json=payload,
                headers=fwd_headers,
            )
            resp.raise_for_status()
            start_task = resp.json()
    except httpx.HTTPStatusError as e:
        raise ToolError(
            f'Failed to create conversation: HTTP {e.response.status_code} - {e.response.text}'
        )
    except Exception as e:
        raise ToolError(f'Failed to create conversation: {e}')

    task_id = start_task.get('id')
    status = start_task.get('status', 'UNKNOWN')
    conversation_id = start_task.get('app_conversation_id')

    if not wait_for_completion:
        # Fire-and-forget: return immediately with the task info
        return json.dumps(
            {
                'mode': 'fire_and_forget',
                'task_id': task_id,
                'status': status,
                'conversation_id': conversation_id,
                'message': 'Conversation creation started. Use get_conversation_status to check progress.',
            }
        )

    # Wait mode: poll the start task until READY or ERROR
    poll_interval = 3
    elapsed = 0

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            while elapsed < wait_timeout_seconds:
                if status == 'READY':
                    break
                if status == 'ERROR':
                    detail = start_task.get('detail', 'Unknown error')
                    return json.dumps(
                        {
                            'mode': 'wait',
                            'task_id': task_id,
                            'status': 'ERROR',
                            'conversation_id': conversation_id,
                            'error': detail,
                        }
                    )

                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

                # Poll start task status
                resp = await client.get(
                    f'{base_url}/api/v1/app-conversations/start-tasks',
                    params={'ids': task_id},
                    headers=fwd_headers,
                )
                resp.raise_for_status()
                tasks = resp.json()
                if tasks and tasks[0]:
                    start_task = tasks[0]
                    status = start_task.get('status', 'UNKNOWN')
                    conversation_id = start_task.get('app_conversation_id')

            if status != 'READY':
                return json.dumps(
                    {
                        'mode': 'wait',
                        'task_id': task_id,
                        'status': 'TIMEOUT',
                        'conversation_id': conversation_id,
                        'message': f'Conversation startup did not complete within {wait_timeout_seconds}s. '
                        f'Last status: {status}. Use get_conversation_status to check progress.',
                    }
                )

            # Conversation is READY. Now poll execution_status until finished.
            assert conversation_id is not None
            while elapsed < wait_timeout_seconds:
                resp = await client.get(
                    f'{base_url}/api/v1/app-conversations',
                    params={'ids': conversation_id},
                    headers=fwd_headers,
                )
                resp.raise_for_status()
                conversations = resp.json()
                if conversations and conversations[0]:
                    conv = conversations[0]
                    exec_status = conv.get('execution_status')
                    if exec_status in ('finished', 'idle', 'error', 'stuck'):
                        return json.dumps(
                            {
                                'mode': 'wait',
                                'task_id': task_id,
                                'status': 'COMPLETED',
                                'conversation_id': conversation_id,
                                'execution_status': exec_status,
                                'title': conv.get('title'),
                            }
                        )

                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

            return json.dumps(
                {
                    'mode': 'wait',
                    'task_id': task_id,
                    'status': 'TIMEOUT',
                    'conversation_id': conversation_id,
                    'message': f'Conversation did not complete within {wait_timeout_seconds}s. '
                    'Use get_conversation_status to check progress.',
                }
            )
    except Exception as e:
        raise ToolError(f'Error while waiting for conversation: {e}')


@mcp_server.tool()
async def get_conversation_status(
    conversation_id: Annotated[
        str,
        Field(
            description='The conversation ID to check status for. '
            'This is the conversation_id returned by create_conversation.'
        ),
    ],
    include_messages: Annotated[
        bool,
        Field(
            description='If true, include the conversation messages in the response. '
            'Messages are returned in chronological order.'
        ),
    ] = True,
    message_limit: Annotated[
        int,
        Field(
            description='Maximum number of messages to return when include_messages is true. '
            'Ignored if include_messages is false.',
            ge=1,
            le=100,
        ),
    ] = 50,
) -> str:
    """Check the status of a conversation.

    Returns the current execution status and metadata of the specified conversation.
    Optionally includes the conversation messages.
    Useful for checking on conversations created with create_conversation in
    fire-and-forget mode, or for monitoring any conversation's progress.
    """
    logger.info(f'Calling OpenHands MCP get_conversation_status for {conversation_id}')

    request = get_http_request()
    base_url = _get_app_server_base_url(request)
    fwd_headers = _build_forwarding_headers(request)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f'{base_url}/api/v1/app-conversations',
                params={'ids': conversation_id},
                headers=fwd_headers,
            )
            resp.raise_for_status()
            conversations = resp.json()

            if not conversations or not conversations[0]:
                return json.dumps(
                    {
                        'conversation_id': conversation_id,
                        'status': 'NOT_FOUND',
                        'message': 'Conversation not found. It may still be starting up. '
                        'If you just created it, try again in a few seconds.',
                    }
                )

            conv = conversations[0]
            result: dict = {
                'conversation_id': conversation_id,
                'execution_status': conv.get('execution_status'),
                'title': conv.get('title'),
                'selected_repository': conv.get('selected_repository'),
                'selected_branch': conv.get('selected_branch'),
                'sandbox_status': conv.get('sandbox_status'),
            }

            if include_messages:
                conversation_url = conv.get('conversation_url')
                if conversation_url:
                    # Use the V1 messages endpoint which proxies to the agent server
                    events_resp = await client.get(
                        f'{base_url}/api/v1/app-conversations/{conversation_id}/messages',
                        params={'limit': message_limit},
                        headers=fwd_headers,
                    )
                    events_resp.raise_for_status()
                    result['messages'] = events_resp.json()
                else:
                    result['messages'] = []
                    result['messages_error'] = (
                        'Conversation sandbox is not running. '
                        'Messages are only available when the sandbox is active.'
                    )

            return json.dumps(result)
    except httpx.HTTPStatusError as e:
        raise ToolError(
            f'Failed to get conversation status: HTTP {e.response.status_code} - {e.response.text}'
        )
    except Exception as e:
        raise ToolError(f'Failed to get conversation status: {e}')
