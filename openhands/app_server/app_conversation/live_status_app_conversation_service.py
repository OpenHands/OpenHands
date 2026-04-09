import asyncio
import json
import logging
import os
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Sequence, cast
from uuid import UUID, uuid4

import httpx
from fastapi import Request
from pydantic import Field, SecretStr, TypeAdapter

from openhands.agent_server.models import (
    ConversationInfo,
    SendMessageRequest,
    StartConversationRequest,
    TextContent,
)
from openhands.app_server.app_conversation.app_conversation_info_service import (
    AppConversationInfoService,
)
from openhands.app_server.app_conversation.app_conversation_models import (
    AgentType,
    AppConversation,
    AppConversationInfo,
    AppConversationPage,
    AppConversationSortOrder,
    AppConversationStartRequest,
    AppConversationStartTask,
    AppConversationStartTaskStatus,
    AppConversationUpdateRequest,
    PluginSpec,
)
from openhands.app_server.app_conversation.app_conversation_service import (
    AppConversationService,
    AppConversationServiceInjector,
)
from openhands.app_server.app_conversation.app_conversation_service_base import (
    AppConversationServiceBase,
    get_project_dir,
)
from openhands.app_server.app_conversation.app_conversation_start_task_service import (
    AppConversationStartTaskService,
)
from openhands.app_server.app_conversation.hook_loader import (
    load_hooks_from_agent_server,
)
from openhands.app_server.app_conversation.sql_app_conversation_info_service import (
    SQLAppConversationInfoService,
)
from openhands.app_server.config import get_event_callback_service
from openhands.app_server.errors import SandboxError
from openhands.app_server.event.event_service import EventService
from openhands.app_server.event_callback.event_callback_models import EventCallback
from openhands.app_server.event_callback.event_callback_service import (
    EventCallbackService,
)
from openhands.app_server.event_callback.set_title_callback_processor import (
    SetTitleCallbackProcessor,
)
from openhands.app_server.pending_messages.pending_message_service import (
    PendingMessageService,
)
from openhands.app_server.sandbox.docker_sandbox_service import DockerSandboxService
from openhands.app_server.sandbox.sandbox_models import (
    AGENT_SERVER,
    SandboxInfo,
    SandboxStatus,
)
from openhands.app_server.sandbox.sandbox_service import SandboxService
from openhands.app_server.sandbox.sandbox_spec_service import SandboxSpecService
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.services.jwt_service import JwtService
from openhands.app_server.user.user_context import UserContext
from openhands.app_server.user.user_models import UserInfo
from openhands.app_server.utils.docker_utils import (
    replace_localhost_hostname_for_docker,
)
from openhands.app_server.utils.llm_metadata import (
    get_llm_metadata,
    should_set_litellm_extra_body,
)
from openhands.integrations.provider import PROVIDER_TOKEN_TYPE, ProviderType
from openhands.integrations.service_types import SuggestedTask
from openhands.sdk import Agent, AgentContext, LocalWorkspace
from openhands.sdk.hooks import HookConfig
from openhands.sdk.llm import LLM
from openhands.sdk.plugin import PluginSource
from openhands.sdk.secret import LookupSecret, SecretValue, StaticSecret
from openhands.sdk.utils.paging import page_iterator
from openhands.utils._redact_compat import sanitize_config
from openhands.sdk.workspace.remote.async_remote_workspace import AsyncRemoteWorkspace
from openhands.server.types import AppMode
from openhands.storage.data_models.conversation_metadata import ConversationTrigger
from openhands.storage.data_models.settings import SandboxGroupingStrategy
from openhands.tools.preset.default import (
    get_default_tools,
)
from openhands.tools.preset.planning import (
    format_plan_structure,
    get_planning_tools,
)
from openhands.utils.git import ensure_valid_git_branch_name

_conversation_info_type_adapter = TypeAdapter(list[ConversationInfo | None])
_logger = logging.getLogger(__name__)

# Planning agent instruction to prevent "Ready to proceed?" behavior
PLANNING_AGENT_INSTRUCTION = """<IMPORTANT_PLANNING_BOUNDARIES>
You are a Planning Agent that can ONLY create plans - you CANNOT execute code or make changes.

After you finalize the plan in PLAN.md:
- Do NOT ask "Ready to proceed?" or offer to execute the plan
- Do NOT attempt to run any implementation commands
- Instead, inform the user they have two options to proceed:
  1. Click the **Build** button below the plan preview - this will automatically switch to the code agent and instruct it to execute the plan
  2. Switch to the code agent manually (click the agent selector button or press Shift+Tab), then send a message instructing it to execute the plan

Your role ends when the plan is finalized. Implementation is handled by the code agent.
</IMPORTANT_PLANNING_BOUNDARIES>"""


@dataclass
class LiveStatusAppConversationService(AppConversationServiceBase):
    """AppConversationService which combines live status info from the sandbox with stored data."""

    user_context: UserContext
    app_conversation_info_service: AppConversationInfoService
    app_conversation_start_task_service: AppConversationStartTaskService
    event_callback_service: EventCallbackService
    event_service: EventService
    sandbox_service: SandboxService
    sandbox_spec_service: SandboxSpecService
    jwt_service: JwtService
    pending_message_service: PendingMessageService
    sandbox_startup_timeout: int
    sandbox_startup_poll_frequency: int
    max_num_conversations_per_sandbox: int
    httpx_client: httpx.AsyncClient
    web_url: str | None
    openhands_provider_base_url: str | None
    access_token_hard_timeout: timedelta | None
    app_mode: str | None = None
    tavily_api_key: str | None = None

    async def _get_sandbox_grouping_strategy(self) -> SandboxGroupingStrategy:
        """Get the sandbox grouping strategy from user settings."""
        user_info = await self.user_context.get_user_info()
        return user_info.sandbox_grouping_strategy

    async def search_app_conversations(
        self,
        title__contains: str | None = None,
        created_at__gte: datetime | None = None,
        created_at__lt: datetime | None = None,
        updated_at__gte: datetime | None = None,
        updated_at__lt: datetime | None = None,
        sandbox_id__eq: str | None = None,
        sort_order: AppConversationSortOrder = AppConversationSortOrder.CREATED_AT_DESC,
        page_id: str | None = None,
        limit: int = 20,
        include_sub_conversations: bool = False,
    ) -> AppConversationPage:
        """Search for sandboxed conversations."""
        page = await self.app_conversation_info_service.search_app_conversation_info(
            title__contains=title__contains,
            created_at__gte=created_at__gte,
            created_at__lt=created_at__lt,
            updated_at__gte=updated_at__gte,
            updated_at__lt=updated_at__lt,
            sandbox_id__eq=sandbox_id__eq,
            sort_order=sort_order,
            page_id=page_id,
            limit=limit,
            include_sub_conversations=include_sub_conversations,
        )
        conversations: list[AppConversation] = await self._build_app_conversations(
            page.items
        )  # type: ignore
        return AppConversationPage(items=conversations, next_page_id=page.next_page_id)

    async def count_app_conversations(
        self,
        title__contains: str | None = None,
        created_at__gte: datetime | None = None,
        created_at__lt: datetime | None = None,
        updated_at__gte: datetime | None = None,
        updated_at__lt: datetime | None = None,
        sandbox_id__eq: str | None = None,
    ) -> int:
        return await self.app_conversation_info_service.count_app_conversation_info(
            title__contains=title__contains,
            created_at__gte=created_at__gte,
            created_at__lt=created_at__lt,
            updated_at__gte=updated_at__gte,
            updated_at__lt=updated_at__lt,
            sandbox_id__eq=sandbox_id__eq,
        )

    async def get_app_conversation(
        self, conversation_id: UUID
    ) -> AppConversation | None:
        info = await self.app_conversation_info_service.get_app_conversation_info(
            conversation_id
        )
        result = await self._build_app_conversations([info])
        return result[0]

    async def batch_get_app_conversations(
        self, conversation_ids: list[UUID]
    ) -> list[AppConversation | None]:
        info = await self.app_conversation_info_service.batch_get_app_conversation_info(
            conversation_ids
        )
        conversations = await self._build_app_conversations(info)
        return conversations

    async def start_app_conversation(
        self, request: AppConversationStartRequest
    ) -> AsyncGenerator[AppConversationStartTask, None]:
        async for task in self._start_app_conversation(request):
            await self.app_conversation_start_task_service.save_app_conversation_start_task(
                task
            )
            yield task

    async def _start_app_conversation(
        self, request: AppConversationStartRequest
    ) -> AsyncGenerator[AppConversationStartTask, None]:
        # Create and yield the start task
        user_id = await self.user_context.get_user_id()

        # Validate and inherit from parent conversation if provided
        if request.parent_conversation_id:
            parent_info = (
                await self.app_conversation_info_service.get_app_conversation_info(
                    request.parent_conversation_id
                )
            )
            if parent_info is None:
                raise ValueError(
                    f'Parent conversation not found: {request.parent_conversation_id}'
                )
            self._inherit_configuration_from_parent(request, parent_info)

        self._apply_suggested_task(request)

        task = AppConversationStartTask(
            created_by_user_id=user_id,
            request=request,
        )
        yield task

        try:
            async for updated_task in self._wait_for_sandbox_start(task):
                yield updated_task

            # Get the sandbox
            sandbox_id = task.sandbox_id
            assert sandbox_id is not None
            sandbox = await self.sandbox_service.get_sandbox(sandbox_id)
            assert sandbox is not None
            agent_server_url = self._get_agent_server_url(sandbox)

            # Get the working dir
            sandbox_spec = await self.sandbox_spec_service.get_sandbox_spec(
                sandbox.sandbox_spec_id
            )
            assert sandbox_spec is not None

            # Set up conversation id
            conversation_id = request.conversation_id or uuid4()

            # Setup working dir based on grouping
            working_dir = sandbox_spec.working_dir
            sandbox_grouping_strategy = await self._get_sandbox_grouping_strategy()
            if sandbox_grouping_strategy != SandboxGroupingStrategy.NO_GROUPING:
                working_dir = f'{working_dir}/{conversation_id.hex}'

            # Run setup scripts
            remote_workspace = AsyncRemoteWorkspace(
                host=agent_server_url,
                api_key=sandbox.session_api_key,
                working_dir=working_dir,
            )
            async for updated_task in self.run_setup_scripts(
                task, sandbox, remote_workspace, agent_server_url
            ):
                yield updated_task

            # Build the start request
            start_conversation_request = (
                await self._build_start_conversation_request_for_user(
                    sandbox,
                    conversation_id,
                    request.initial_message,
                    request.system_message_suffix,
                    request.git_provider,
                    working_dir,
                    request.agent_type,
                    request.llm_model,
                    remote_workspace=remote_workspace,
                    selected_repository=request.selected_repository,
                    plugins=request.plugins,
                )
            )

            # update status
            task.status = AppConversationStartTaskStatus.STARTING_CONVERSATION
            task.agent_server_url = agent_server_url
            yield task

            # Start conversation...
            body_json = start_conversation_request.model_dump(
                mode='json', context={'expose_secrets': True}
            )
            # Log hook_config to verify it's being passed
            hook_config_in_request = body_json.get('hook_config')
            _logger.debug(
                f'Sending StartConversationRequest with hook_config: '
                f'{hook_config_in_request}'
            )
            headers = (
                {'X-Session-API-Key': sandbox.session_api_key}
                if sandbox.session_api_key
                else {}
            )
            response = await self.httpx_client.post(
                f'{agent_server_url}/api/conversations',
                json=body_json,
                headers=headers,
                timeout=self.sandbox_startup_timeout,
            )

            response.raise_for_status()
            info = ConversationInfo.model_validate(response.json())

            # Store info...
            user_id = await self.user_context.get_user_id()
            app_conversation_info = AppConversationInfo(
                id=info.id,
                title=f'Conversation {info.id.hex[:5]}',
                sandbox_id=sandbox.id,
                created_by_user_id=user_id,
                llm_model=start_conversation_request.agent.llm.model,
                # Git parameters
                selected_repository=request.selected_repository,
                selected_branch=request.selected_branch,
                git_provider=request.git_provider,
                trigger=request.trigger,
                pr_number=request.pr_number,
                parent_conversation_id=request.parent_conversation_id,
            )
            await self.app_conversation_info_service.save_app_conversation_info(
                app_conversation_info
            )

            # Setup default processors
            processors = request.processors or []

            # Always ensure SetTitleCallbackProcessor is included
            has_set_title_processor = any(
                isinstance(processor, SetTitleCallbackProcessor)
                for processor in processors
            )
            if not has_set_title_processor:
                processors.append(SetTitleCallbackProcessor())

            # Save processors
            for processor in processors:
                await self.event_callback_service.save_event_callback(
                    EventCallback(
                        conversation_id=info.id,
                        processor=processor,
                    )
                )

            # Set security analyzer from settings
            user = await self.user_context.get_user_info()
            await self._set_security_analyzer_from_settings(
                agent_server_url,
                sandbox.session_api_key,
                info.id,
                user.security_analyzer,
                self.httpx_client,
            )

            # Update the start task
            task.status = AppConversationStartTaskStatus.READY
            task.app_conversation_id = info.id
            yield task

            # Process any pending messages queued while waiting for conversation
            if sandbox.session_api_key:
                await self._process_pending_messages(
                    task_id=task.id,
                    conversation_id=info.id,
                    agent_server_url=agent_server_url,
                    session_api_key=sandbox.session_api_key,
                )

        except Exception as exc:
            _logger.exception('Error starting conversation', stack_info=True)
            task.status = AppConversationStartTaskStatus.ERROR
            task.detail = str(exc)
            yield task

    async def _build_app_conversations(
        self, app_conversation_infos: Sequence[AppConversationInfo | None]
    ) -> list[AppConversation | None]:
        sandbox_id_to_conversation_ids = self._get_sandbox_id_to_conversation_ids(
            app_conversation_infos
        )

        # Get referenced sandboxes in a single batch operation...
        sandboxes = await self.sandbox_service.batch_get_sandboxes(
            list(sandbox_id_to_conversation_ids)
        )
        sandboxes_by_id = {sandbox.id: sandbox for sandbox in sandboxes if sandbox}

        # Gather the running conversations
        tasks = [
            self._get_live_conversation_info(
                sandbox, sandbox_id_to_conversation_ids.get(sandbox.id)
            )
            for sandbox in sandboxes
            if sandbox and sandbox.status == SandboxStatus.RUNNING
        ]
        if tasks:
            sandbox_conversation_infos = await asyncio.gather(*tasks)
        else:
            sandbox_conversation_infos = []

        # Collect the results into a single dictionary
        conversation_info_by_id = {}
        for conversation_infos in sandbox_conversation_infos:
            for conversation_info in conversation_infos:
                conversation_info_by_id[conversation_info.id] = conversation_info

        # Build app_conversation from info
        result = [
            (
                self._build_conversation(
                    app_conversation_info,
                    sandboxes_by_id.get(app_conversation_info.sandbox_id),
                    conversation_info_by_id.get(app_conversation_info.id),
                )
                if app_conversation_info
                else None
            )
            for app_conversation_info in app_conversation_infos
        ]

        return result

    async def _get_live_conversation_info(
        self,
        sandbox: SandboxInfo,
        conversation_ids: list[str],
    ) -> list[ConversationInfo]:
        """Get agent status for multiple conversations from the Agent Server."""
        try:
            # Build the URL with query parameters
            agent_server_url = self._get_agent_server_url(sandbox)
            url = f'{agent_server_url.rstrip("/")}/api/conversations'
            params = {'ids': conversation_ids}

            # Set up headers
            headers = {}
            if sandbox.session_api_key:
                headers['X-Session-API-Key'] = sandbox.session_api_key

            response = await self.httpx_client.get(url, params=params, headers=headers)
            response.raise_for_status()

            data = response.json()
            conversation_info = _conversation_info_type_adapter.validate_python(data)
            conversation_info = [c for c in conversation_info if c]
            return conversation_info
        except httpx.HTTPStatusError:
            # The runtime API stops idle sandboxes all the time and they return a 404 or a 503.
            # This is normal and should not be considered an error.
            _logger.warning(
                f'Error getting conversation status from sandbox {sandbox.id}',
                exc_info=True,
            )
            return []
        except Exception:
            # Not getting a status is not a fatal error - we just mark the conversation as stopped
            _logger.exception(
                f'Error getting conversation status from sandbox {sandbox.id}',
                stack_info=True,
            )
            return []

    def _build_conversation(
        self,
        app_conversation_info: AppConversationInfo | None,
        sandbox: SandboxInfo | None,
        conversation_info: ConversationInfo | None,
    ) -> AppConversation | None:
        if app_conversation_info is None:
            return None
        sandbox_status = sandbox.status if sandbox else SandboxStatus.MISSING
        execution_status = (
            conversation_info.execution_status if conversation_info else None
        )
        conversation_url = None
        session_api_key = None
        if sandbox and sandbox.exposed_urls:
            conversation_url = next(
                (
                    exposed_url.url
                    for exposed_url in sandbox.exposed_urls
                    if exposed_url.name == AGENT_SERVER
                ),
                None,
            )
            if conversation_url:
                conversation_url += f'/api/conversations/{app_conversation_info.id.hex}'
            session_api_key = sandbox.session_api_key

        return AppConversation(
            **app_conversation_info.model_dump(),
            sandbox_status=sandbox_status,
            execution_status=execution_status,
            conversation_url=conversation_url,
            session_api_key=session_api_key,
        )

    def _get_sandbox_id_to_conversation_ids(
        self, stored_conversations: Sequence[AppConversationInfo | None]
    ):
        result = defaultdict(list)
        for stored_conversation in stored_conversations:
            if stored_conversation:
                result[stored_conversation.sandbox_id].append(stored_conversation.id)
        return result

    async def _find_running_sandbox_for_user(self) -> SandboxInfo | None:
        """Find a running sandbox for the current user based on the grouping strategy.

        Returns:
            SandboxInfo if a running sandbox is found, None otherwise.
        """
        try:
            user_id = await self.user_context.get_user_id()
            sandbox_grouping_strategy = await self._get_sandbox_grouping_strategy()

            # If no grouping, return None to force creation of a new sandbox
            if sandbox_grouping_strategy == SandboxGroupingStrategy.NO_GROUPING:
                return None

            # Collect all running sandboxes for this user
            running_sandboxes = []
            page_id = None
            while True:
                page = await self.sandbox_service.search_sandboxes(
                    page_id=page_id, limit=100
                )

                for sandbox in page.items:
                    if (
                        sandbox.status == SandboxStatus.RUNNING
                        and sandbox.created_by_user_id == user_id
                    ):
                        running_sandboxes.append(sandbox)

                if page.next_page_id is None:
                    break
                page_id = page.next_page_id

            if not running_sandboxes:
                return None

            # Apply the grouping strategy
            return await self._select_sandbox_by_strategy(
                running_sandboxes, sandbox_grouping_strategy
            )

        except Exception as e:
            _logger.warning(
                f'Error finding running sandbox for user: {e}', exc_info=True
            )
            return None

    async def _select_sandbox_by_strategy(
        self,
        running_sandboxes: list[SandboxInfo],
        sandbox_grouping_strategy: SandboxGroupingStrategy,
    ) -> SandboxInfo | None:
        """Select a sandbox from the list based on the configured grouping strategy.

        Args:
            running_sandboxes: List of running sandboxes for the user
            sandbox_grouping_strategy: The strategy to use for selection

        Returns:
            Selected sandbox based on the strategy, or None if no sandbox is available
            (e.g., all sandboxes have reached max_num_conversations_per_sandbox)
        """
        # Get conversation counts for filtering by max_num_conversations_per_sandbox
        sandbox_conversation_counts = await self._get_conversation_counts_by_sandbox(
            [s.id for s in running_sandboxes]
        )

        # Filter out sandboxes that have reached the max number of conversations
        available_sandboxes = [
            s
            for s in running_sandboxes
            if sandbox_conversation_counts.get(s.id, 0)
            < self.max_num_conversations_per_sandbox
        ]

        if not available_sandboxes:
            # All sandboxes have reached the max - need to create a new one
            return None

        if sandbox_grouping_strategy == SandboxGroupingStrategy.ADD_TO_ANY:
            # Return the first available sandbox
            return available_sandboxes[0]

        elif sandbox_grouping_strategy == SandboxGroupingStrategy.GROUP_BY_NEWEST:
            # Return the most recently created sandbox
            return max(available_sandboxes, key=lambda s: s.created_at)

        elif sandbox_grouping_strategy == SandboxGroupingStrategy.LEAST_RECENTLY_USED:
            # Return the least recently created sandbox (oldest)
            return min(available_sandboxes, key=lambda s: s.created_at)

        elif sandbox_grouping_strategy == SandboxGroupingStrategy.FEWEST_CONVERSATIONS:
            # Return the one with fewest conversations
            return min(
                available_sandboxes,
                key=lambda s: sandbox_conversation_counts.get(s.id, 0),
            )

        else:
            # Default fallback - return first sandbox
            return available_sandboxes[0]

    async def _get_conversation_counts_by_sandbox(
        self, sandbox_ids: list[str]
    ) -> dict[str, int]:
        """Get the count of conversations for each sandbox.

        Args:
            sandbox_ids: List of sandbox IDs to count conversations for

        Returns:
            Dictionary mapping sandbox_id to conversation count
        """
        try:
            # Query count for each sandbox individually
            # This is efficient since there are at most ~8 running sandboxes per user
            counts: dict[str, int] = {}
            for sandbox_id in sandbox_ids:
                count = await self.app_conversation_info_service.count_app_conversation_info(
                    sandbox_id__eq=sandbox_id
                )
                counts[sandbox_id] = count
            return counts
        except Exception as e:
            _logger.warning(
                f'Error counting conversations by sandbox: {e}', exc_info=True
            )
            # Return empty counts on error - will default to first sandbox
            return {}

    async def _wait_for_sandbox_start(
        self, task: AppConversationStartTask
    ) -> AsyncGenerator[AppConversationStartTask, None]:
        """Wait for sandbox to start and return info."""
        # Get or create the sandbox
        if not task.request.sandbox_id:
            # First try to find a running sandbox for the current user
            sandbox = await self._find_running_sandbox_for_user()
            if sandbox is None:
                # No running sandbox found, start a new one

                # Convert conversation_id to hex string if present
                sandbox_id_str = (
                    task.request.conversation_id.hex
                    if task.request.conversation_id is not None
                    else None
                )

                sandbox = await self.sandbox_service.start_sandbox(
                    sandbox_id=sandbox_id_str
                )
            task.sandbox_id = sandbox.id
        else:
            sandbox_info = await self.sandbox_service.get_sandbox(
                task.request.sandbox_id
            )
            if sandbox_info is None:
                raise SandboxError(f'Sandbox not found: {task.request.sandbox_id}')
            sandbox = sandbox_info

        # Update the listener with sandbox info
        task.status = AppConversationStartTaskStatus.WAITING_FOR_SANDBOX
        task.sandbox_id = sandbox.id

        # Log sandbox assignment for observability
        conversation_id_str = (
            str(task.request.conversation_id)
            if task.request.conversation_id is not None
            else 'unknown'
        )
        _logger.info(
            f'Assigned sandbox {sandbox.id} to conversation {conversation_id_str}'
        )

        yield task

        # Resume if paused
        if sandbox.status == SandboxStatus.PAUSED:
            await self.sandbox_service.resume_sandbox(sandbox.id)

        # Check for immediate error states
        if sandbox.status in (None, SandboxStatus.ERROR):
            raise SandboxError(f'Sandbox status: {sandbox.status}')

        # For non-STARTING/RUNNING states (except PAUSED which we just resumed), fail fast
        if sandbox.status not in (
            SandboxStatus.STARTING,
            SandboxStatus.RUNNING,
            SandboxStatus.PAUSED,
        ):
            raise SandboxError(f'Sandbox not startable: {sandbox.id}')

        # Use shared wait_for_sandbox_running utility to poll for ready state
        await self.sandbox_service.wait_for_sandbox_running(
            sandbox.id,
            timeout=self.sandbox_startup_timeout,
            poll_interval=self.sandbox_startup_poll_frequency,
            httpx_client=self.httpx_client,
        )

    def _get_agent_server_url(self, sandbox: SandboxInfo) -> str:
        """Get agent server url for running sandbox."""
        exposed_urls = sandbox.exposed_urls
        assert exposed_urls is not None
        agent_server_url = next(
            exposed_url.url
            for exposed_url in exposed_urls
            if exposed_url.name == AGENT_SERVER
        )
        agent_server_url = replace_localhost_hostname_for_docker(agent_server_url)
        return agent_server_url

    def _inherit_configuration_from_parent(
        self, request: AppConversationStartRequest, parent_info: AppConversationInfo
    ) -> None:
        """Inherit configuration from parent conversation if not explicitly provided.

        This ensures sub-conversations automatically inherit:
        - Sandbox ID (to share the same workspace/environment)
        - Git parameters (repository, branch, provider)
        - LLM model

        Args:
            request: The conversation start request to modify
            parent_info: The parent conversation info to inherit from
        """
        # Inherit sandbox_id from parent to share the same workspace/environment
        if not request.sandbox_id:
            request.sandbox_id = parent_info.sandbox_id

        # Inherit git parameters from parent if not provided
        if not request.selected_repository:
            request.selected_repository = parent_info.selected_repository
        if not request.selected_branch:
            request.selected_branch = parent_info.selected_branch
        if not request.git_provider:
            request.git_provider = parent_info.git_provider

        # Inherit LLM model from parent if not provided
        if not request.llm_model and parent_info.llm_model:
            request.llm_model = parent_info.llm_model

    def _apply_suggested_task(self, request: AppConversationStartRequest) -> None:
        """Apply suggested task defaults to the start request."""
        suggested_task: SuggestedTask | None = request.suggested_task
        if not suggested_task:
            return

        if request.initial_message is not None:
            raise ValueError(
                'initial_message cannot be provided when suggested_task is present'
            )

        prompt = suggested_task.get_prompt_for_task()
        if not prompt:
            raise ValueError(
                f'Suggested task returned empty prompt for task type {suggested_task.task_type}'
            )
        request.initial_message = SendMessageRequest(
            role='user',
            content=[TextContent(text=prompt)],
        )
        request.trigger = ConversationTrigger.SUGGESTED_TASK

        if not request.selected_repository:
            request.selected_repository = suggested_task.repo
        if not request.git_provider:
            request.git_provider = suggested_task.git_provider

    def _compute_plan_path(
        self,
        working_dir: str,
        git_provider: ProviderType | None,
    ) -> str:
        """Compute the PLAN.md path based on provider type.

        Args:
            working_dir: The workspace working directory
            git_provider: The git provider type (GitHub, GitLab, Azure DevOps, etc.)

        Returns:
            Absolute path to PLAN.md file in the appropriate config directory
        """
        # GitLab and Azure DevOps use agents-tmp-config (since .agents_tmp is invalid)
        if git_provider in (ProviderType.GITLAB, ProviderType.AZURE_DEVOPS):
            config_dir = 'agents-tmp-config'
        else:
            config_dir = '.agents_tmp'

        return f'{working_dir}/{config_dir}/PLAN.md'

    async def _setup_secrets_for_git_providers(self, user: UserInfo) -> dict:
        """Set up secrets for all git provider authentication.

        Args:
            user: User information containing authentication details

        Returns:
            Dictionary of secrets for the conversation
        """
        secrets = await self.user_context.get_secrets()

        # Get all provider tokens from user authentication
        provider_tokens = cast(
            PROVIDER_TOKEN_TYPE | None,
            await self.user_context.get_provider_tokens(),
        )
        if not provider_tokens:
            return secrets

        # Create secrets for each provider token
        for provider_type, provider_token in provider_tokens.items():
            if not provider_token.token:
                continue

            secret_name = f'{provider_type.name}_TOKEN'
            description = f'{provider_type.name} authentication token'

            if self.web_url:
                # Create an access token for web-based authentication
                access_token = self.jwt_service.create_jws_token(
                    payload={
                        'user_id': user.id,
                        'provider_type': provider_type.value,
                    },
                    expires_in=self.access_token_hard_timeout,
                )
                headers = {'X-Access-Token': access_token}

                secrets[secret_name] = LookupSecret(
                    url=self.web_url + '/api/v1/webhooks/secrets',
                    headers=headers,
                    description=description,
                )
            else:
                # Use static token for environments without web URL access
                static_token = await self.user_context.get_latest_token(provider_type)
                if static_token:
                    secrets[secret_name] = StaticSecret(
                        value=SecretStr(static_token), description=description
                    )

        return secrets

    def _configure_llm(self, user: UserInfo, llm_model: str | None) -> LLM:
        """Configure LLM settings.

        Args:
            user: User information containing LLM preferences
            llm_model: Optional specific model to use, falls back to user default

        Returns:
            Configured LLM instance
        """
        model: str = llm_model or user.llm_model or LLM.model_fields['model'].default
        base_url = user.llm_base_url
        if model and (
            model.startswith('openhands/') or model.startswith('litellm_proxy/')
        ):
            base_url = user.llm_base_url or self.openhands_provider_base_url

        return LLM(
            model=model,
            base_url=base_url,
            api_key=user.llm_api_key,
            usage_id='agent',
        )

    async def _get_tavily_api_key(self, user: UserInfo) -> str | None:
        """Get Tavily search API key, prioritizing user's key over service key.

        Args:
            user: User information

        Returns:
            Tavily API key if available, None otherwise
        """
        # Get the actual API key values, prioritizing user's key over service key
        user_search_key = None
        if user.search_api_key:
            key_value = user.search_api_key.get_secret_value()
            if key_value and key_value.strip():
                user_search_key = key_value

        service_tavily_key = None
        if self.tavily_api_key:
            # tavily_api_key is already a string (extracted in the factory method)
            if self.tavily_api_key.strip():
                service_tavily_key = self.tavily_api_key

        return user_search_key or service_tavily_key

    async def _add_system_mcp_servers(
        self, mcp_servers: dict[str, Any], user: UserInfo, conversation_id: UUID
    ) -> None:
        """Add system-generated MCP servers (default OpenHands server and Tavily).

        Args:
            mcp_servers: Dictionary to add servers to
            user: User information for API keys
            conversation_id: Conversation ID forwarded to the OpenHands MCP server
        """
        if not self.web_url:
            return

        # Add default OpenHands MCP server
        mcp_url = f'{self.web_url}/mcp/mcp'
        mcp_servers['default'] = {
            'url': mcp_url,
            'headers': {'X-OpenHands-ServerConversation-ID': str(conversation_id)},
        }

        # Add API key if available
        mcp_api_key = await self.user_context.get_mcp_api_key()
        if mcp_api_key:
            mcp_servers['default']['headers']['X-Session-API-Key'] = mcp_api_key

        # Add Tavily search if API key is available
        tavily_api_key = await self._get_tavily_api_key(user)
        if tavily_api_key:
            _logger.info('Adding search engine to MCP config')
            mcp_servers['tavily'] = {
                'url': f'https://mcp.tavily.com/mcp/?tavilyApiKey={tavily_api_key}'
            }
        else:
            _logger.info('No search engine API key found, skipping search engine')

    def _add_custom_sse_servers(
        self, mcp_servers: dict[str, Any], sse_servers: list
    ) -> None:
        """Add custom SSE MCP servers from user configuration.

        Args:
            mcp_servers: Dictionary to add servers to
            sse_servers: List of SSE server configurations
        """
        for sse_server in sse_servers:
            server_config = {
                'url': sse_server.url,
                'transport': 'sse',
            }
            if sse_server.api_key:
                server_config['headers'] = {
                    'Authorization': f'Bearer {sse_server.api_key}'
                }

            # Generate unique server name using UUID
            # TODO: Let the users specify the server name
            server_name = f'sse_{uuid4().hex[:8]}'
            mcp_servers[server_name] = server_config
            _logger.debug(
                f'Added custom SSE server: {server_name} for {sse_server.url}'
            )

    def _add_custom_shttp_servers(
        self, mcp_servers: dict[str, Any], shttp_servers: list
    ) -> None:
        """Add custom SHTTP MCP servers from user configuration.

        Args:
            mcp_servers: Dictionary to add servers to
            shttp_servers: List of SHTTP server configurations
        """
        for shttp_server in shttp_servers:
            server_config = {
                'url': shttp_server.url,
                'transport': 'streamable-http',
            }
            if shttp_server.api_key:
                server_config['headers'] = {
                    'Authorization': f'Bearer {shttp_server.api_key}'
                }
            if shttp_server.timeout:
                server_config['timeout'] = shttp_server.timeout

            # Generate unique server name using UUID
            # TODO: Let the users specify the server name
            server_name = f'shttp_{uuid4().hex[:8]}'
            mcp_servers[server_name] = server_config
            _logger.debug(
                f'Added custom SHTTP server: {server_name} for {shttp_server.url}'
            )

    def _add_custom_stdio_servers(
        self, mcp_servers: dict[str, Any], stdio_servers: list
    ) -> None:
        """Add custom STDIO MCP servers from user configuration.

        Args:
            mcp_servers: Dictionary to add servers to
            stdio_servers: List of STDIO server configurations
        """
        for stdio_server in stdio_servers:
            server_config = {
                'command': stdio_server.command,
                'args': stdio_server.args,
            }
            if stdio_server.env:
                server_config['env'] = stdio_server.env

            # STDIO servers have an explicit name field
            mcp_servers[stdio_server.name] = server_config
            _logger.debug(f'Added custom STDIO server: {stdio_server.name}')

    def _merge_custom_mcp_config(
        self, mcp_servers: dict[str, Any], user: UserInfo
    ) -> None:
        """Merge custom MCP configuration from user settings.

        Args:
            mcp_servers: Dictionary to add servers to
            user: User information containing custom MCP config
        """
        if not user.mcp_config:
            return

        try:
            sse_count = len(user.mcp_config.sse_servers)
            shttp_count = len(user.mcp_config.shttp_servers)
            stdio_count = len(user.mcp_config.stdio_servers)

            _logger.info(
                f'Loading custom MCP config from user settings: '
                f'{sse_count} SSE, {shttp_count} SHTTP, {stdio_count} STDIO servers'
            )

            # Add each type of custom server
            self._add_custom_sse_servers(mcp_servers, user.mcp_config.sse_servers)
            self._add_custom_shttp_servers(mcp_servers, user.mcp_config.shttp_servers)
            self._add_custom_stdio_servers(mcp_servers, user.mcp_config.stdio_servers)

            _logger.info(
                f'Successfully merged custom MCP config: added {sse_count} SSE, '
                f'{shttp_count} SHTTP, and {stdio_count} STDIO servers'
            )

        except Exception as e:
            _logger.error(
                f'Error loading custom MCP config from user settings: {e}',
                exc_info=True,
            )
            # Continue with system config only, don't fail conversation startup
            _logger.warning(
                'Continuing with system-generated MCP config only due to custom config error'
            )

    async def _configure_llm_and_mcp(
        self, user: UserInfo, llm_model: str | None, conversation_id: UUID
    ) -> tuple[LLM, dict]:
        """Configure LLM and MCP (Model Context Protocol) settings.

        Args:
            user: User information containing LLM preferences
            llm_model: Optional specific model to use, falls back to user default
            conversation_id: Conversation ID forwarded to the OpenHands MCP server

        Returns:
            Tuple of (configured LLM instance, MCP config dictionary)
        """
        # Configure LLM
        llm = self._configure_llm(user, llm_model)

        # Configure MCP - SDK expects format: {'mcpServers': {'server_name': {...}}}
        mcp_servers: dict[str, Any] = {}

        # Add system-generated servers (default + tavily)
        await self._add_system_mcp_servers(mcp_servers, user, conversation_id)

        # Merge custom servers from user settings
        self._merge_custom_mcp_config(mcp_servers, user)

        # Wrap in the mcpServers structure required by the SDK
        mcp_config = {'mcpServers': mcp_servers} if mcp_servers else {}
        _logger.info(f'Final MCP configuration: {sanitize_config(mcp_config)}')

        return llm, mcp_config

    def _create_agent_with_context(
        self,
        llm: LLM,
        agent_type: AgentType,
        system_message_suffix: str | None,
        mcp_config: dict,
        condenser_max_size: int | None,
        secrets: dict[str, SecretValue] | None = None,
        git_provider: ProviderType | None = None,
        working_dir: str | None = None,
    ) -> Agent:
        """Create an agent with appropriate tools and context based on agent type.

        Args:
            llm: Configured LLM instance
            agent_type: Type of agent to create (PLAN or DEFAULT)
            system_message_suffix: Optional suffix for system messages
            mcp_config: MCP configuration dictionary
            condenser_max_size: condenser_max_size setting
            secrets: Optional dictionary of secrets for authentication
            git_provider: Optional git provider type for computing plan path
            working_dir: Optional working directory for computing plan path

        Returns:
            Configured Agent instance with context
        """
        # Create condenser with user's settings
        condenser = self._create_condenser(llm, agent_type, condenser_max_size)

        # Create agent based on type
        if agent_type == AgentType.PLAN:
            # Compute plan path if working_dir is provided
            plan_path = None
            if working_dir:
                plan_path = self._compute_plan_path(working_dir, git_provider)

            agent = Agent(
                llm=llm,
                tools=get_planning_tools(plan_path=plan_path),
                system_prompt_filename='system_prompt_planning.j2',
                system_prompt_kwargs={'plan_structure': format_plan_structure()},
                condenser=condenser,
                mcp_config=mcp_config,
            )
        else:
            agent = Agent(
                llm=llm,
                tools=get_default_tools(enable_browser=True),
                system_prompt_kwargs={'cli_mode': False},
                condenser=condenser,
                mcp_config=mcp_config,
            )

        # Prepare system message suffix based on agent type
        effective_system_message_suffix = system_message_suffix
        if agent_type == AgentType.PLAN:
            # Prepend planning-specific instruction to prevent "Ready to proceed?" behavior
            if system_message_suffix:
                effective_system_message_suffix = (
                    f'{PLANNING_AGENT_INSTRUCTION}\n\n{system_message_suffix}'
                )
            else:
                effective_system_message_suffix = PLANNING_AGENT_INSTRUCTION

        # Add agent context
        agent_context = AgentContext(
            system_message_suffix=effective_system_message_suffix, secrets=secrets
        )
        agent = agent.model_copy(update={'agent_context': agent_context})

        return agent

    def _update_agent_with_llm_metadata(
        self,
        agent: Agent,
        conversation_id: UUID,
        user_id: str | None,
    ) -> Agent:
        """Update agent's LLM and condenser LLM with litellm_extra_body metadata.

        This adds tracing metadata (conversation_id, user_id, etc.) to the LLM
        for analytics and debugging purposes. Only applies to openhands/ models.

        Args:
            agent: The agent to update
            conversation_id: The conversation ID
            user_id: The user ID (can be None)

        Returns:
            Updated agent with LLM metadata
        """
        updates: dict[str, Any] = {}

        # Update main LLM if it's an openhands model
        if should_set_litellm_extra_body(agent.llm.model):
            llm_metadata = get_llm_metadata(
                model_name=agent.llm.model,
                llm_type=agent.llm.usage_id or 'agent',
                conversation_id=conversation_id,
                user_id=user_id,
            )
            updated_llm = agent.llm.model_copy(
                update={'litellm_extra_body': {'metadata': llm_metadata}}
            )
            updates['llm'] = updated_llm

        # Update condenser LLM if it exists and is an openhands model
        if agent.condenser and hasattr(agent.condenser, 'llm'):
            condenser_llm = agent.condenser.llm
            if should_set_litellm_extra_body(condenser_llm.model):
                condenser_metadata = get_llm_metadata(
                    model_name=condenser_llm.model,
                    llm_type=condenser_llm.usage_id or 'condenser',
                    conversation_id=conversation_id,
                    user_id=user_id,
                )
                updated_condenser_llm = condenser_llm.model_copy(
                    update={'litellm_extra_body': {'metadata': condenser_metadata}}
                )
                updated_condenser = agent.condenser.model_copy(
                    update={'llm': updated_condenser_llm}
                )
                updates['condenser'] = updated_condenser

        # Return updated agent if there are changes
        if updates:
            return agent.model_copy(update=updates)
        return agent

    def _construct_initial_message_with_plugin_params(
        self,
        initial_message: SendMessageRequest | None,
        plugins: list[PluginSpec] | None,
    ) -> SendMessageRequest | None:
        """Incorporate plugin parameters into the initial message if specified.

        Plugin parameters are formatted and appended to the initial message so the
        agent has context about the user-provided configuration values.

        Args:
            initial_message: The original initial message, if any
            plugins: List of plugin specifications with optional parameters

        Returns:
            The initial message with plugin parameters incorporated, or the
            original message if no plugin parameters are specified
        """
        from openhands.agent_server.models import TextContent

        if not plugins:
            return initial_message

        # Collect formatted parameters from plugins that have them
        plugins_with_params = [p for p in plugins if p.parameters]
        if not plugins_with_params:
            return initial_message

        # Format parameters, grouped by plugin if multiple
        if len(plugins_with_params) == 1:
            params_text = plugins_with_params[0].format_params_as_text()
            plugin_params_message = (
                f'\n\nPlugin Configuration Parameters:\n{params_text}'
            )
        else:
            # Group by plugin name for clarity
            formatted_plugins = []
            for plugin in plugins_with_params:
                params_text = plugin.format_params_as_text(indent='  ')
                if params_text:
                    formatted_plugins.append(f'{plugin.display_name}:\n{params_text}')

            plugin_params_message = (
                '\n\nPlugin Configuration Parameters:\n' + '\n'.join(formatted_plugins)
            )

        if initial_message is None:
            # Create a new message with just the plugin parameters
            return SendMessageRequest(
                content=[TextContent(text=plugin_params_message.strip())],
                run=True,
            )

        # Append plugin parameters to existing message content
        new_content = list(initial_message.content)
        if new_content and isinstance(new_content[-1], TextContent):
            # Append to the last text content
            last_content = new_content[-1]
            new_content[-1] = TextContent(
                text=last_content.text + plugin_params_message,
                cache_prompt=last_content.cache_prompt,
            )
        else:
            # Add as new text content
            new_content.append(TextContent(text=plugin_params_message.strip()))

        return SendMessageRequest(
            role=initial_message.role,
            content=new_content,
            run=initial_message.run,
        )

    async def _load_hooks_from_workspace(
        self,
        remote_workspace: AsyncRemoteWorkspace,
        project_dir: str,
    ) -> HookConfig | None:
        """Load hooks from .openhands/hooks.json in the remote workspace.

        This enables project-level hooks to be automatically loaded when starting
        a conversation, similar to how OpenHands-CLI loads hooks from the workspace.

        Uses the agent-server's /api/hooks endpoint, consistent with how skills
        are loaded via /api/skills.

        Args:
            remote_workspace: AsyncRemoteWorkspace for accessing the agent server
            project_dir: Project root directory path in the sandbox. This should
                already be the resolved project directory (e.g.,
                {working_dir}/{repo_name} when a repo is selected).

        Returns:
            HookConfig if hooks.json exists and is valid, None otherwise.
            Returns None in the following cases:
            - hooks.json file does not exist
            - hooks.json contains invalid JSON
            - hooks.json contains an empty hooks configuration
            - Agent server is unreachable or returns an error

        Note:
            This method implements graceful degradation - if hooks cannot be loaded
            for any reason, it returns None rather than raising an exception. This
            ensures that conversation startup is not blocked by hook loading failures.
            Errors are logged as warnings for debugging purposes.
        """
        return await load_hooks_from_agent_server(
            agent_server_url=remote_workspace.host,
            session_api_key=remote_workspace._headers.get('X-Session-API-Key'),
            project_dir=project_dir,
            httpx_client=self.httpx_client,
        )

    async def _finalize_conversation_request(
        self,
        agent: Agent,
        conversation_id: UUID,
        user: UserInfo,
        workspace: LocalWorkspace,
        initial_message: SendMessageRequest | None,
        secrets: dict[str, SecretValue],
        sandbox: SandboxInfo,
        remote_workspace: AsyncRemoteWorkspace | None,
        selected_repository: str | None,
        working_dir: str,
        plugins: list[PluginSpec] | None = None,
    ) -> StartConversationRequest:
        """Finalize the conversation request with skills and metadata.

        Args:
            agent: The configured agent
            conversation_id: Conversation ID
            user: User information
            workspace: Local workspace instance
            initial_message: Optional initial message for the conversation
            secrets: Dictionary of secrets for authentication
            sandbox: Sandbox information
            remote_workspace: Optional remote workspace for skills loading
            selected_repository: Optional repository name
            working_dir: Working directory path
            plugins: Optional list of plugin specifications to load

        Returns:
            Complete StartConversationRequest ready for use
        """
        # Update agent's LLM with litellm_extra_body metadata for tracing
        agent = self._update_agent_with_llm_metadata(agent, conversation_id, user.id)

        # Load and merge skills if remote workspace is available
        hook_config: HookConfig | None = None
        if remote_workspace:
            try:
                agent = await self._load_skills_and_update_agent(
                    sandbox,
                    agent,
                    remote_workspace,
                    selected_repository,
                    working_dir,
                    disabled_skills=user.disabled_skills,
                )
            except Exception as e:
                _logger.warning(f'Failed to load skills: {e}', exc_info=True)
                # Continue without skills - don't fail conversation startup

            # Load hooks from workspace (.openhands/hooks.json)
            # Note: working_dir is already the resolved project_dir
            # (includes repo name when a repo is selected), so we pass
            # it directly without appending the repo name again.
            try:
                _logger.debug(
                    f'Attempting to load hooks from workspace: '
                    f'project_dir={working_dir}'
                )
                hook_config = await self._load_hooks_from_workspace(
                    remote_workspace, working_dir
                )
                if hook_config:
                    _logger.debug(
                        f'Successfully loaded hooks: {hook_config.model_dump()}'
                    )
                else:
                    _logger.debug('No hooks found in workspace')
            except Exception as e:
                _logger.warning(f'Failed to load hooks: {e}', exc_info=True)
                # Continue without hooks - don't fail conversation startup

        # Incorporate plugin parameters into initial message if specified
        final_initial_message = self._construct_initial_message_with_plugin_params(
            initial_message, plugins
        )

        # Convert PluginSpec list to SDK PluginSource list for agent server
        sdk_plugins: list[PluginSource] | None = None
        if plugins:
            sdk_plugins = [
                PluginSource(
                    source=p.source,
                    ref=p.ref,
                    repo_path=p.repo_path,
                )
                for p in plugins
            ]

        # Create and return the final request
        return StartConversationRequest(
            conversation_id=conversation_id,
            agent=agent,
            workspace=workspace,
            confirmation_policy=self._select_confirmation_policy(
                bool(user.confirmation_mode), user.security_analyzer
            ),
            initial_message=final_initial_message,
            secrets=secrets,
            plugins=sdk_plugins,
            hook_config=hook_config,
        )

    async def _build_start_conversation_request_for_user(
        self,
        sandbox: SandboxInfo,
        conversation_id: UUID,
        initial_message: SendMessageRequest | None,
        system_message_suffix: str | None,
        git_provider: ProviderType | None,
        working_dir: str,
        agent_type: AgentType = AgentType.DEFAULT,
        llm_model: str | None = None,
        remote_workspace: AsyncRemoteWorkspace | None = None,
        selected_repository: str | None = None,
        plugins: list[PluginSpec] | None = None,
    ) -> StartConversationRequest:
        """Build a complete conversation request for a user.

        This method orchestrates the creation of a conversation request by:
        1. Setting up git provider secrets
        2. Configuring LLM and MCP settings
        3. Creating an agent with appropriate context
        4. Finalizing the request with skills and metadata
        5. Passing plugins to the agent server for remote plugin loading
        """
        user = await self.user_context.get_user_info()

        # Compute the project root — this is the repo directory when a repo is
        # selected, or the sandbox working_dir otherwise. All tools, hooks,
        # setup scripts, and plan paths must use this consistently.
        project_dir = get_project_dir(working_dir, selected_repository)
        workspace = LocalWorkspace(working_dir=project_dir)

        # Set up secrets for all git providers
        secrets = await self._setup_secrets_for_git_providers(user)

        # Configure LLM and MCP
        llm, mcp_config = await self._configure_llm_and_mcp(
            user, llm_model, conversation_id
        )

        # Create agent with context
        agent = self._create_agent_with_context(
            llm,
            agent_type,
            system_message_suffix,
            mcp_config,
            user.condenser_max_size,
            secrets=secrets,
            git_provider=git_provider,
            working_dir=project_dir,
        )

        # Finalize and return the conversation request
        return await self._finalize_conversation_request(
            agent,
            conversation_id,
            user,
            workspace,
            initial_message,
            secrets,
            sandbox,
            remote_workspace,
            selected_repository,
            project_dir,
            plugins=plugins,
        )

    async def _process_pending_messages(
        self,
        task_id: UUID,
        conversation_id: UUID,
        agent_server_url: str,
        session_api_key: str,
    ) -> None:
        """Process pending messages queued before conversation was ready.

        Messages are delivered concurrently to the agent server. After processing,
        all messages are deleted from the database regardless of success or failure.

        Args:
            task_id: The start task ID (may have been used as conversation_id initially)
            conversation_id: The real conversation ID
            agent_server_url: URL of the agent server
            session_api_key: API key for authenticating with agent server
        """
        # Convert UUIDs to strings for the pending message service
        # The frontend uses task-{uuid.hex} format (no hyphens), matching OpenHandsUUID serialization
        task_id_str = f'task-{task_id.hex}'
        # conversation_id uses standard format (with hyphens) for agent server API compatibility
        conversation_id_str = str(conversation_id)

        _logger.info(f'task_id={task_id_str} conversation_id={conversation_id_str}')

        # First, update any messages that were queued with the task_id
        updated_count = await self.pending_message_service.update_conversation_id(
            old_conversation_id=task_id_str,
            new_conversation_id=conversation_id_str,
        )
        _logger.info(f'updated_count={updated_count} ')
        if updated_count > 0:
            _logger.info(
                f'Updated {updated_count} pending messages from task_id={task_id_str} '
                f'to conversation_id={conversation_id_str}'
            )

        # Get all pending messages for this conversation
        pending_messages = await self.pending_message_service.get_pending_messages(
            conversation_id_str
        )

        if not pending_messages:
            return

        _logger.info(
            f'Processing {len(pending_messages)} pending messages for '
            f'conversation {conversation_id_str}'
        )

        # Process messages sequentially to preserve order
        for msg in pending_messages:
            try:
                # Serialize content objects to JSON-compatible dicts
                content_json = [item.model_dump() for item in msg.content]
                # Use the events endpoint which handles message sending
                response = await self.httpx_client.post(
                    f'{agent_server_url}/api/conversations/{conversation_id_str}/events',
                    json={
                        'role': msg.role,
                        'content': content_json,
                        'run': True,
                    },
                    headers={'X-Session-API-Key': session_api_key},
                    timeout=30.0,
                )
                response.raise_for_status()
                _logger.debug(f'Delivered pending message {msg.id}')
            except Exception as e:
                _logger.warning(f'Failed to deliver pending message {msg.id}: {e}')

        # Delete all pending messages after processing (regardless of success/failure)
        deleted_count = (
            await self.pending_message_service.delete_messages_for_conversation(
                conversation_id_str
            )
        )
        _logger.info(
            f'Finished processing pending messages for conversation {conversation_id_str}. '
            f'Deleted {deleted_count} messages.'
        )

    async def update_agent_server_conversation_title(
        self,
        conversation_id: str,
        new_title: str,
        app_conversation_info: AppConversationInfo,
    ) -> None:
        """Update the conversation title in the agent-server.

        Args:
            conversation_id: The conversation ID as a string
            new_title: The new title to set
            app_conversation_info: The app conversation info containing sandbox_id
        """
        # Get the sandbox info to find the agent-server URL
        sandbox = await self.sandbox_service.get_sandbox(
            app_conversation_info.sandbox_id
        )
        assert sandbox is not None, (
            f'Sandbox {app_conversation_info.sandbox_id} not found for conversation {conversation_id}'
        )
        assert sandbox.exposed_urls is not None, (
            f'Sandbox {app_conversation_info.sandbox_id} has no exposed URLs for conversation {conversation_id}'
        )

        # Use the existing method to get the agent-server URL
        agent_server_url = self._get_agent_server_url(sandbox)

        # Prepare the request
        url = f'{agent_server_url.rstrip("/")}/api/conversations/{conversation_id}'
        headers = {}
        if sandbox.session_api_key:
            headers['X-Session-API-Key'] = sandbox.session_api_key

        payload = {'title': new_title}

        # Make the PATCH request to the agent-server
        response = await self.httpx_client.patch(
            url,
            json=payload,
            headers=headers,
            timeout=30.0,
        )
        response.raise_for_status()

        _logger.info(
            f'Successfully updated agent-server conversation {conversation_id} title to "{new_title}"'
        )

    def _validate_repository_update(
        self,
        request: AppConversationUpdateRequest,
        existing_branch: str | None = None,
    ) -> None:
        """Validate repository-related fields in the update request.

        Args:
            request: The update request containing fields to validate
            existing_branch: The conversation's current branch (if any)

        Raises:
            ValueError: If validation fails
        """
        # Check if repository is being set
        if 'selected_repository' in request.model_fields_set:
            repo = request.selected_repository
            if repo is not None:
                # Validate repository format (owner/repo)
                if '/' not in repo or repo.count('/') != 1:
                    raise ValueError(
                        f"Invalid repository format: '{repo}'. Expected 'owner/repo'."
                    )

                # Sanitize: check for dangerous characters
                if any(c in repo for c in [';', '&', '|', '$', '`', '\n', '\r']):
                    raise ValueError(f"Invalid characters in repository name: '{repo}'")

                # If setting a repository, branch should also be provided
                # (either in this request or already exists in conversation)
                if (
                    'selected_branch' not in request.model_fields_set
                    and existing_branch is None
                ):
                    _logger.warning(
                        f'Repository {repo} set without branch in the same request '
                        'and no existing branch in conversation'
                    )
            else:
                # Repository is being removed (set to null)
                # Enforce consistency: branch and provider must also be cleared
                if 'selected_branch' in request.model_fields_set:
                    if request.selected_branch is not None:
                        raise ValueError(
                            'When removing repository, branch must also be cleared'
                        )
                if 'git_provider' in request.model_fields_set:
                    if request.git_provider is not None:
                        raise ValueError(
                            'When removing repository, git_provider must also be cleared'
                        )

        # Validate branch if provided
        if 'selected_branch' in request.model_fields_set:
            branch = request.selected_branch
            if branch is not None:
                ensure_valid_git_branch_name(branch)

    async def update_app_conversation(
        self, conversation_id: UUID, request: AppConversationUpdateRequest
    ) -> AppConversation | None:
        """Update an app conversation and return it.

        Return None if the conversation did not exist.

        Only fields that are explicitly set in the request will be updated.
        This allows partial updates where only specific fields are modified.
        Fields can be set to None to clear them (e.g., removing a repository).

        Raises:
            ValueError: If repository/branch validation fails
        """
        info = await self.app_conversation_info_service.get_app_conversation_info(
            conversation_id
        )
        if info is None:
            return None

        # Validate repository-related fields before updating
        # Pass existing branch to avoid false warnings when only updating repository
        self._validate_repository_update(request, existing_branch=info.selected_branch)

        # Only update fields that were explicitly provided in the request
        # This uses Pydantic's model_fields_set to detect which fields were set,
        # allowing us to distinguish between "not provided" and "explicitly set to None"
        for field_name in request.model_fields_set:
            value = getattr(request, field_name)
            setattr(info, field_name, value)

        info = await self.app_conversation_info_service.save_app_conversation_info(info)
        conversations = await self._build_app_conversations([info])
        return conversations[0]

    async def delete_app_conversation(
        self, conversation_id: UUID, skip_agent_server_delete: bool = False
    ) -> bool:
        """Delete a V1 conversation and all its associated data.

        This method will also cascade delete all sub-conversations of the parent.

        Args:
            conversation_id: The UUID of the conversation to delete.
            skip_agent_server_delete: If True, skip the agent server DELETE call.
                This should be set when the sandbox is shared with other
                conversations (e.g. created via /new) to avoid destabilizing
                the shared runtime.
        """
        # Check if we have the required SQL implementation for transactional deletion
        if not isinstance(
            self.app_conversation_info_service, SQLAppConversationInfoService
        ):
            _logger.error(
                f'Cannot delete V1 conversation {conversation_id}: SQL implementation required for transactional deletion',
                extra={'conversation_id': str(conversation_id)},
            )
            return False

        try:
            # First, fetch the conversation to get the full object needed for agent server deletion
            app_conversation = await self.get_app_conversation(conversation_id)
            if not app_conversation:
                _logger.warning(
                    f'V1 conversation {conversation_id} not found for deletion',
                    extra={'conversation_id': str(conversation_id)},
                )
                return False

            # Delete all sub-conversations first (to maintain referential integrity)
            await self._delete_sub_conversations(conversation_id)

            # Now delete the parent conversation
            # Delete from agent server if sandbox is running (skip if sandbox is shared)
            if not skip_agent_server_delete:
                await self._delete_from_agent_server(app_conversation)

            # Delete from database using the conversation info from app_conversation
            # AppConversation extends AppConversationInfo, so we can use it directly
            return await self._delete_from_database(app_conversation)

        except Exception as e:
            _logger.error(
                f'Error deleting V1 conversation {conversation_id}: {e}',
                extra={'conversation_id': str(conversation_id)},
                exc_info=True,
            )
            return False

    async def _delete_sub_conversations(self, parent_conversation_id: UUID) -> None:
        """Delete all sub-conversations of a parent conversation.

        This method handles errors gracefully, continuing to delete remaining
        sub-conversations even if one fails.

        Args:
            parent_conversation_id: The UUID of the parent conversation.
        """
        sub_conversation_ids = (
            await self.app_conversation_info_service.get_sub_conversation_ids(
                parent_conversation_id
            )
        )

        for sub_id in sub_conversation_ids:
            try:
                sub_conversation = await self.get_app_conversation(sub_id)
                if sub_conversation:
                    # Delete from agent server if sandbox is running
                    await self._delete_from_agent_server(sub_conversation)
                    # Delete from database
                    await self._delete_from_database(sub_conversation)
                    _logger.info(
                        f'Successfully deleted sub-conversation {sub_id}',
                        extra={'conversation_id': str(sub_id)},
                    )
            except Exception as e:
                # Log error but continue deleting remaining sub-conversations
                _logger.warning(
                    f'Error deleting sub-conversation {sub_id}: {e}',
                    extra={'conversation_id': str(sub_id)},
                    exc_info=True,
                )

    async def _delete_from_agent_server(
        self, app_conversation: AppConversation
    ) -> None:
        """Delete conversation from agent server if sandbox is running."""
        conversation_id = app_conversation.id
        if not (
            app_conversation.sandbox_status == SandboxStatus.RUNNING
            and app_conversation.session_api_key
        ):
            return

        try:
            # Get sandbox info to find agent server URL
            sandbox = await self.sandbox_service.get_sandbox(
                app_conversation.sandbox_id
            )
            if sandbox and sandbox.exposed_urls:
                agent_server_url = self._get_agent_server_url(sandbox)

                # Call agent server delete API
                response = await self.httpx_client.delete(
                    f'{agent_server_url}/api/conversations/{conversation_id}',
                    headers={'X-Session-API-Key': app_conversation.session_api_key},
                    timeout=30.0,
                )
                response.raise_for_status()
        except Exception as e:
            _logger.warning(
                f'Failed to delete conversation from agent server: {e}',
                extra={'conversation_id': str(conversation_id)},
            )
            # Continue with database cleanup even if agent server call fails

    async def _delete_from_database(
        self, app_conversation_info: AppConversationInfo
    ) -> bool:
        """Delete conversation from database.

        Args:
            app_conversation_info: The app conversation info to delete (already fetched).
        """
        # The session is already managed by the dependency injection system
        # No need for explicit transaction management here
        deleted_info = (
            await self.app_conversation_info_service.delete_app_conversation_info(
                app_conversation_info.id
            )
        )
        deleted_tasks = await self.app_conversation_start_task_service.delete_app_conversation_start_tasks(
            app_conversation_info.id
        )

        return deleted_info or deleted_tasks

    async def export_conversation(self, conversation_id: UUID) -> bytes:
        """Download a conversation trajectory as a zip file.

        Args:
            conversation_id: The UUID of the conversation to download.

        Returns the zip file as bytes.
        """
        # Get the conversation info to verify it exists and user has access
        conversation_info = (
            await self.app_conversation_info_service.get_app_conversation_info(
                conversation_id
            )
        )
        if not conversation_info:
            raise ValueError(f'Conversation not found: {conversation_id}')

        # Create a temporary directory to store files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Get all events for this conversation
            i = 0
            async for event in page_iterator(
                self.event_service.search_events, conversation_id=conversation_id
            ):
                event_filename = f'event_{i:06d}_{event.id}.json'
                event_path = os.path.join(temp_dir, event_filename)

                with open(event_path, 'w') as f:
                    # Use model_dump with mode='json' to handle UUID serialization
                    event_data = event.model_dump(mode='json')
                    json.dump(event_data, f, indent=2)
                i += 1

            # Create meta.json with conversation info
            meta_path = os.path.join(temp_dir, 'meta.json')
            with open(meta_path, 'w') as f:
                f.write(conversation_info.model_dump_json(indent=2))

            # Create zip file in memory
            zip_buffer = tempfile.NamedTemporaryFile()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add all files from temp directory to zip
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_dir)
                        zipf.write(file_path, arcname)

            # Read the zip file content
            zip_buffer.seek(0)
            zip_content = zip_buffer.read()
            zip_buffer.close()

            return zip_content


class LiveStatusAppConversationServiceInjector(AppConversationServiceInjector):
    sandbox_startup_timeout: int = Field(
        default=120, description='The max timeout time for sandbox startup'
    )
    sandbox_startup_poll_frequency: int = Field(
        default=2, description='The frequency to poll for sandbox readiness'
    )
    max_num_conversations_per_sandbox: int = Field(
        default=20,
        description='The maximum number of conversations allowed per sandbox',
    )
    init_git_in_empty_workspace: bool = Field(
        default=True,
        description='Whether to initialize a git repo when the workspace is empty',
    )
    access_token_hard_timeout: int | None = Field(
        default=14 * 86400,
        description=(
            'A security measure - the time after which git tokens may no longer '
            'be retrieved by a sandboxed conversation.'
        ),
    )
    tavily_api_key: SecretStr | None = Field(
        default=None,
        description='The Tavily Search API key to add to MCP integration',
    )

    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[AppConversationService, None]:
        from openhands.app_server.config import (
            get_app_conversation_info_service,
            get_app_conversation_start_task_service,
            get_event_service,
            get_global_config,
            get_httpx_client,
            get_jwt_service,
            get_pending_message_service,
            get_sandbox_service,
            get_sandbox_spec_service,
            get_user_context,
        )

        async with (
            get_user_context(state, request) as user_context,
            get_sandbox_service(state, request) as sandbox_service,
            get_sandbox_spec_service(state, request) as sandbox_spec_service,
            get_app_conversation_info_service(
                state, request
            ) as app_conversation_info_service,
            get_app_conversation_start_task_service(
                state, request
            ) as app_conversation_start_task_service,
            get_event_callback_service(state, request) as event_callback_service,
            get_event_service(state, request) as event_service,
            get_jwt_service(state, request) as jwt_service,
            get_httpx_client(state, request) as httpx_client,
            get_pending_message_service(state, request) as pending_message_service,
        ):
            access_token_hard_timeout = None
            if self.access_token_hard_timeout:
                access_token_hard_timeout = timedelta(
                    seconds=float(self.access_token_hard_timeout)
                )
            config = get_global_config()

            # If no web url has been set and we are using docker, we can use host.docker.internal
            web_url = config.web_url
            if web_url is None:
                if isinstance(sandbox_service, DockerSandboxService):
                    web_url = f'http://host.docker.internal:{sandbox_service.host_port}'

            # Get app_mode for SaaS mode
            app_mode = None
            try:
                from openhands.server.shared import server_config

                app_mode = (
                    server_config.app_mode.value if server_config.app_mode else None
                )
            except (ImportError, AttributeError):
                # If server_config is not available (e.g., in tests), continue without it
                pass

            # We supply the global tavily key only if the app mode is not SAAS, where
            # currently the search endpoints are patched into the app server instead
            # so the tavily key does not need to be shared
            if self.tavily_api_key and app_mode != AppMode.SAAS:
                tavily_api_key = self.tavily_api_key.get_secret_value()
            else:
                tavily_api_key = None

            yield LiveStatusAppConversationService(
                init_git_in_empty_workspace=self.init_git_in_empty_workspace,
                user_context=user_context,
                sandbox_service=sandbox_service,
                sandbox_spec_service=sandbox_spec_service,
                app_conversation_info_service=app_conversation_info_service,
                app_conversation_start_task_service=app_conversation_start_task_service,
                event_callback_service=event_callback_service,
                event_service=event_service,
                jwt_service=jwt_service,
                pending_message_service=pending_message_service,
                sandbox_startup_timeout=self.sandbox_startup_timeout,
                sandbox_startup_poll_frequency=self.sandbox_startup_poll_frequency,
                max_num_conversations_per_sandbox=self.max_num_conversations_per_sandbox,
                httpx_client=httpx_client,
                web_url=web_url,
                openhands_provider_base_url=config.openhands_provider_base_url,
                access_token_hard_timeout=access_token_hard_timeout,
                app_mode=app_mode,
                tavily_api_key=tavily_api_key,
            )
