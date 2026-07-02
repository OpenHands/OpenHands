import asyncio
import importlib.metadata
import io
import json
import logging
import os
import zipfile
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, BinaryIO, Sequence, cast
from uuid import UUID, uuid4

import httpx
from fastapi import Request
from packaging.version import InvalidVersion, Version
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
    ACP_SERVER_TAG_KEY,
    ARCHIVE_WORKSPACE_PATH_TAG_KEY,
    AgentType,
    AppConversation,
    AppConversationInfo,
    AppConversationPage,
    AppConversationSortOrder,
    AppConversationStartRequest,
    AppConversationStartTask,
    AppConversationStartTaskStatus,
    AppConversationUpdateRequest,
    ConversationTrigger,
    PluginSpec,
    SandboxGroupingStrategy,
)
from openhands.app_server.app_conversation.app_conversation_service import (
    AppConversationService,
    AppConversationServiceInjector,
    ConversationExportAlreadyRunning,
    ConversationExportLockUnavailable,
    ConversationExportTooLarge,
)
from openhands.app_server.app_conversation.app_conversation_service_base import (
    AppConversationServiceBase,
    get_project_dir,
)
from openhands.app_server.app_conversation.app_conversation_start_task_service import (
    AppConversationStartTaskService,
)
from openhands.app_server.app_conversation.conversation_secret_enricher import (
    ConversationSecretEnricher,
)
from openhands.app_server.app_conversation.hook_loader import (
    load_hooks_from_agent_server,
)
from openhands.app_server.app_conversation.sql_app_conversation_info_service import (
    SQLAppConversationInfoService,
)
from openhands.app_server.config import (
    get_event_callback_service,
    resolve_provider_llm_base_url,
)
from openhands.app_server.errors import SandboxError
from openhands.app_server.event.event_service import EventService
from openhands.app_server.event_callback.event_callback_models import EventCallback
from openhands.app_server.event_callback.event_callback_service import (
    EventCallbackService,
)
from openhands.app_server.event_callback.set_title_callback_processor import (
    SetTitleCallbackProcessor,
)
from openhands.app_server.integrations.provider import PROVIDER_TOKEN_TYPE, ProviderType
from openhands.app_server.integrations.service_types import SuggestedTask
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
from openhands.app_server.sandbox.sandbox_spec_service import (
    SandboxSpecService,
    get_agent_server_image,
    is_custom_agent_server_image,
)
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.services.jwt_service import JwtService
from openhands.app_server.settings.llm_profiles import resolve_profile_llm
from openhands.app_server.settings.marketplace_composition import (
    load_composed_marketplaces,
    marketplace_plugin_loading_enabled,
)
from openhands.app_server.settings.settings_models import (
    MarketplaceRegistration,
    grouped_workspace_dir,
)
from openhands.app_server.user.user_context import UserContext
from openhands.app_server.user.user_models import UserInfo
from openhands.app_server.utils.docker_utils import (
    replace_localhost_hostname_for_docker,
)
from openhands.app_server.utils.git import ensure_valid_git_branch_name
from openhands.app_server.utils.import_utils import get_impl
from openhands.app_server.utils.llm_metadata import (
    get_llm_metadata,
    should_set_litellm_extra_body,
)
from openhands.app_server.utils.redis_lock import (
    LockError,
    RedisLockUnavailable,
    refresh_lock_periodically,
    try_acquire_redis_lock,
)
from openhands.sdk import Agent, AgentContext, LocalWorkspace
from openhands.sdk.hooks import HookConfig
from openhands.sdk.llm import LLM
from openhands.sdk.llm.llm_profile_store import PROFILE_NAME_REGEX
from openhands.sdk.plugin import PluginSource
from openhands.sdk.secret import LookupSecret, StaticSecret
from openhands.sdk.settings import ACPAgentSettings
from openhands.sdk.subagent import get_registered_agent_definitions
from openhands.sdk.tool.builtins import SwitchLLMTool
from openhands.sdk.utils.redact import (
    redact_api_key_literals,
    redact_text_secrets,
    sanitize_config,
)
from openhands.sdk.workspace.remote.async_remote_workspace import AsyncRemoteWorkspace
from openhands.tools.preset.default import (
    get_default_tools,
    register_builtins_agents,
)
from openhands.tools.preset.planning import (
    format_plan_structure,
    get_planning_tools,
)

_conversation_info_type_adapter = TypeAdapter(list[ConversationInfo | None])
_logger = logging.getLogger(__name__)

_EXPORT_LOCK_KEY_PREFIX = 'app_conversation_export'


class _StreamingZipBuffer(io.RawIOBase):
    """Small non-seekable writer used by zipfile to emit chunks incrementally.

    zipfile.ZipFile only needs write() and tell() from its underlying file
    object when writing to a non-seekable stream.  Everything else
    (flush, writable, seekable) is either unused by zipfile or already
    handled by the io.RawIOBase defaults.
    """

    def __init__(self):
        self._chunks: list[bytes] = []
        self._position = 0

    def write(self, data) -> int:
        chunk = bytes(data)
        if chunk:
            self._chunks.append(chunk)
            self._position += len(chunk)
        return len(data)

    def tell(self) -> int:
        return self._position

    def drain(self) -> list[bytes]:
        chunks = self._chunks
        self._chunks = []
        return chunks


def _expected_sdk_version() -> str | None:
    """App's pinned openhands-sdk version, or None if its metadata is unresolvable."""
    try:
        return importlib.metadata.version('openhands-sdk')
    except importlib.metadata.PackageNotFoundError:
        return None


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

GIT_SHALLOW_CLONE_CONTEXT = """<GIT_WORKSPACE_CONTEXT>
The selected repository was cloned as a shallow clone. Git history may be incomplete. Before using operations that depend on full history, tags, merge bases, historical blame, or arbitrary commit checkout, run `git rev-parse --is-shallow-repository`. If full history is needed, run `git fetch --unshallow` or `git fetch --deepen=<n>`.
</GIT_WORKSPACE_CONTEXT>"""


def append_system_context(existing: str | None, block: str) -> str:
    if not existing:
        return block
    if block in existing:
        return existing
    return f'{existing.rstrip()}\n\n{block}'


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
    conversation_secret_enricher: ConversationSecretEnricher = field(
        default_factory=ConversationSecretEnricher
    )
    app_mode: str | None = None
    export_max_events: int = 10000
    export_lock_ttl_seconds: int = 3600
    export_lock_refresh_interval_seconds: int = 30
    export_lock_required: bool | None = None

    def _maybe_append_shallow_clone_context(
        self,
        user: UserInfo,
        selected_repository: str | None,
        system_message_suffix: str | None,
    ) -> str | None:
        if selected_repository and not bool(getattr(user, 'git_full_clone', False)):
            return append_system_context(
                system_message_suffix, GIT_SHALLOW_CLONE_CONTEXT
            )
        return system_message_suffix

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
        # Prefer the user's email as the Laminar trace user id so traces are
        # immediately attributable in the Laminar UI instead of showing only
        # a pseudo-anonymous internal id. Falls back to ``user_id`` when no
        # email is available (e.g. OSS mode).
        laminar_user_id = await self.user_context.get_user_email() or user_id

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

            # Custom sandbox images can ship an incompatible openhands-sdk; fail
            # fast with a clear error instead of an opaque 500 on create.
            await self._verify_agent_server_version(
                agent_server_url, sandbox.session_api_key
            )

            # Mirror the user's LLM profiles into the sandbox so the agent's
            # built-in switch_llm tool can resolve them (in SaaS profiles live
            # on the app-server, not the sandbox filesystem). Before conversation
            # creation, so the tool is enabled; re-runs on every start/resume.
            await self._seed_sandbox_profiles(agent_server_url, sandbox.session_api_key)

            # Get the working dir
            sandbox_spec = await self.sandbox_spec_service.get_sandbox_spec(
                sandbox.sandbox_spec_id
            )
            assert sandbox_spec is not None

            # Set up conversation id
            conversation_id = request.conversation_id or uuid4()

            # Setup working dir based on grouping
            sandbox_grouping_strategy = await self._get_sandbox_grouping_strategy()
            working_dir = grouped_workspace_dir(
                sandbox_spec.working_dir,
                sandbox_grouping_strategy,
                conversation_id.hex,
            )

            # Run setup scripts
            remote_workspace = AsyncRemoteWorkspace(
                host=agent_server_url,
                api_key=sandbox.session_api_key,
                working_dir=working_dir,
            )
            async for updated_task in self.run_setup_scripts(
                task, sandbox, remote_workspace, agent_server_url, conversation_id
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
                    trigger=request.trigger,
                    remote_workspace=remote_workspace,
                    selected_repository=request.selected_repository,
                    selected_branch=request.selected_branch,
                    plugins=request.plugins,
                    api_secrets=request.secrets,
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
            # Inject ``user_id`` into the start-conversation body so the
            # agent-server can call ``Laminar.set_trace_user_id()`` and tag
            # traces with the authenticated user. The currently pinned
            # ``openhands-sdk`` release does not yet expose ``user_id`` on
            # ``StartConversationRequest`` (added in software-agent-sdk#3242),
            # so passing it to ``create_request(...)`` is silently dropped by
            # pydantic. The agent-server reads the field directly from the
            # request body, so injecting it here works regardless of whether
            # the local SDK model knows about it. Remove this once OpenHands
            # pins to an SDK release that exposes ``user_id`` on
            # ``StartConversationRequest``.
            if laminar_user_id:
                body_json['user_id'] = laminar_user_id
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

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                # A custom image that 500s on create is usually an openhands-sdk
                # mismatch /server_info couldn't reveal; add an actionable hint.
                if is_custom_agent_server_image():
                    expected = _expected_sdk_version()
                    raise SandboxError(
                        f'Conversation create failed (HTTP '
                        f'{exc.response.status_code}) on custom sandbox image '
                        f'{get_agent_server_image()}. Verify its openhands-sdk '
                        f'matches this release'
                        + (f' ({expected})' if expected else '')
                        + '; rebuild/re-pin the image if not.'
                    ) from exc
                raise
            info = ConversationInfo.model_validate(response.json())
            # Determine kind / llm_model from the request we built (its
            # ``agent`` is the source of truth here): the response echoes
            # the same agent back through the AgentBase discriminator.
            request_agent = start_conversation_request.agent
            tags: dict[str, str] = {}
            # Pin where the workspace was actually created so the delete-time
            # archive captures the right directory without re-deriving the path
            # from settings (e.g. grouping) that may change before delete.
            tags[ARCHIVE_WORKSPACE_PATH_TAG_KEY] = working_dir
            if request_agent.agent_kind == 'acp':
                llm_model = request_agent.acp_model
                agent_kind = 'acp'
                # Persist the active ACP provider key so the conversation UI
                # can resolve a brand label ("Claude Code", "Codex", …) via
                # the SDK registry without keeping a per-conversation column.
                # Surfaced to the UI as the projected ``acp_server`` field.
                acp_user = await self.user_context.get_user_info()
                if isinstance(acp_user.agent_settings, ACPAgentSettings):
                    tags[ACP_SERVER_TAG_KEY] = acp_user.agent_settings.acp_server
            else:
                llm_model = request_agent.llm.model
                agent_kind = 'openhands'

            app_conversation_info = AppConversationInfo(
                id=info.id,
                title=f'Conversation {info.id.hex[:5]}',
                sandbox_id=sandbox.id,
                created_by_user_id=user_id,
                llm_model=llm_model,
                agent_kind=agent_kind,
                tags=tags,
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
                        event_kind=processor.get_event_kind(),
                        processor=processor,
                    )
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
            task.detail = redact_text_secrets(redact_api_key_literals(str(exc)))
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
                sandbox,
                sandbox_id_to_conversation_ids.get(sandbox.id),
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
        conversation_ids: list[UUID],
    ) -> list[ConversationInfo]:
        """Get agent status for multiple conversations from the Agent Server.

        Uses the unified ``/api/conversations`` endpoint, which accepts both
        regular and ACP agents through the ``AgentBase`` discriminated union.
        """
        if not conversation_ids:
            return []

        agent_server_url = self._get_agent_server_url(sandbox)
        headers: dict[str, str] = {}
        if sandbox.session_api_key:
            headers['X-Session-API-Key'] = sandbox.session_api_key

        try:
            url = f'{agent_server_url.rstrip("/")}/api/conversations'
            response = await self.httpx_client.get(
                url,
                params={'ids': [str(c) for c in conversation_ids]},
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
            infos = _conversation_info_type_adapter.validate_python(data)
            return [c for c in infos if c]
        except httpx.HTTPStatusError:
            _logger.warning(
                f'Error getting conversation status from sandbox {sandbox.id}',
                exc_info=True,
            )
        except Exception:
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

    async def _verify_agent_server_version(
        self, agent_server_url: str, session_api_key: str | None
    ) -> None:
        """Fail fast with a clear error when an admin-pinned custom sandbox image
        runs a different openhands-sdk minor than this app, instead of the opaque
        500 the agent-server returns on create. Best-effort: only custom images are
        checked, and we fail open on anything we can't read."""
        if os.getenv('OH_SKIP_AGENT_SERVER_VERSION_CHECK', '').strip().lower() in (
            '1',
            'true',
            'yes',
        ):
            return
        # Proxy-default images move with the release; only custom-pinned can drift.
        if not is_custom_agent_server_image():
            return
        expected = _expected_sdk_version()
        if not expected:
            return
        try:
            headers = {'X-Session-API-Key': session_api_key} if session_api_key else {}
            resp = await self.httpx_client.get(
                f'{agent_server_url.rstrip("/")}/server_info',
                headers=headers,
                timeout=30.0,
            )
            resp.raise_for_status()
            reported = str(resp.json().get('sdk_version', '')).strip()
        except Exception:
            # 404 (image predates /server_info) or transient errors: can't verify,
            # so don't block — the create POST still surfaces a custom-image hint.
            _logger.warning(
                'Could not read /server_info to verify agent-server SDK version',
                exc_info=True,
            )
            return
        # Endpoint present but metadata missing -> nothing to compare against.
        if reported in ('', 'unknown'):
            return
        try:
            if Version(reported).release[:2] == Version(expected).release[:2]:
                return
        except InvalidVersion:
            return
        raise SandboxError(
            f'Sandbox image {get_agent_server_image()} runs openhands-sdk '
            f'{reported}, but this release requires {expected}. Rebuild/re-pin the '
            'custom sandbox image to a matching openhands-sdk, or set '
            'OH_SKIP_AGENT_SERVER_VERSION_CHECK=1 to bypass.'
        )

    async def _seed_sandbox_profiles(
        self, agent_server_url: str, session_api_key: str | None
    ) -> None:
        """Mirror the user's saved LLM profiles into the sandbox profile store.

        The agent's built-in ``switch_llm`` tool resolves profiles from the
        sandbox filesystem; in SaaS they live on the app-server, so without this
        the tool sees none. Upserts the current profiles (so adds/edits/renames
        land) and prunes ones deleted on the app-server, keeping the sandbox in
        sync. Best-effort: failures are logged, never raised, so they can't block
        the conversation.
        """
        # Imported lazily: settings_router transitively imports this service, so
        # a module-level import would be circular.
        from openhands.app_server.settings.settings_router import LITE_LLM_API_URL

        headers = {'X-Session-API-Key': session_api_key} if session_api_key else {}
        base_url = f'{agent_server_url}/api/profiles'
        try:
            user = await self.user_context.get_user_info()
            profiles = user.llm_profiles.profiles
            settings_llm = getattr(user.agent_settings, 'llm', None)
            fallback_api_key = getattr(settings_llm, 'api_key', None)
        except Exception:
            _logger.exception(
                'Failed to load profiles for sandbox %s', agent_server_url
            )
            return

        # Upsert each profile independently so one failure can't dark the rest.
        for name, profile_llm in profiles.items():
            # Org profile names aren't character-restricted, so skip any the
            # agent-server's store would reject — both to avoid a futile call and
            # to keep an exotic name (e.g. ``..``) from path-injecting the api-key
            # payload into a different request URL.
            if not PROFILE_NAME_REGEX.match(name):
                continue
            try:
                resolved = resolve_profile_llm(
                    profile_llm,
                    managed_proxy_url=LITE_LLM_API_URL,
                    fallback_api_key=fallback_api_key,
                )
                response = await self.httpx_client.post(
                    f'{base_url}/{name}',
                    json={
                        'include_secrets': True,
                        'llm': resolved.model_dump(
                            mode='json',
                            exclude_none=True,
                            context={'expose_secrets': True},
                        ),
                    },
                    headers=headers,
                    timeout=30.0,
                )
                response.raise_for_status()
            except Exception:
                _logger.warning(
                    'Failed to seed LLM profile %r into sandbox', name, exc_info=True
                )

        # Prune profiles deleted/renamed on the app-server so the agent can't
        # switch to a stale one. Independent best-effort.
        try:
            listed = await self.httpx_client.get(
                base_url, headers=headers, timeout=30.0
            )
            listed.raise_for_status()
            stored = {p['name'] for p in listed.json().get('profiles', [])}
            for stale_name in stored - set(profiles):
                await self.httpx_client.delete(
                    f'{base_url}/{stale_name}', headers=headers, timeout=30.0
                )
        except Exception:
            _logger.warning('Failed to prune sandbox profiles', exc_info=True)

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
        if provider_tokens:
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
                    static_token = await self.user_context.get_latest_token(
                        provider_type
                    )
                    if static_token:
                        secrets[secret_name] = StaticSecret(
                            value=SecretStr(static_token), description=description
                        )

        return secrets

    async def _setup_conversation_secrets(
        self,
        user: UserInfo,
        trigger: ConversationTrigger | None,
        system_message_suffix: str | None,
    ) -> tuple[dict, str | None]:
        """Set up custom, git provider, and integration-scoped secrets.

        Args:
            user: User information containing authentication details
            trigger: Trigger that started the conversation.
            system_message_suffix: Current system message suffix.

        Returns:
            Tuple of secrets and updated system message suffix.
        """
        secrets = await self._setup_secrets_for_git_providers(user)

        enrichment = await self.conversation_secret_enricher.enrich(
            user_context=self.user_context,
            user=user,
            trigger=trigger,
            system_message_suffix=system_message_suffix,
            web_url=self.web_url,
            jwt_service=self.jwt_service,
            access_token_hard_timeout=self.access_token_hard_timeout,
        )
        for name, source in enrichment.secrets.items():
            if name in secrets:
                _logger.warning(
                    'Integration-provided secret %r overrides existing secret', name
                )
            secrets[name] = source

        return secrets, enrichment.system_message_suffix

    def _configure_llm(self, user: UserInfo, llm_model: str | None) -> LLM:
        """Configure LLM settings.

        Starts from the user's saved LLM configuration and overrides only
        the fields that the server needs to resolve (model name, base URL,
        and usage ID).  All other user-configured fields (e.g.
        ``reasoning_effort``, ``extended_thinking_budget``, ``drop_params``)
        are preserved so that they reach the agent-server unchanged.

        Args:
            user: User information containing LLM preferences
            llm_model: Optional specific model to use, falls back to user default

        Returns:
            Configured LLM instance
        """
        model: str = (
            llm_model
            or user.agent_settings.llm.model
            or LLM.model_fields['model'].default
        )

        base_url = resolve_provider_llm_base_url(
            model,
            user.agent_settings.llm.base_url,
            provider_base_url=self.openhands_provider_base_url,
        )

        return user.agent_settings.llm.model_copy(
            update={
                'model': model,
                'base_url': base_url,
                'api_key': user.agent_settings.llm.api_key,
                'usage_id': 'agent',
                # Force streaming on (the SDK LLM defaults stream=False).
                'stream': True,
            }
        )

    async def _add_system_mcp_servers(
        self, mcp_servers: dict[str, Any], conversation_id: UUID
    ) -> None:
        """Add system-generated MCP servers (default OpenHands server).

        The default server includes the Tavily search proxy if configured.
        Tavily search is proxied through the app server to avoid exposing
        the API key to sandboxes.

        Args:
            mcp_servers: Dictionary to add servers to
            conversation_id: Conversation ID forwarded to the OpenHands MCP server
        """
        if not self.web_url:
            return

        # Add default OpenHands MCP server (includes Tavily proxy if configured)
        mcp_url = f'{self.web_url}/mcp/mcp'
        mcp_servers['default'] = {
            'url': mcp_url,
            'headers': {'X-OpenHands-ServerConversation-ID': str(conversation_id)},
        }

        # Add API key if available
        mcp_api_key = await self.user_context.get_mcp_api_key()
        if mcp_api_key:
            mcp_servers['default']['headers']['X-Session-API-Key'] = mcp_api_key

    def _merge_custom_mcp_config(
        self, mcp_servers: dict[str, Any], user: UserInfo
    ) -> None:
        """Merge custom MCP configuration from user settings.

        Args:
            mcp_servers: Dictionary to add servers to
            user: User information containing custom MCP config
        """
        if isinstance(user.agent_settings, ACPAgentSettings):
            return

        sdk_mcp = user.agent_settings.mcp_config
        if not sdk_mcp or not sdk_mcp.mcpServers:
            return

        try:
            count = len(sdk_mcp.mcpServers)
            _logger.info(
                f'Loading custom MCP config from user settings: {count} servers'
            )

            for name, server in sdk_mcp.mcpServers.items():
                mcp_servers[name] = server.model_dump(exclude_none=True)

            _logger.info(
                f'Successfully merged custom MCP config: added {count} servers'
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

        # Add system-generated servers (default MCP server with Tavily proxy)
        await self._add_system_mcp_servers(mcp_servers, conversation_id)

        # Merge custom servers from user settings
        self._merge_custom_mcp_config(mcp_servers, user)

        # Wrap in the mcpServers structure required by the SDK
        mcp_config = {'mcpServers': mcp_servers} if mcp_servers else {}
        _logger.info(f'Final MCP configuration: {sanitize_config(mcp_config)}')

        return llm, mcp_config

    @staticmethod
    def _apply_server_agent_overrides(
        agent: Agent,
        agent_type: AgentType,
        mcp_config: dict,
        conversation_id: UUID,
        user_id: str | None,
    ) -> Agent:
        """Apply server-only fields that have no place in ``AgentSettings``.

        * System-prompt filename / kwargs (planning vs default agent).
        * LLM tracing metadata for SaaS analytics.
        """
        overrides: dict[str, Any] = {}
        if agent_type == AgentType.PLAN:
            overrides['system_prompt_filename'] = 'system_prompt_planning.j2'
            overrides['system_prompt_kwargs'] = {
                'plan_structure': format_plan_structure()
            }
        else:
            overrides['system_prompt_kwargs'] = {'cli_mode': False}

        # LLM tracing metadata for openhands/ models
        if should_set_litellm_extra_body(agent.llm.model):
            llm_metadata = get_llm_metadata(
                model_name=agent.llm.model,
                llm_type=agent.llm.usage_id or 'agent',
                conversation_id=conversation_id,
                user_id=user_id,
            )
            overrides['llm'] = agent.llm.model_copy(
                update={'litellm_extra_body': {'metadata': llm_metadata}}
            )

        # Condenser LLM tracing
        if agent.condenser is not None and hasattr(agent.condenser, 'llm'):
            condenser_llm = agent.condenser.llm
            condenser_updates: dict[str, Any] = {}
            if not condenser_llm.usage_id or condenser_llm.usage_id == 'agent':
                condenser_updates['usage_id'] = 'condenser'
            if should_set_litellm_extra_body(condenser_llm.model):
                condenser_metadata = get_llm_metadata(
                    model_name=condenser_llm.model,
                    llm_type='condenser',
                    conversation_id=conversation_id,
                    user_id=user_id,
                )
                condenser_updates['litellm_extra_body'] = {
                    'metadata': condenser_metadata
                }
            if condenser_updates:
                updated_condenser = agent.condenser.model_copy(
                    update={'llm': condenser_llm.model_copy(update=condenser_updates)}
                )
                overrides['condenser'] = updated_condenser

        return agent.model_copy(update=overrides)

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

    @staticmethod
    async def _resolve_head_commit(
        remote_workspace: AsyncRemoteWorkspace, project_dir: str
    ) -> str:
        """Best-effort post-clone HEAD sha for the Laminar trace; '' on failure.

        ``--verify --quiet`` guarantees stdout is a validated object id (never
        the literal ``HEAD`` an unborn repo would echo) and fails silently
        rather than logging a ``fatal:`` line. The short timeout keeps this
        trace-only lookup from delaying conversation startup if the workspace is
        unresponsive — the metadata is simply dropped instead.
        """
        try:
            result = await remote_workspace.execute_command(
                'git rev-parse --verify --quiet HEAD', project_dir, timeout=10.0
            )
        except Exception as e:
            _logger.debug('HEAD commit lookup for trace metadata failed: %s', e)
            return ''
        return result.stdout.strip() if not result.exit_code else ''

    async def _build_observability_metadata(
        self,
        remote_workspace: AsyncRemoteWorkspace | None,
        project_dir: str,
        selected_repository: str | None,
        selected_branch: str | None,
        git_provider: ProviderType | None,
    ) -> dict[str, str]:
        """Repo identity for the Laminar trace so trajectories are searchable
        by repo / branch / commit (the trace UI has no DB to join against).

        Shared by the OpenHands and ACP request builders. The commit is the
        post-clone HEAD, resolved best-effort from the (already-cloned)
        workspace; omitted entirely for a repo-less conversation.
        """
        commit = ''
        if selected_repository and remote_workspace is not None:
            commit = await self._resolve_head_commit(remote_workspace, project_dir)
        return {
            key: value
            for key, value in (
                ('repo', selected_repository),
                ('branch', selected_branch),
                ('git_provider', git_provider.value if git_provider else None),
                ('commit', commit),
            )
            if value
        }

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
        trigger: ConversationTrigger | None = None,
        remote_workspace: AsyncRemoteWorkspace | None = None,
        selected_repository: str | None = None,
        selected_branch: str | None = None,
        plugins: list[PluginSpec] | None = None,
        api_secrets: dict[str, SecretStr] | None = None,
    ) -> StartConversationRequest:
        """Build a complete StartConversationRequest for a user.

        Resolves LLM, MCP, tools, secrets and agent context, then
        builds the ``Agent`` via ``AgentSettings.create_agent()``.
        Server-only overrides (system prompts, LLM tracing metadata,
        skills, hooks) are applied to the agent after creation.
        Finally delegates to ``ConversationSettings.create_request()``.

        For ACP agent settings, routes to ``_build_acp_start_conversation_request``.

        Args:
            sandbox: Sandbox information
            conversation_id: Unique conversation identifier
            initial_message: Optional initial message to send
            system_message_suffix: Optional suffix for system message
            git_provider: Optional git provider type
            working_dir: Working directory path
            agent_type: Type of agent (DEFAULT or PLAN)
            llm_model: Optional specific LLM model to use
            trigger: Optional conversation trigger.
            remote_workspace: Optional remote workspace instance
            selected_repository: Optional repository name
            plugins: Optional list of plugins to load
            api_secrets: Optional secrets passed directly via the API.
                These are merged with existing secrets (from database
                and git providers), with API-provided secrets taking
                precedence.
        """
        user = await self.user_context.get_user_info()

        # Compose instance + org + user marketplaces once for both arms below.
        # Enabled by default; inert (None) only when ENABLE_MARKETPLACE_PLUGIN_LOADING
        # is explicitly disabled or no marketplaces are configured.
        registered_marketplaces = await self._resolve_registered_marketplaces(user)

        # Route ACP agent settings to the ACP-specific builder
        if isinstance(user.agent_settings, ACPAgentSettings):
            acp_request = await self._build_acp_start_conversation_request(
                sandbox=sandbox,
                conversation_id=conversation_id,
                initial_message=initial_message,
                system_message_suffix=system_message_suffix,
                trigger=trigger,
                working_dir=working_dir,
                git_provider=git_provider,
                selected_repository=selected_repository,
                selected_branch=selected_branch,
                remote_workspace=remote_workspace,
                plugins=plugins,
                api_secrets=api_secrets,
            )
            if remote_workspace:
                acp_request = await self._load_skills_onto_request(
                    acp_request,
                    sandbox,
                    remote_workspace,
                    selected_repository,
                    get_project_dir(working_dir, selected_repository),
                    user.disabled_skills,
                    registered_marketplaces,
                )
            return acp_request

        project_dir = get_project_dir(working_dir, selected_repository)
        workspace = LocalWorkspace(working_dir=project_dir)

        # --- secrets --------------------------------------------------------
        # Start with secrets from git providers and database
        secrets, system_message_suffix = await self._setup_conversation_secrets(
            user,
            trigger,
            system_message_suffix,
        )

        # Merge API-provided secrets (they take precedence over existing ones)
        if api_secrets:
            from openhands.app_server.constants import (
                validate_secret_name,
                validate_secrets_dict,
            )

            # Validate overall dict size limits first
            # Cast to Mapping for mypy compatibility (Mapping is covariant in value type)
            validate_secrets_dict(cast('Mapping[str, object]', api_secrets))

            for name, value in api_secrets.items():
                validate_secret_name(name)
                if name in secrets:
                    _logger.warning(
                        'API-provided secret %r overrides existing secret', name
                    )
                secrets[name] = StaticSecret(value=value)

        system_message_suffix = self._maybe_append_shallow_clone_context(
            user, selected_repository, system_message_suffix
        )

        # --- LLM + MCP -----------------------------------------------------
        llm, mcp_config = await self._configure_llm_and_mcp(
            user, llm_model, conversation_id
        )

        # --- system_message_suffix (planning-agent prefix) ------------------
        effective_suffix = system_message_suffix
        if agent_type == AgentType.PLAN:
            if system_message_suffix:
                effective_suffix = (
                    f'{PLANNING_AGENT_INSTRUCTION}\n\n{system_message_suffix}'
                )
            else:
                effective_suffix = PLANNING_AGENT_INSTRUCTION

        # --- web host context -----------------------------------------------
        # Add WEB_HOST to agent context if available
        if self.web_url:
            web_host_context = f'<HOST>\n{self.web_url}\n</HOST>'
            if effective_suffix:
                effective_suffix = f'{effective_suffix}\n\n{web_host_context}'
            else:
                effective_suffix = web_host_context

        # --- tools ----------------------------------------------------------
        agent_definitions: list[Any] = []
        if agent_type == AgentType.PLAN:
            plan_path = None
            if project_dir:
                plan_path = self._compute_plan_path(project_dir, git_provider)
            tools = get_planning_tools(plan_path=plan_path)
        else:
            register_builtins_agents(enable_browser=True)
            tools = get_default_tools(
                enable_browser=True,
                enable_sub_agents=user.agent_settings.enable_sub_agents,
            )
            if user.agent_settings.enable_sub_agents:
                agent_definitions = list(get_registered_agent_definitions())

        # --- build AgentSettings and create agent ---------------------------
        from fastmcp.mcp_config import MCPConfig

        configured_agent_settings = user.agent_settings.model_copy(
            update={
                'llm': llm,
                'tools': tools,
                'mcp_config': MCPConfig(**mcp_config) if mcp_config else None,
                'agent_context': AgentContext(
                    system_message_suffix=effective_suffix,
                    secrets=secrets,
                ),
            }
        )
        agent = configured_agent_settings.create_agent()

        # SaaS profiles live on the user/org record, not the sandbox
        # filesystem, so we attach the agent's built-in switch_llm tool
        # ourselves rather than relying on create_agent()'s gating. Enabled
        # whenever there are at least two valid saved profiles (a switch needs
        # a target).
        valid_profile_names = [
            name
            for name in user.llm_profiles.profiles
            if PROFILE_NAME_REGEX.match(name)
        ]
        if (
            len(valid_profile_names) >= 2
            and SwitchLLMTool.__name__ not in agent.include_default_tools
        ):
            agent = agent.model_copy(
                update={
                    'include_default_tools': [
                        *agent.include_default_tools,
                        SwitchLLMTool.__name__,
                    ]
                }
            )

        agent = self._apply_server_agent_overrides(
            agent, agent_type, mcp_config, conversation_id, user.id
        )

        # --- hooks (require remote workspace; must precede request build) -----
        hook_config: HookConfig | None = None
        if remote_workspace:
            try:
                _logger.debug(
                    f'Attempting to load hooks from workspace: '
                    f'project_dir={project_dir}'
                )
                hook_config = await self._load_hooks_from_workspace(
                    remote_workspace, project_dir
                )
                if hook_config:
                    _logger.debug(
                        f'Successfully loaded hooks: {sanitize_config(hook_config.model_dump())}'
                    )
                else:
                    _logger.debug('No hooks found in workspace')
            except Exception as e:
                _logger.warning(f'Failed to load hooks: {e}', exc_info=True)

        # --- plugins --------------------------------------------------------
        final_initial_message = self._construct_initial_message_with_plugin_params(
            initial_message, plugins
        )
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

        # --- populate ConversationSettings and build request ----------------
        conv_settings = user.conversation_settings.model_copy(
            update={
                'agent_settings': configured_agent_settings,
                'workspace': workspace,
                'conversation_id': conversation_id,
                'initial_message': final_initial_message,
                'agent_definitions': agent_definitions,
                'plugins': sdk_plugins,
                'hook_config': hook_config,
            }
        )

        # Pass agent explicitly — it has server-only overrides (system
        # prompts, LLM metadata, skills) applied after create_agent().
        # ``user_id`` is forwarded so the agent-server can attach it to
        # observability spans (see software-agent-sdk#3242). We prefer the
        # user's email so Laminar traces are immediately attributable, and
        # fall back to the internal user id when no email is available.
        # The kwarg is dropped silently by pydantic on SDK versions that
        # don't yet expose the field; the start-conversation POST also
        # injects it directly into the JSON body as a forward-compatible
        # fallback.
        laminar_user_id = await self.user_context.get_user_email() or user.id
        observability_metadata = await self._build_observability_metadata(
            remote_workspace,
            project_dir,
            selected_repository,
            selected_branch,
            git_provider,
        )
        create_kwargs: dict[str, Any] = {'agent': agent, 'user_id': laminar_user_id}
        if observability_metadata:
            create_kwargs['observability_metadata'] = observability_metadata
        request = conv_settings.create_request(
            StartConversationRequest, **create_kwargs
        )

        # --- skills (require remote workspace) ------------------------------
        if remote_workspace:
            request = await self._load_skills_onto_request(
                request,
                sandbox,
                remote_workspace,
                selected_repository,
                project_dir,
                user.disabled_skills,
                registered_marketplaces,
            )

        return request

    async def _resolve_registered_marketplaces(
        self, user: UserInfo
    ) -> list[MarketplaceRegistration] | None:
        """Compose instance + org + user marketplaces for conversation start.

        Enabled by default; returns ``None`` (feature inert) when
        ENABLE_MARKETPLACE_PLUGIN_LOADING is explicitly disabled. Never raises:
        any failure degrades to no marketplaces so it can never block
        conversation creation.
        """
        if not marketplace_plugin_loading_enabled():
            return None
        try:
            from openhands.app_server.shared import SettingsStoreImpl

            user_id = await self.user_context.get_user_id()
            settings_store = await SettingsStoreImpl.get_instance(user_id)
            composed = await load_composed_marketplaces(
                user_id, user.registered_marketplaces, settings_store
            )
            return composed.all or None
        except Exception as e:
            _logger.warning(
                'Failed to compose marketplaces for conversation start: %s', e
            )
            return None

    async def _load_skills_onto_request(
        self,
        request: StartConversationRequest,
        sandbox: SandboxInfo,
        remote_workspace: AsyncRemoteWorkspace,
        selected_repository: str | None,
        project_dir: str,
        disabled_skills: list[str] | None,
        registered_marketplaces: list[MarketplaceRegistration] | None = None,
    ) -> StartConversationRequest:
        """Load workspace skills onto a conversation request's agent.

        Used by both the LLM and ACP arms of
        ``_build_start_conversation_request_for_user`` so that skill-loading
        semantics only need to change in one place.
        """
        try:
            updated_agent = await self._load_skills_and_update_agent(
                sandbox,
                request.agent,
                remote_workspace,
                selected_repository,
                project_dir,
                disabled_skills=disabled_skills,
                registered_marketplaces=registered_marketplaces,
            )
            return request.model_copy(update={'agent': updated_agent})
        except Exception as e:
            _logger.warning(f'Failed to load skills: {e}', exc_info=True)
            return request

    async def _build_acp_start_conversation_request(
        self,
        sandbox: SandboxInfo,
        conversation_id: UUID,
        initial_message: SendMessageRequest | None,
        working_dir: str,
        system_message_suffix: str | None = None,
        trigger: ConversationTrigger | None = None,
        git_provider: ProviderType | None = None,
        selected_repository: str | None = None,
        selected_branch: str | None = None,
        remote_workspace: AsyncRemoteWorkspace | None = None,
        plugins: list[PluginSpec] | None = None,
        api_secrets: dict[str, SecretStr] | None = None,
    ) -> StartConversationRequest:
        """Build a StartConversationRequest for ACP agent conversations.

        User secrets (Secrets panel + git provider tokens) flow through
        ``request.secrets`` — the canonical cipher-protected wire channel.
        In SaaS mode each secret is a ``LookupSecret`` pointing at
        ``/api/v1/webhooks/custom-secret`` with a per-secret scoped JWT, so
        values are never materialised in this process.  In OSS mode (no
        ``web_url``) they remain ``StaticSecret``.  Secrets are passed
        directly as ``secrets=`` to ``create_request()``; no ``AgentContext``
        relay is needed (software-agent-sdk #3464;
        OpenHands/agent-canvas#1039).

        Args:
            sandbox: Sandbox information
            conversation_id: Unique conversation identifier
            initial_message: Optional initial message to send
            working_dir: Working directory path
            system_message_suffix: Optional suffix for system message.
            trigger: Optional conversation trigger.
            git_provider: Optional git provider type
            selected_repository: Optional repository name
            selected_branch: Optional branch name
            remote_workspace: Optional remote workspace instance, used to
                resolve the HEAD commit for the Laminar trace metadata.
            plugins: Optional list of plugins to load
            api_secrets: Optional secrets passed directly via the API.
        """
        user = await self.user_context.get_user_info()

        project_dir = get_project_dir(working_dir, selected_repository)
        workspace = LocalWorkspace(working_dir=project_dir)

        # --- secrets --------------------------------------------------------
        # ACP secrets must be StaticSecrets — LookupSecrets with JWT headers
        # (e.g. X-Access-Token) are redacted by the SDK serializer because
        # "TOKEN" matches SECRET_KEY_PATTERNS, leaving headers: {} and
        # causing provider auth to silently fail at subprocess launch.
        # Use raw custom secrets, static git provider tokens, and static
        # integration-scoped secrets for ACP conversations.
        secrets: dict = await self.user_context.get_secrets()
        provider_tokens = cast(
            PROVIDER_TOKEN_TYPE | None,
            await self.user_context.get_provider_tokens(),
        )
        if provider_tokens:
            for provider_type, provider_token in provider_tokens.items():
                if not provider_token.token:
                    continue
                secret_name = f'{provider_type.name}_TOKEN'
                static_token = await self.user_context.get_latest_token(provider_type)
                if static_token:
                    secrets[secret_name] = StaticSecret(
                        value=SecretStr(static_token),
                        description=f'{provider_type.name} authentication token',
                    )

        enrichment = await self.conversation_secret_enricher.enrich(
            user_context=self.user_context,
            user=user,
            trigger=trigger,
            system_message_suffix=system_message_suffix,
            web_url=None,
            jwt_service=self.jwt_service,
            access_token_hard_timeout=self.access_token_hard_timeout,
        )
        for name, source in enrichment.secrets.items():
            if name in secrets:
                _logger.warning(
                    'Integration-provided secret %r overrides existing secret', name
                )
            secrets[name] = source
        system_message_suffix = enrichment.system_message_suffix

        if api_secrets:
            from openhands.app_server.constants import (
                validate_secret_name,
                validate_secrets_dict,
            )

            validate_secrets_dict(cast('Mapping[str, object]', api_secrets))
            for name, value in api_secrets.items():
                validate_secret_name(name)
                if name in secrets:
                    _logger.warning(
                        'API-provided secret %r overrides existing secret', name
                    )
                secrets[name] = StaticSecret(value=value)

        system_message_suffix = self._maybe_append_shallow_clone_context(
            user, selected_repository, system_message_suffix
        )

        # --- build the ACP agent ------------------------------------------
        acp_settings = user.agent_settings  # already verified to be ACPAgentSettings
        assert isinstance(acp_settings, ACPAgentSettings)

        # Isolate the CLI data dir onto the durable /workspace tree so the SDK
        # self-resumes the provider session (session/load from base_state.json)
        # across pause/resume — matching the regular-agent lifecycle (#1274).
        # Strip llm.api_key/base_url to prevent proxy settings from leaking
        # into the subprocess env (ACP CLIs handle their own LLM calls).
        settings_update: dict[str, Any] = {
            'acp_isolate_data_dir': True,
            'llm': acp_settings.llm.model_copy(
                update={'api_key': None, 'base_url': None}
            ),
        }
        if system_message_suffix:
            settings_update['agent_context'] = AgentContext(
                system_message_suffix=system_message_suffix
            )
        acp_settings_for_agent = acp_settings.model_copy(update=settings_update)
        acp_agent = acp_settings_for_agent.create_agent()

        sdk_plugins: list[PluginSource] | None = None
        if plugins:
            sdk_plugins = [
                PluginSource(source=p.source, ref=p.ref, repo_path=p.repo_path)
                for p in plugins
            ]

        # Mirror the regular path: populate ConversationSettings and delegate
        # to create_request() so that max_iterations, confirmation_mode, and
        # security_analyzer flow through to ACP conversations too.
        conv_settings = user.conversation_settings.model_copy(
            update={
                'workspace': workspace,
                'conversation_id': conversation_id,
                'initial_message': self._construct_initial_message_with_plugin_params(
                    initial_message, plugins
                ),
                'plugins': sdk_plugins,
            }
        )
        # ``user_id`` is forwarded for observability; see the LLM path above
        # for behavior on SDK versions that don't yet expose the field. We
        # prefer email over the internal id so Laminar traces are immediately
        # attributable, falling back to ``user.id`` when no email is available.
        laminar_user_id = await self.user_context.get_user_email() or user.id
        observability_metadata = await self._build_observability_metadata(
            remote_workspace,
            project_dir,
            selected_repository,
            selected_branch,
            git_provider,
        )
        create_kwargs: dict[str, Any] = {
            'agent': acp_agent,
            'user_id': laminar_user_id,
            'secrets': secrets,
        }
        if observability_metadata:
            create_kwargs['observability_metadata'] = observability_metadata
        return conv_settings.create_request(StartConversationRequest, **create_kwargs)

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

    async def _get_conversation_export_info(
        self, conversation_id: UUID
    ) -> AppConversationInfo:
        conversation_info = (
            await self.app_conversation_info_service.get_app_conversation_info(
                conversation_id
            )
        )
        if not conversation_info:
            raise ValueError(f'Conversation not found: {conversation_id}')
        return conversation_info

    async def _validate_conversation_export_size(self, conversation_id: UUID):
        if self.export_max_events <= 0:
            return

        event_count = await self.event_service.count_events(conversation_id)
        if event_count > self.export_max_events:
            raise ConversationExportTooLarge(
                f'Conversation export contains {event_count} events, '
                f'exceeding the limit of {self.export_max_events}'
            )

    def _conversation_export_lock_key(self, conversation_id: UUID) -> str:
        return f'{_EXPORT_LOCK_KEY_PREFIX}:{conversation_id.hex}'

    def _conversation_export_lock_refresh_interval(self) -> int:
        ttl_seconds = max(1, self.export_lock_ttl_seconds)
        configured_interval = max(1, self.export_lock_refresh_interval_seconds)
        return min(configured_interval, max(1, ttl_seconds // 2))

    def _conversation_export_lock_required(self) -> bool:
        if self.export_lock_required is not None:
            return self.export_lock_required
        return (self.app_mode or '').lower() == 'saas'

    async def _stream_conversation_zip(
        self, conversation_id: UUID, conversation_info: AppConversationInfo
    ) -> AsyncGenerator[bytes, None]:
        zip_buffer = _StreamingZipBuffer()
        with zipfile.ZipFile(
            cast(BinaryIO, zip_buffer), 'w', zipfile.ZIP_DEFLATED
        ) as zipf:
            zipf.writestr('meta.json', conversation_info.model_dump_json(indent=2))
            for chunk in zip_buffer.drain():
                yield chunk

            i = 0
            async for event in self.event_service.iter_events_for_export(
                conversation_id
            ):
                event_filename = f'event_{i:06d}_{event.id}.json'
                event_data = event.model_dump(mode='json')
                event_json = json.dumps(event_data, indent=2)
                zipf.writestr(event_filename, event_json)
                for chunk in zip_buffer.drain():
                    yield chunk
                i += 1

        for chunk in zip_buffer.drain():
            yield chunk

    async def open_conversation_export(
        self, conversation_id: UUID
    ) -> AsyncGenerator[bytes, None]:
        """Prepare a locked streaming conversation trajectory export."""
        conversation_info = await self._get_conversation_export_info(conversation_id)
        lock_key = self._conversation_export_lock_key(conversation_id)
        lock_unavailable = False
        try:
            export_lock = await try_acquire_redis_lock(
                lock_key, max(1, self.export_lock_ttl_seconds)
            )
        except RedisLockUnavailable as e:
            if self._conversation_export_lock_required():
                raise ConversationExportLockUnavailable(
                    f'Could not acquire export lock for conversation {conversation_id}'
                ) from e
            _logger.warning(
                'conversation_export:lock_unavailable_proceeding_without_lock',
                extra={'conversation_id': str(conversation_id)},
            )
            export_lock = None
            lock_unavailable = True

        if export_lock is None and not lock_unavailable:
            raise ConversationExportAlreadyRunning(
                f'Conversation export already running: {conversation_id}'
            )

        try:
            await self._validate_conversation_export_size(conversation_id)
        except Exception:
            if export_lock:
                try:
                    await export_lock.release()
                except LockError:
                    pass
            raise

        refresh_interval = self._conversation_export_lock_refresh_interval()

        async def stream():
            # Refresh the lock in a background task so the streaming loop stays
            # simple and lock maintenance doesn't block chunk generation.
            refresh_task = (
                asyncio.create_task(
                    refresh_lock_periodically(export_lock, refresh_interval)
                )
                if export_lock
                else None
            )
            try:
                async for chunk in self._stream_conversation_zip(
                    conversation_id, conversation_info
                ):
                    yield chunk
            finally:
                if refresh_task is not None:
                    refresh_task.cancel()
                if export_lock:
                    try:
                        await export_lock.release()
                    except LockError:
                        _logger.warning(
                            'conversation_export:lock_release_failed',
                            extra={'conversation_id': str(conversation_id)},
                        )

        return stream()

    async def export_conversation(self, conversation_id: UUID) -> bytes:
        """Download a conversation trajectory as a zip file.

        Args:
            conversation_id: The UUID of the conversation to download.

        Returns the zip file as bytes.
        """
        conversation_info = await self._get_conversation_export_info(conversation_id)
        await self._validate_conversation_export_size(conversation_id)

        chunks = []
        async for chunk in self._stream_conversation_zip(
            conversation_id, conversation_info
        ):
            chunks.append(chunk)

        return b''.join(chunks)


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
    export_max_events: int = Field(
        default=10000,
        description='The maximum number of events allowed in a conversation export',
    )
    export_lock_ttl_seconds: int = Field(
        default=3600,
        description='Redis lock TTL for a single conversation export',
    )
    export_lock_refresh_interval_seconds: int = Field(
        default=30,
        description='How often to refresh the Redis lock during a conversation export',
    )
    export_lock_required: bool | None = Field(
        default=None,
        description=(
            'Whether Redis export locking is required. Defaults to required in '
            'SAAS mode and best-effort elsewhere.'
        ),
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
            conversation_secret_enricher = ConversationSecretEnricher()
            try:
                from openhands.app_server.shared import server_config

                app_mode = (
                    server_config.app_mode.value if server_config.app_mode else None
                )
                enricher_cls = get_impl(
                    ConversationSecretEnricher,
                    server_config.conversation_secret_enricher_class,
                )
                conversation_secret_enricher = enricher_cls()
            except (ImportError, AttributeError):
                # If server_config is not available (e.g., in tests), continue without it
                pass

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
                conversation_secret_enricher=conversation_secret_enricher,
                app_mode=app_mode,
                export_max_events=self.export_max_events,
                export_lock_ttl_seconds=self.export_lock_ttl_seconds,
                export_lock_refresh_interval_seconds=(
                    self.export_lock_refresh_interval_seconds
                ),
                export_lock_required=self.export_lock_required,
            )
