import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from time import time
from typing import Any, AsyncGenerator, Sequence
from uuid import UUID, uuid4

import httpx
from fastapi import Request
from pydantic import Field, SecretStr, TypeAdapter

from openhands.agent_server.models import (
    ConversationInfo,
    NeverConfirm,
    SendMessageRequest,
    StartConversationRequest,
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
)
from openhands.app_server.app_conversation.app_conversation_service import (
    AppConversationService,
    AppConversationServiceInjector,
)
from openhands.app_server.app_conversation.app_conversation_service_base import (
    AppConversationServiceBase,
)
from openhands.app_server.app_conversation.app_conversation_start_task_service import (
    AppConversationStartTaskService,
)
from openhands.app_server.app_conversation.sql_app_conversation_info_service import (
    SQLAppConversationInfoService,
)
from openhands.app_server.config import get_event_callback_service
from openhands.app_server.errors import SandboxError
from openhands.app_server.event_callback.event_callback_models import EventCallback
from openhands.app_server.event_callback.event_callback_service import (
    EventCallbackService,
)
from openhands.app_server.event_callback.set_title_callback_processor import (
    SetTitleCallbackProcessor,
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
from openhands.experiments.experiment_manager import ExperimentManagerImpl
from openhands.integrations.provider import ProviderType
from openhands.sdk import Agent, AgentContext, LocalWorkspace
from openhands.sdk.llm import LLM
from openhands.sdk.secret import LookupSecret, StaticSecret
from openhands.sdk.security.confirmation_policy import AlwaysConfirm
from openhands.sdk.workspace.remote.async_remote_workspace import AsyncRemoteWorkspace
from openhands.server.types import AppMode
from openhands.tools.preset.default import (
    get_default_tools,
)
from openhands.tools.preset.planning import (
    format_plan_structure,
    get_planning_tools,
)

_conversation_info_type_adapter = TypeAdapter(list[ConversationInfo | None])
_logger = logging.getLogger(__name__)


@dataclass
class LiveStatusAppConversationService(AppConversationServiceBase):
    """AppConversationService which combines live status info from the sandbox with stored data."""

    user_context: UserContext
    app_conversation_info_service: AppConversationInfoService
    app_conversation_start_task_service: AppConversationStartTaskService
    event_callback_service: EventCallbackService
    sandbox_service: SandboxService
    sandbox_spec_service: SandboxSpecService
    jwt_service: JwtService
    sandbox_startup_timeout: int
    sandbox_startup_poll_frequency: int
    httpx_client: httpx.AsyncClient
    web_url: str | None
    openhands_provider_base_url: str | None
    access_token_hard_timeout: timedelta | None
    app_mode: str | None = None
    keycloak_auth_cookie: str | None = None
    tavily_api_key: str | None = None

    async def search_app_conversations(
        self,
        title__contains: str | None = None,
        created_at__gte: datetime | None = None,
        created_at__lt: datetime | None = None,
        updated_at__gte: datetime | None = None,
        updated_at__lt: datetime | None = None,
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
    ) -> int:
        return await self.app_conversation_info_service.count_app_conversation_info(
            title__contains=title__contains,
            created_at__gte=created_at__gte,
            created_at__lt=created_at__lt,
            updated_at__gte=updated_at__gte,
            updated_at__lt=updated_at__lt,
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

            # Run setup scripts
            remote_workspace = AsyncRemoteWorkspace(
                host=agent_server_url,
                api_key=sandbox.session_api_key,
                working_dir=sandbox_spec.working_dir,
            )
            async for updated_task in self.run_setup_scripts(
                task, sandbox, remote_workspace
            ):
                yield updated_task

            # Build the start request
            start_conversation_request = (
                await self._build_start_conversation_request_for_user(
                    sandbox,
                    request.initial_message,
                    request.system_message_suffix,
                    request.git_provider,
                    sandbox_spec.working_dir,
                    request.agent_type,
                    request.llm_model,
                    request.conversation_id,
                    remote_workspace=remote_workspace,
                    selected_repository=request.selected_repository,
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
            response = await self.httpx_client.post(
                f'{agent_server_url}/api/conversations',
                json=body_json,
                headers={'X-Session-API-Key': sandbox.session_api_key},
                timeout=self.sandbox_startup_timeout,
            )

            response.raise_for_status()
            info = ConversationInfo.model_validate(response.json())

            # Store info...
            user_id = await self.user_context.get_user_id()
            app_conversation_info = AppConversationInfo(
                id=info.id,
                title=f'Conversation {info.id.hex}',
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

            # Update the start task
            task.status = AppConversationStartTaskStatus.READY
            task.app_conversation_id = info.id
            yield task

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

    async def _wait_for_sandbox_start(
        self, task: AppConversationStartTask
    ) -> AsyncGenerator[AppConversationStartTask, None]:
        """Wait for sandbox to start and return info."""
        # Get the sandbox
        if not task.request.sandbox_id:
            sandbox = await self.sandbox_service.start_sandbox()
            task.sandbox_id = sandbox.id
        else:
            sandbox_info = await self.sandbox_service.get_sandbox(
                task.request.sandbox_id
            )
            if sandbox_info is None:
                raise SandboxError(f'Sandbox not found: {task.request.sandbox_id}')
            sandbox = sandbox_info

        # Update the listener
        task.status = AppConversationStartTaskStatus.WAITING_FOR_SANDBOX
        task.sandbox_id = sandbox.id
        yield task

        if sandbox.status == SandboxStatus.PAUSED:
            await self.sandbox_service.resume_sandbox(sandbox.id)
        if sandbox.status in (None, SandboxStatus.ERROR):
            raise SandboxError(f'Sandbox status: {sandbox.status}')
        if sandbox.status == SandboxStatus.RUNNING:
            return
        if sandbox.status != SandboxStatus.STARTING:
            raise SandboxError(f'Sandbox not startable: {sandbox.id}')

        start = time()
        while time() - start <= self.sandbox_startup_timeout:
            await asyncio.sleep(self.sandbox_startup_poll_frequency)
            sandbox_info = await self.sandbox_service.get_sandbox(sandbox.id)
            if sandbox_info is None:
                raise SandboxError(f'Sandbox not found: {sandbox.id}')
            if sandbox.status not in (SandboxStatus.STARTING, SandboxStatus.RUNNING):
                raise SandboxError(f'Sandbox not startable: {sandbox.id}')
            if sandbox_info.status == SandboxStatus.RUNNING:
                return
        raise SandboxError(f'Sandbox failed to start: {sandbox.id}')

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

    async def _setup_secrets_for_git_providers(self, user: UserInfo) -> dict:
        """Set up secrets for all git provider authentication.

        Args:
            user: User information containing authentication details

        Returns:
            Dictionary of secrets for the conversation
        """
        secrets = await self.user_context.get_secrets()

        # Get all provider tokens from user authentication
        provider_tokens = await self.user_context.get_provider_tokens()
        if not provider_tokens:
            return secrets

        # Create secrets for each provider token
        for provider_type, provider_token in provider_tokens.items():
            if not provider_token.token:
                continue

            secret_name = f'{provider_type.name}_TOKEN'

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

                # Include keycloak_auth cookie in headers if app_mode is SaaS
                if self.app_mode == 'saas' and self.keycloak_auth_cookie:
                    headers['Cookie'] = f'keycloak_auth={self.keycloak_auth_cookie}'

                secrets[secret_name] = LookupSecret(
                    url=self.web_url + '/api/v1/webhooks/secrets',
                    headers=headers,
                )
            else:
                # Use static token for environments without web URL access
                static_token = await self.user_context.get_latest_token(provider_type)
                if static_token:
                    secrets[secret_name] = StaticSecret(value=static_token)

        return secrets

    async def _configure_llm_and_mcp(
        self, user: UserInfo, llm_model: str | None
    ) -> tuple[LLM, dict]:
        """Configure LLM and MCP (Model Context Protocol) settings.

        Args:
            user: User information containing LLM preferences
            llm_model: Optional specific model to use, falls back to user default

        Returns:
            Tuple of (configured LLM instance, MCP config dictionary)
        """
        # Configure LLM
        model = llm_model or user.llm_model
        base_url = user.llm_base_url
        if model and model.startswith('openhands/'):
            base_url = user.llm_base_url or self.openhands_provider_base_url
        llm = LLM(
            model=model,
            base_url=base_url,
            api_key=user.llm_api_key,
            usage_id='agent',
        )

        # Configure MCP
        mcp_config: dict[str, Any] = {}
        if self.web_url:
            mcp_url = f'{self.web_url}/mcp/mcp'
            mcp_config = {
                'default': {
                    'url': mcp_url,
                }
            }

            # Add API key if available
            mcp_api_key = await self.user_context.get_mcp_api_key()
            if mcp_api_key:
                mcp_config['default']['headers'] = {
                    'X-Session-API-Key': mcp_api_key,
                }

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

            tavily_api_key = user_search_key or service_tavily_key

            if tavily_api_key:
                _logger.info('Adding search engine to MCP config')
                mcp_config['tavily'] = {
                    'url': f'https://mcp.tavily.com/mcp/?tavilyApiKey={tavily_api_key}'
                }
            else:
                _logger.info('No search engine API key found, skipping search engine')

        return llm, mcp_config

    def _create_agent_with_context(
        self,
        llm: LLM,
        agent_type: AgentType,
        system_message_suffix: str | None,
        mcp_config: dict,
        condenser_max_size: int | None,
    ) -> Agent:
        """Create an agent with appropriate tools and context based on agent type.

        Args:
            llm: Configured LLM instance
            agent_type: Type of agent to create (PLAN or DEFAULT)
            system_message_suffix: Optional suffix for system messages
            mcp_config: MCP configuration dictionary
            condenser_max_size: condenser_max_size setting

        Returns:
            Configured Agent instance with context
        """
        # Create condenser with user's settings
        condenser = self._create_condenser(llm, agent_type, condenser_max_size)

        # Create agent based on type
        if agent_type == AgentType.PLAN:
            agent = Agent(
                llm=llm,
                tools=get_planning_tools(),
                system_prompt_filename='system_prompt_planning.j2',
                system_prompt_kwargs={'plan_structure': format_plan_structure()},
                condenser=condenser,
                security_analyzer=None,
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

        # Add agent context
        agent_context = AgentContext(system_message_suffix=system_message_suffix)
        agent = agent.model_copy(update={'agent_context': agent_context})

        return agent

    async def _finalize_conversation_request(
        self,
        agent: Agent,
        conversation_id: UUID | None,
        user: UserInfo,
        workspace: LocalWorkspace,
        initial_message: SendMessageRequest | None,
        secrets: dict,
        sandbox: SandboxInfo,
        remote_workspace: AsyncRemoteWorkspace | None,
        selected_repository: str | None,
        working_dir: str,
    ) -> StartConversationRequest:
        """Finalize the conversation request with experiment variants and skills.

        Args:
            agent: The configured agent
            conversation_id: Optional conversation ID, generates new one if None
            user: User information
            workspace: Local workspace instance
            initial_message: Optional initial message for the conversation
            secrets: Dictionary of secrets for authentication
            sandbox: Sandbox information
            remote_workspace: Optional remote workspace for skills loading
            selected_repository: Optional repository name
            working_dir: Working directory path

        Returns:
            Complete StartConversationRequest ready for use
        """
        # Generate conversation ID if not provided
        conversation_id = conversation_id or uuid4()

        # Apply experiment variants
        agent = ExperimentManagerImpl.run_agent_variant_tests__v1(
            user.id, conversation_id, agent
        )

        # Load and merge skills if remote workspace is available
        if remote_workspace:
            try:
                agent = await self._load_skills_and_update_agent(
                    sandbox, agent, remote_workspace, selected_repository, working_dir
                )
            except Exception as e:
                _logger.warning(f'Failed to load skills: {e}', exc_info=True)
                # Continue without skills - don't fail conversation startup

        # Create and return the final request
        return StartConversationRequest(
            conversation_id=conversation_id,
            agent=agent,
            workspace=workspace,
            confirmation_policy=(
                AlwaysConfirm() if user.confirmation_mode else NeverConfirm()
            ),
            initial_message=initial_message,
            secrets=secrets,
        )

    async def _build_start_conversation_request_for_user(
        self,
        sandbox: SandboxInfo,
        initial_message: SendMessageRequest | None,
        system_message_suffix: str | None,
        git_provider: ProviderType | None,
        working_dir: str,
        agent_type: AgentType = AgentType.DEFAULT,
        llm_model: str | None = None,
        conversation_id: UUID | None = None,
        remote_workspace: AsyncRemoteWorkspace | None = None,
        selected_repository: str | None = None,
    ) -> StartConversationRequest:
        """Build a complete conversation request for a user.

        This method orchestrates the creation of a conversation request by:
        1. Setting up git provider secrets
        2. Configuring LLM and MCP settings
        3. Creating an agent with appropriate context
        4. Finalizing the request with skills and experiment variants
        """
        user = await self.user_context.get_user_info()
        workspace = LocalWorkspace(working_dir=working_dir)

        # Set up secrets for all git providers
        secrets = await self._setup_secrets_for_git_providers(user)

        # Configure LLM and MCP
        llm, mcp_config = await self._configure_llm_and_mcp(user, llm_model)

        # Create agent with context
        agent = self._create_agent_with_context(
            llm, agent_type, system_message_suffix, mcp_config, user.condenser_max_size
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
            working_dir,
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

    async def delete_app_conversation(self, conversation_id: UUID) -> bool:
        """Delete a V1 conversation and all its associated data.

        This method will also cascade delete all sub-conversations of the parent.

        Args:
            conversation_id: The UUID of the conversation to delete.
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
            # Delete from agent server if sandbox is running
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


class LiveStatusAppConversationServiceInjector(AppConversationServiceInjector):
    sandbox_startup_timeout: int = Field(
        default=120, description='The max timeout time for sandbox startup'
    )
    sandbox_startup_poll_frequency: int = Field(
        default=2, description='The frequency to poll for sandbox readiness'
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
            get_global_config,
            get_httpx_client,
            get_jwt_service,
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
            get_jwt_service(state, request) as jwt_service,
            get_httpx_client(state, request) as httpx_client,
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

            # Get app_mode and keycloak_auth cookie for SaaS mode
            app_mode = None
            keycloak_auth_cookie = None
            try:
                from openhands.server.shared import server_config

                app_mode = (
                    server_config.app_mode.value if server_config.app_mode else None
                )
                if request and server_config.app_mode == AppMode.SAAS:
                    keycloak_auth_cookie = request.cookies.get('keycloak_auth')
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
                jwt_service=jwt_service,
                sandbox_startup_timeout=self.sandbox_startup_timeout,
                sandbox_startup_poll_frequency=self.sandbox_startup_poll_frequency,
                httpx_client=httpx_client,
                web_url=web_url,
                openhands_provider_base_url=config.openhands_provider_base_url,
                access_token_hard_timeout=access_token_hard_timeout,
                app_mode=app_mode,
                keycloak_auth_cookie=keycloak_auth_cookie,
                tavily_api_key=tavily_api_key,
            )
