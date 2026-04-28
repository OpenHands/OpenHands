"""Jira Data Center view implementations and factory.

Views are responsible for:
- Holding the webhook payload and auth context
- Creating conversations using V1 app conversation system
"""

from dataclasses import dataclass, field
from uuid import UUID, uuid4

from integrations.jira_dc.jira_dc_types import (
    JiraDcViewInterface,
    StartingConvoException,
)
from integrations.jira_dc.jira_dc_v1_callback_processor import JiraDcV1CallbackProcessor
from integrations.models import JobContext
from integrations.resolver_context import ResolverUserContext
from integrations.resolver_org_router import resolve_org_for_repo
from integrations.utils import CONVERSATION_URL
from jinja2 import Environment
from storage.jira_dc_conversation import JiraDcConversation
from storage.jira_dc_integration_store import JiraDcIntegrationStore
from storage.jira_dc_user import JiraDcUser
from storage.jira_dc_workspace import JiraDcWorkspace

from openhands.agent_server.models import SendMessageRequest
from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationStartRequest,
    AppConversationStartTaskStatus,
    ConversationTrigger,
)
from openhands.app_server.config import get_app_conversation_service
from openhands.app_server.integrations.provider import ProviderHandler, ProviderType
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.user.specifiy_user_context import USER_CONTEXT_ATTR
from openhands.core.logger import openhands_logger as logger
from openhands.sdk import TextContent
from openhands.server.user_auth.user_auth import UserAuth

integration_store = JiraDcIntegrationStore.get_instance()


@dataclass
class JiraDcNewConversationView(JiraDcViewInterface):
    """View for creating a new Jira DC conversation."""

    job_context: JobContext
    saas_user_auth: UserAuth
    jira_dc_user: JiraDcUser
    jira_dc_workspace: JiraDcWorkspace
    selected_repo: str | None
    conversation_id: str

    # Decrypted API key (set by manager)
    _decrypted_api_key: str = field(default='', repr=False)

    # Resolved org ID for V1 conversations
    resolved_org_id: UUID | None = None

    async def _get_instructions(self, jinja_env: Environment) -> tuple[str, str]:
        """Instructions passed when conversation is first initialized."""
        instructions_template = jinja_env.get_template('jira_dc_instructions.j2')
        instructions = instructions_template.render()

        user_msg_template = jinja_env.get_template('jira_dc_new_conversation.j2')

        user_msg = user_msg_template.render(
            issue_key=self.job_context.issue_key,
            issue_title=self.job_context.issue_title,
            issue_description=self.job_context.issue_description,
            user_message=self.job_context.user_msg or '',
        )

        return instructions, user_msg

    async def create_or_update_conversation(self, jinja_env: Environment) -> str:
        """Create a new Jira DC conversation using V1 app conversation system.

        Returns:
            The conversation ID

        Raises:
            StartingConvoException: If conversation creation fails
        """
        if not self.selected_repo:
            raise StartingConvoException('No repository selected for this conversation')

        # Generate conversation ID
        self.conversation_id = uuid4().hex

        # Save the JiraDC conversation mapping
        jira_dc_conversation = JiraDcConversation(
            conversation_id=self.conversation_id,
            issue_id=self.job_context.issue_id,
            issue_key=self.job_context.issue_key,
            jira_dc_user_id=self.jira_dc_user.id,
        )
        await integration_store.create_conversation(jira_dc_conversation)

        # Create V1 conversation
        await self._create_v1_conversation(jinja_env)
        return self.conversation_id

    async def _create_v1_conversation(self, jinja_env: Environment):
        """Create conversation using the V1 app conversation system."""
        logger.info('[Jira DC]: Creating V1 conversation')

        instructions, user_msg = await self._get_instructions(jinja_env)

        # Create the initial message request
        initial_message = SendMessageRequest(
            role='user', content=[TextContent(text=user_msg)]
        )

        # Create the Jira DC V1 callback processor
        jira_dc_callback_processor = self._create_jira_dc_v1_callback_processor()

        # Resolve org ID for the V1 system
        self.resolved_org_id = await self._get_resolved_org_id()

        # Determine git provider
        git_provider = await self._get_git_provider()

        injector_state = InjectorState()

        # Create the V1 conversation start request
        start_request = AppConversationStartRequest(
            conversation_id=UUID(self.conversation_id),
            system_message_suffix=instructions if instructions else None,
            initial_message=initial_message,
            selected_repository=self.selected_repo,
            selected_branch=None,
            git_provider=git_provider,
            title=f'Jira DC Issue {self.job_context.issue_key}: {self.job_context.issue_title or "Unknown"}',
            trigger=ConversationTrigger.JIRA,
            processors=[jira_dc_callback_processor],
        )

        # Set up the Jira DC user context for the V1 system
        jira_dc_user_context = ResolverUserContext(
            saas_user_auth=self.saas_user_auth,
            resolver_org_id=self.resolved_org_id,
        )
        setattr(injector_state, USER_CONTEXT_ATTR, jira_dc_user_context)

        async with get_app_conversation_service(
            injector_state
        ) as app_conversation_service:
            async for task in app_conversation_service.start_app_conversation(
                start_request
            ):
                if task.status == AppConversationStartTaskStatus.ERROR:
                    logger.error(f'Failed to start V1 conversation: {task.detail}')
                    raise RuntimeError(
                        f'Failed to start V1 conversation: {task.detail}'
                    )

        logger.info(f'[Jira DC]: Created new conversation: {self.conversation_id}')

    def _create_jira_dc_v1_callback_processor(self) -> JiraDcV1CallbackProcessor:
        """Create a V1 callback processor for Jira DC integration."""
        return JiraDcV1CallbackProcessor(
            issue_key=self.job_context.issue_key,
            workspace_name=self.jira_dc_workspace.name,
            base_api_url=self.job_context.base_api_url,
            svc_acc_api_key=self._decrypted_api_key,
        )

    async def _get_git_provider(self) -> ProviderType | None:
        """Determine the git provider from the selected repository."""
        if not self.selected_repo:
            return None

        provider_tokens = await self.saas_user_auth.get_provider_tokens()
        if not provider_tokens:
            return None

        try:
            provider_handler = ProviderHandler(provider_tokens)
            repository = await provider_handler.verify_repo_provider(self.selected_repo)
            return repository.git_provider
        except Exception as e:
            logger.warning(
                f'[Jira DC] Failed to determine git provider for {self.selected_repo}: {e}'
            )
            return None

    async def _get_resolved_org_id(self) -> UUID | None:
        """Resolve the org ID for V1 conversations."""
        provider_tokens = await self.saas_user_auth.get_provider_tokens()
        if not provider_tokens or not self.selected_repo:
            return None

        try:
            provider_handler = ProviderHandler(provider_tokens)
            repository = await provider_handler.verify_repo_provider(self.selected_repo)
            resolved_org_id = await resolve_org_for_repo(
                provider=repository.git_provider.value,
                full_repo_name=self.selected_repo,
                keycloak_user_id=self.jira_dc_user.keycloak_user_id,
            )
            return resolved_org_id
        except Exception as e:
            logger.warning(
                f'[Jira DC] Failed to resolve org for {self.selected_repo}: {e}'
            )
            return None

    def get_response_msg(self) -> str:
        """Get the response message to send back to Jira DC."""
        conversation_link = CONVERSATION_URL.format(self.conversation_id)
        return f"I'm on it! {self.job_context.display_name} can [track my progress here|{conversation_link}]."


@dataclass
class JiraDcExistingConversationView(JiraDcViewInterface):
    """View for sending messages to an existing Jira DC conversation."""

    job_context: JobContext
    saas_user_auth: UserAuth
    jira_dc_user: JiraDcUser
    jira_dc_workspace: JiraDcWorkspace
    selected_repo: str | None
    conversation_id: str

    async def _get_instructions(self, jinja_env: Environment) -> tuple[str, str]:
        """Instructions passed when conversation is updated."""
        user_msg_template = jinja_env.get_template('jira_dc_existing_conversation.j2')
        user_msg = user_msg_template.render(
            issue_key=self.job_context.issue_key,
            user_message=self.job_context.user_msg or '',
            issue_title=self.job_context.issue_title,
            issue_description=self.job_context.issue_description,
        )

        return '', user_msg

    async def create_or_update_conversation(self, jinja_env: Environment) -> str:
        """Send a message to an existing V1 conversation.

        Returns:
            The conversation ID
        """
        await self._send_message_to_v1_conversation(jinja_env)
        return self.conversation_id

    async def _send_message_to_v1_conversation(self, jinja_env: Environment):
        """Send a message to an existing V1 conversation using the agent server API."""
        import httpx

        from openhands.app_server.config import (
            get_app_conversation_info_service,
            get_httpx_client,
            get_sandbox_service,
        )
        from openhands.app_server.event_callback.util import (
            ensure_conversation_found,
            get_agent_server_url_from_sandbox,
        )
        from openhands.app_server.sandbox.sandbox_models import SandboxStatus
        from openhands.app_server.services.injector import InjectorState
        from openhands.app_server.user.specifiy_user_context import (
            ADMIN,
            USER_CONTEXT_ATTR,
        )

        _, user_msg = await self._get_instructions(jinja_env)

        # Create injector state for dependency injection
        state = InjectorState()
        setattr(state, USER_CONTEXT_ATTR, ADMIN)

        async with (
            get_app_conversation_info_service(state) as app_conversation_info_service,
            get_sandbox_service(state) as sandbox_service,
            get_httpx_client(state) as httpx_client,
        ):
            # 1. Conversation lookup
            conversation_uuid = UUID(self.conversation_id)
            app_conversation_info = ensure_conversation_found(
                await app_conversation_info_service.get_app_conversation_info(
                    conversation_uuid
                ),
                conversation_uuid,
            )

            # 2. Sandbox lookup + validation
            sandbox = await sandbox_service.get_sandbox(
                app_conversation_info.sandbox_id
            )

            if sandbox is None or sandbox.status != SandboxStatus.RUNNING:
                logger.warning(
                    f'[Jira DC] Sandbox not running for conversation {self.conversation_id}'
                )
                return

            if sandbox.session_api_key is None:
                logger.warning(
                    f'[Jira DC] No session API key for sandbox: {sandbox.id}'
                )
                return

            # 3. Build URL and send message
            agent_server_url = get_agent_server_url_from_sandbox(sandbox)

            send_message_request = SendMessageRequest(
                role='user', content=[TextContent(text=user_msg)]
            )

            url = (
                f"{agent_server_url.rstrip('/')}"
                f'/api/conversations/{self.conversation_id}/messages'
            )
            headers = {'X-Session-API-Key': sandbox.session_api_key}
            payload = send_message_request.model_dump()

            try:
                response = await httpx_client.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=30.0,
                )
                response.raise_for_status()
                logger.info(
                    f'[Jira DC] Sent message to existing conversation {self.conversation_id}'
                )
            except httpx.HTTPStatusError as e:
                logger.error(
                    f'[Jira DC] Failed to send message: HTTP {e.response.status_code}'
                )
                raise
            except Exception as e:
                logger.error(f'[Jira DC] Failed to send message: {e}')
                raise

    def get_response_msg(self) -> str:
        """Get the response message to send back to Jira."""
        conversation_link = CONVERSATION_URL.format(self.conversation_id)
        return f"I'm on it! {self.job_context.display_name} can [continue tracking my progress here|{conversation_link}]."


class JiraDcFactory:
    """Factory class for creating Jira DC views based on message type."""

    @staticmethod
    async def create_jira_dc_view_from_payload(
        job_context: JobContext,
        saas_user_auth: UserAuth,
        jira_dc_user: JiraDcUser,
        jira_dc_workspace: JiraDcWorkspace,
    ) -> JiraDcViewInterface:
        """Create appropriate Jira DC view based on the payload."""
        if not jira_dc_user or not saas_user_auth or not jira_dc_workspace:
            raise StartingConvoException('User not authenticated with Jira integration')

        conversation = await integration_store.get_user_conversations_by_issue_id(
            job_context.issue_id, jira_dc_user.id
        )

        if conversation:
            return JiraDcExistingConversationView(
                job_context=job_context,
                saas_user_auth=saas_user_auth,
                jira_dc_user=jira_dc_user,
                jira_dc_workspace=jira_dc_workspace,
                selected_repo=None,
                conversation_id=conversation.conversation_id,
            )

        return JiraDcNewConversationView(
            job_context=job_context,
            saas_user_auth=saas_user_auth,
            jira_dc_user=jira_dc_user,
            jira_dc_workspace=jira_dc_workspace,
            selected_repo=None,  # Will be set later after repo inference
            conversation_id='',  # Will be set when conversation is created
        )
