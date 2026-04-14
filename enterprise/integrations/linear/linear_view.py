"""Linear view implementations and factory.

Views are responsible for:
- Holding the webhook payload and auth context
- Creating conversations with the selected repository
"""

from dataclasses import dataclass, field
from uuid import UUID, uuid4

from integrations.linear.linear_types import LinearViewInterface, StartingConvoException
from integrations.linear.linear_v1_callback_processor import LinearV1CallbackProcessor
from integrations.models import JobContext
from integrations.resolver_context import ResolverUserContext
from integrations.resolver_org_router import resolve_org_for_repo
from integrations.utils import CONVERSATION_URL
from jinja2 import Environment
from storage.linear_conversation import LinearConversation
from storage.linear_integration_store import LinearIntegrationStore
from storage.linear_user import LinearUser
from storage.linear_workspace import LinearWorkspace

from openhands.agent_server.models import SendMessageRequest
from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationStartRequest,
    AppConversationStartTaskStatus,
)
from openhands.app_server.config import get_app_conversation_service
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.user.specifiy_user_context import USER_CONTEXT_ATTR
from openhands.core.logger import openhands_logger as logger
from openhands.integrations.provider import ProviderHandler
from openhands.integrations.service_types import ProviderType
from openhands.sdk import TextContent
from openhands.server.user_auth.user_auth import UserAuth
from openhands.storage.data_models.conversation_metadata import (
    ConversationMetadata,
    ConversationTrigger,
)

integration_store = LinearIntegrationStore.get_instance()


@dataclass
class LinearNewConversationView(LinearViewInterface):
    """View for creating a new Linear conversation.

    This view holds the job context directly and creates V1 conversations
    using the app conversation service.
    """

    job_context: JobContext
    saas_user_auth: UserAuth
    linear_user: LinearUser
    linear_workspace: LinearWorkspace
    selected_repo: str | None
    conversation_id: str

    # Decrypted API key (set by manager)
    _decrypted_api_key: str = field(default='', repr=False)

    # Resolved org ID for V1 conversations
    resolved_org_id: UUID | None = None

    async def _get_instructions(self, jinja_env: Environment) -> tuple[str, str]:
        """Instructions passed when conversation is first initialized"""

        instructions_template = jinja_env.get_template('linear_instructions.j2')
        instructions = instructions_template.render()

        user_msg_template = jinja_env.get_template('linear_new_conversation.j2')

        user_msg = user_msg_template.render(
            issue_key=self.job_context.issue_key,
            issue_title=self.job_context.issue_title,
            issue_description=self.job_context.issue_description,
            user_message=self.job_context.user_msg or '',
        )

        return instructions, user_msg

    async def create_or_update_conversation(self, jinja_env: Environment) -> str:
        """Create a new Linear conversation using V1 system."""

        if not self.selected_repo:
            raise StartingConvoException('No repository selected for this conversation')

        # Store Linear conversation mapping first
        linear_conversation = LinearConversation(
            conversation_id=self.conversation_id,
            issue_id=self.job_context.issue_id,
            issue_key=self.job_context.issue_key,
            linear_user_id=self.linear_user.id,
        )
        await integration_store.create_conversation(linear_conversation)

        # Create V1 conversation
        conversation_metadata = await self._create_v1_metadata()
        await self._create_v1_conversation(jinja_env, conversation_metadata)
        return self.conversation_id

    async def _create_v1_metadata(self) -> ConversationMetadata:
        """Create conversation metadata for V1 conversations.

        The LinearConversation mapping is saved to the integration store (above), but
        V1 conversation metadata is managed by the app conversation system, not
        the legacy conversation store.
        """
        logger.info('[Linear]: Creating V1 metadata')

        # Generate a conversation ID for V1
        self.conversation_id = uuid4().hex
        self.resolved_org_id = await self._get_resolved_org_id()

        return ConversationMetadata(
            conversation_id=self.conversation_id,
            selected_repository=self.selected_repo,
        )

    async def _create_v1_conversation(
        self,
        jinja_env: Environment,
        conversation_metadata: ConversationMetadata,
    ):
        """Create conversation using the new V1 app conversation system."""
        logger.info('[Linear]: Creating V1 conversation')

        initial_user_text = await self._get_v1_initial_user_message(jinja_env)

        # Create the initial message request
        initial_message = SendMessageRequest(
            role='user', content=[TextContent(text=initial_user_text)]
        )

        # Create the Linear V1 callback processor
        linear_callback_processor = self._create_linear_v1_callback_processor()

        injector_state = InjectorState()

        # Resolve git provider for the repository
        git_provider = await self._resolve_git_provider()

        # Create the V1 conversation start request
        start_request = AppConversationStartRequest(
            conversation_id=UUID(conversation_metadata.conversation_id),
            system_message_suffix=None,
            initial_message=initial_message,
            selected_repository=self.selected_repo,
            selected_branch=None,
            git_provider=git_provider,
            title=f'Linear Issue {self.job_context.issue_key}: {self.job_context.issue_title or "Unknown"}',
            trigger=ConversationTrigger.LINEAR,
            processors=[linear_callback_processor],
        )

        # Set up the Linear user context for the V1 system
        linear_user_context = ResolverUserContext(
            saas_user_auth=self.saas_user_auth,
            resolver_org_id=self.resolved_org_id,
        )
        setattr(injector_state, USER_CONTEXT_ATTR, linear_user_context)

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

    async def _get_v1_initial_user_message(self, jinja_env: Environment) -> str:
        """Build the initial user message for V1 conversations."""
        user_msg_template = jinja_env.get_template('linear_new_conversation.j2')
        user_msg = user_msg_template.render(
            issue_key=self.job_context.issue_key,
            issue_title=self.job_context.issue_title,
            issue_description=self.job_context.issue_description,
            user_message=self.job_context.user_msg or '',
        )

        return user_msg

    def _create_linear_v1_callback_processor(self):
        """Create a V1 callback processor for Linear integration."""
        return LinearV1CallbackProcessor(
            decrypted_api_key=self._decrypted_api_key,
            issue_id=self.job_context.issue_id,
            issue_key=self.job_context.issue_key,
            workspace_name=self.linear_workspace.name,
        )

    async def _resolve_git_provider(self) -> ProviderType | None:
        """Resolve the git provider for the repository."""
        provider_tokens = await self.saas_user_auth.get_provider_tokens()
        if not provider_tokens or not self.selected_repo:
            return None

        try:
            provider_handler = ProviderHandler(provider_tokens)
            repository = await provider_handler.verify_repo_provider(self.selected_repo)
            return repository.git_provider
        except Exception as e:
            logger.warning(
                f'[Linear] Failed to resolve git provider for {self.selected_repo}: {e}'
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
                keycloak_user_id=self.linear_user.keycloak_user_id,
            )
            return resolved_org_id
        except Exception as e:
            logger.warning(
                f'[Linear] Failed to resolve org for {self.selected_repo}: {e}'
            )
            return None

    def get_response_msg(self) -> str:
        """Get the response message to send back to Linear"""
        conversation_link = CONVERSATION_URL.format(self.conversation_id)
        return f"I'm on it! {self.job_context.display_name} can [track my progress here]({conversation_link})."


@dataclass
class LinearExistingConversationView(LinearViewInterface):
    """View for updating an existing Linear conversation.

    This view handles follow-up messages to existing conversations
    using the V1 app conversation system.
    """

    job_context: JobContext
    saas_user_auth: UserAuth
    linear_user: LinearUser
    linear_workspace: LinearWorkspace
    selected_repo: str | None
    conversation_id: str

    async def _get_instructions(self, jinja_env: Environment) -> tuple[str, str]:
        """Instructions passed when conversation is first initialized"""

        user_msg_template = jinja_env.get_template('linear_existing_conversation.j2')
        user_msg = user_msg_template.render(
            issue_key=self.job_context.issue_key,
            user_message=self.job_context.user_msg or '',
            issue_title=self.job_context.issue_title,
            issue_description=self.job_context.issue_description,
        )

        return '', user_msg

    async def create_or_update_conversation(self, jinja_env: Environment) -> str:
        """Update an existing Linear conversation using V1 system."""
        logger.info(f'[Linear] Updating existing conversation {self.conversation_id}')

        try:
            sandbox = await self._get_running_sandbox()
            _, user_msg = await self._get_instructions(jinja_env)
            await self._send_followup_message(sandbox, user_msg)
            return self.conversation_id

        except StartingConvoException:
            raise
        except Exception as e:
            logger.error(
                f'[Linear] Failed to update conversation: {str(e)}', exc_info=True
            )
            raise StartingConvoException(f'Failed to update conversation: {str(e)}')

    async def _get_running_sandbox(self):
        """Get the running sandbox for the conversation."""
        from openhands.app_server.config import (
            get_app_conversation_info_service,
            get_sandbox_service,
        )
        from openhands.app_server.user.specifiy_user_context import ADMIN

        # Set up admin context for V1 API calls
        injector_state = InjectorState()
        setattr(injector_state, USER_CONTEXT_ATTR, ADMIN)

        async with get_app_conversation_info_service(
            injector_state
        ) as app_conversation_info_service:
            # Check if conversation exists
            conversation_info = (
                await app_conversation_info_service.get_app_conversation_info(
                    UUID(self.conversation_id)
                )
            )
            if not conversation_info:
                raise StartingConvoException('Conversation no longer exists.')

            async with get_sandbox_service(injector_state) as sandbox_service:
                # Check sandbox is running
                sandbox = await sandbox_service.get_sandbox(
                    conversation_info.sandbox_id
                )
                if not sandbox or sandbox.status != 'running':
                    raise StartingConvoException(
                        'Conversation sandbox is not available.'
                    )
                return sandbox  # type: ignore[unreachable]

    async def _send_followup_message(self, sandbox, message: str):
        """Send a follow-up message to an existing conversation via V1 API."""
        import httpx

        from openhands.app_server.event_callback.util import (
            get_agent_server_url_from_sandbox,
        )
        from openhands.utils.http_session import httpx_verify_option

        if not sandbox.session_api_key:
            raise StartingConvoException('No session API key for sandbox')

        agent_server_url = get_agent_server_url_from_sandbox(sandbox)
        url = (
            f"{agent_server_url.rstrip('/')}"
            f"/api/conversations/{self.conversation_id}/send-message"
        )
        headers = {'X-Session-API-Key': sandbox.session_api_key}

        # Create the message request
        message_request = SendMessageRequest(
            role='user', content=[TextContent(text=message)]
        )

        async with httpx.AsyncClient(verify=httpx_verify_option()) as client:
            response = await client.post(
                url,
                json=message_request.model_dump(),
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()

        logger.info(
            f'[Linear] Sent follow-up message to conversation {self.conversation_id}'
        )

    def get_response_msg(self) -> str:
        """Get the response message to send back to Linear"""
        conversation_link = CONVERSATION_URL.format(self.conversation_id)
        return f"I'm on it! {self.job_context.display_name} can [continue tracking my progress here]({conversation_link})."


class LinearFactory:
    """Factory for creating Linear views based on message content.

    The factory is responsible for:
    - Creating the appropriate view type (new or existing conversation)
    - Looking up existing conversations for the issue
    """

    @staticmethod
    async def create_linear_view_from_payload(
        job_context: JobContext,
        saas_user_auth: UserAuth,
        linear_user: LinearUser,
        linear_workspace: LinearWorkspace,
        decrypted_api_key: str = '',
    ) -> LinearViewInterface:
        """Create appropriate Linear view based on the message and user state.

        Args:
            job_context: The job context with issue details
            saas_user_auth: OpenHands user authentication
            linear_user: The Linear user
            linear_workspace: The Linear workspace
            decrypted_api_key: Decrypted service account API key

        Returns:
            A LinearViewInterface for the appropriate conversation type
        """
        if not linear_user or not saas_user_auth or not linear_workspace:
            raise StartingConvoException(
                'User not authenticated with Linear integration'
            )

        conversation = await integration_store.get_user_conversations_by_issue_id(
            job_context.issue_id, linear_user.id
        )
        if conversation:
            logger.info(
                f'[Linear] Found existing conversation for issue {job_context.issue_id}'
            )
            return LinearExistingConversationView(
                job_context=job_context,
                saas_user_auth=saas_user_auth,
                linear_user=linear_user,
                linear_workspace=linear_workspace,
                selected_repo=None,
                conversation_id=conversation.conversation_id,
            )

        return LinearNewConversationView(
            job_context=job_context,
            saas_user_auth=saas_user_auth,
            linear_user=linear_user,
            linear_workspace=linear_workspace,
            selected_repo=None,  # Will be set later after repo inference
            conversation_id='',  # Will be set when conversation is created
            _decrypted_api_key=decrypted_api_key,
        )
