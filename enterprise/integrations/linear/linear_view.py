from dataclasses import dataclass
from uuid import uuid4

from integrations.linear.linear_types import LinearViewInterface, StartingConvoException
from integrations.models import JobContext
from integrations.resolver_org_router import resolve_org_for_repo
from integrations.utils import CONVERSATION_URL, get_final_agent_observation
from jinja2 import Environment
from server.config import get_config
from storage.linear_conversation import LinearConversation
from storage.linear_integration_store import LinearIntegrationStore
from storage.linear_user import LinearUser
from storage.linear_workspace import LinearWorkspace
from storage.saas_conversation_store import SaasConversationStore

from openhands.core.logger import openhands_logger as logger
from openhands.core.schema.agent import AgentState
from openhands.events.action import MessageAction
from openhands.events.serialization.event import event_to_dict
from openhands.integrations.provider import ProviderHandler
from openhands.server.services.conversation_service import (
    setup_init_conversation_settings,
    start_conversation,
)
from openhands.server.shared import ConversationStoreImpl, config, conversation_manager
from openhands.server.user_auth.user_auth import UserAuth
from openhands.storage.data_models.conversation_metadata import (
    ConversationMetadata,
    ConversationTrigger,
)
from openhands.utils.conversation_summary import get_default_conversation_title

integration_store = LinearIntegrationStore.get_instance()


@dataclass
class LinearNewConversationView(LinearViewInterface):
    job_context: JobContext
    saas_user_auth: UserAuth
    linear_user: LinearUser
    linear_workspace: LinearWorkspace
    selected_repo: str | None
    conversation_id: str

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
        """Create a new Linear conversation"""

        if not self.selected_repo:
            raise StartingConvoException('No repository selected for this conversation')

        provider_tokens = await self.saas_user_auth.get_provider_tokens()
        user_secrets = await self.saas_user_auth.get_secrets()
        instructions, user_msg = await self._get_instructions(jinja_env)

        try:
            user_id = self.linear_user.keycloak_user_id

            # Resolve git provider from repository
            resolved_git_provider = None
            if provider_tokens:
                try:
                    provider_handler = ProviderHandler(provider_tokens)
                    repository = await provider_handler.verify_repo_provider(
                        self.selected_repo
                    )
                    resolved_git_provider = repository.git_provider
                except Exception as e:
                    logger.warning(
                        f'[Linear] Failed to resolve git provider for {self.selected_repo}: {e}'
                    )

            # Resolve target org based on claimed git organizations
            resolved_org_id = None
            if resolved_git_provider and self.selected_repo:
                try:
                    resolved_org_id = await resolve_org_for_repo(
                        provider=resolved_git_provider.value,
                        full_repo_name=self.selected_repo,
                        keycloak_user_id=user_id,
                    )
                except Exception as e:
                    logger.warning(
                        f'[Linear] Failed to resolve org for {self.selected_repo}: {e}'
                    )

            # Create the conversation store with resolver org routing
            # (bypasses initialize_conversation to avoid threading enterprise-only
            # resolver_org_id through the generic OSS interface)
            store = await SaasConversationStore.get_resolver_instance(
                get_config(),
                user_id,
                resolved_org_id,
            )

            conversation_id = uuid4().hex
            conversation_metadata = ConversationMetadata(
                trigger=ConversationTrigger.LINEAR,
                conversation_id=conversation_id,
                title=get_default_conversation_title(conversation_id),
                user_id=user_id,
                selected_repository=self.selected_repo,
                selected_branch=None,
                git_provider=resolved_git_provider,
            )
            await store.save_metadata(conversation_metadata)

            await start_conversation(
                user_id=user_id,
                git_provider_tokens=provider_tokens,
                custom_secrets=user_secrets.custom_secrets if user_secrets else None,
                initial_user_msg=user_msg,
                image_urls=None,
                replay_json=None,
                conversation_id=conversation_id,
                conversation_metadata=conversation_metadata,
                conversation_instructions=instructions,
            )

            self.conversation_id = conversation_id

            logger.info(f'[Linear] Created conversation {self.conversation_id}')

            # Store Linear conversation mapping
            linear_conversation = LinearConversation(
                conversation_id=self.conversation_id,
                issue_id=self.job_context.issue_id,
                issue_key=self.job_context.issue_key,
                linear_user_id=self.linear_user.id,
            )

            await integration_store.create_conversation(linear_conversation)

            return self.conversation_id
        except Exception as e:
            logger.error(
                f'[Linear] Failed to create conversation: {str(e)}', exc_info=True
            )
            raise StartingConvoException(f'Failed to create conversation: {str(e)}')

    def get_response_msg(self) -> str:
        """Get the response message to send back to Linear"""
        conversation_link = CONVERSATION_URL.format(self.conversation_id)
        return f"I'm on it! {self.job_context.display_name} can [track my progress here]({conversation_link})."


@dataclass
class LinearExistingConversationView(LinearViewInterface):
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
        """Update an existing Linear conversation"""

        user_id = self.linear_user.keycloak_user_id

        try:
            conversation_store = await ConversationStoreImpl.get_instance(
                config, user_id
            )

            try:
                await conversation_store.get_metadata(self.conversation_id)
            except FileNotFoundError:
                raise StartingConvoException('Conversation no longer exists.')

            provider_tokens = await self.saas_user_auth.get_provider_tokens()
            if provider_tokens is None:
                raise ValueError('Could not load provider tokens')
            providers_set = list(provider_tokens.keys())

            conversation_init_data = await setup_init_conversation_settings(
                user_id, self.conversation_id, providers_set
            )

            # Either join ongoing conversation, or restart the conversation
            agent_loop_info = await conversation_manager.maybe_start_agent_loop(
                self.conversation_id, conversation_init_data, user_id
            )

            if agent_loop_info.event_store is None:
                raise StartingConvoException('Event store not available')

            final_agent_observation = get_final_agent_observation(
                agent_loop_info.event_store
            )
            agent_state = (
                None
                if len(final_agent_observation) == 0
                else final_agent_observation[0].agent_state
            )

            if not agent_state or agent_state == AgentState.LOADING:
                raise StartingConvoException('Conversation is still starting')

            _, user_msg = await self._get_instructions(jinja_env)
            user_message_event = MessageAction(content=user_msg)
            await conversation_manager.send_event_to_conversation(
                self.conversation_id, event_to_dict(user_message_event)
            )

            return self.conversation_id
        except Exception as e:
            logger.error(
                f'[Linear] Failed to create conversation: {str(e)}', exc_info=True
            )
            raise StartingConvoException(f'Failed to create conversation: {str(e)}')

    def get_response_msg(self) -> str:
        """Get the response message to send back to Linear"""
        conversation_link = CONVERSATION_URL.format(self.conversation_id)
        return f"I'm on it! {self.job_context.display_name} can [continue tracking my progress here]({conversation_link})."


class LinearFactory:
    """Factory for creating Linear views based on message content"""

    @staticmethod
    async def create_linear_view_from_payload(
        job_context: JobContext,
        saas_user_auth: UserAuth,
        linear_user: LinearUser,
        linear_workspace: LinearWorkspace,
    ) -> LinearViewInterface:
        """Create appropriate Linear view based on the message and user state"""

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
        )
