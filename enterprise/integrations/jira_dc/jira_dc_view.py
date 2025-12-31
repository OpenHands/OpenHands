from dataclasses import dataclass

from integrations.jira_dc.jira_dc_types import (
    JiraDcViewInterface,
    StartingConvoException,
)
from integrations.models import JobContext
from integrations.utils import CONVERSATION_URL, get_final_agent_observation
from jinja2 import Environment
from storage.jira_dc_conversation import JiraDcConversation
from storage.jira_dc_integration_store import JiraDcIntegrationStore
from storage.jira_dc_user import JiraDcUser
from storage.jira_dc_workspace import JiraDcWorkspace

from openhands.core.logger import openhands_logger as logger
from openhands.core.schema.agent import AgentState
from openhands.events.action import MessageAction
from openhands.events.serialization.event import event_to_dict
from openhands.server.services.conversation_service import (
    create_new_conversation,
    setup_init_conversation_settings,
)
from openhands.server.shared import ConversationStoreImpl, config, conversation_manager
from openhands.server.user_auth.user_auth import UserAuth
from openhands.storage.data_models.conversation_metadata import ConversationTrigger

integration_store = JiraDcIntegrationStore.get_instance()


@dataclass
class JiraDcNewConversationView(JiraDcViewInterface):
    job_context: JobContext
    saas_user_auth: UserAuth
    jira_dc_user: JiraDcUser
    jira_dc_workspace: JiraDcWorkspace
    selected_repo: str | None
    conversation_id: str

    def _get_instructions(self, jinja_env: Environment) -> tuple[str, str]:
        """Instructions passed when conversation is first initialized"""

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
        """Create a new Jira DC conversation"""

        if not self.selected_repo:
            raise StartingConvoException('No repository selected for this conversation')

        provider_tokens = await self.saas_user_auth.get_provider_tokens()
        user_secrets = await self.saas_user_auth.get_secrets()
        instructions, user_msg = self._get_instructions(jinja_env)

        try:
            agent_loop_info = await create_new_conversation(
                user_id=self.jira_dc_user.keycloak_user_id,
                git_provider_tokens=provider_tokens,
                selected_repository=self.selected_repo,
                selected_branch=None,
                initial_user_msg=user_msg,
                conversation_instructions=instructions,
                image_urls=None,
                replay_json=None,
                conversation_trigger=ConversationTrigger.JIRA_DC,
                custom_secrets=user_secrets.custom_secrets if user_secrets else None,
            )

            self.conversation_id = agent_loop_info.conversation_id

            logger.info(f'[Jira DC] Created conversation {self.conversation_id}')

            # Store Jira DC conversation mapping
            jira_dc_conversation = JiraDcConversation(
                conversation_id=self.conversation_id,
                issue_id=self.job_context.issue_id,
                issue_key=self.job_context.issue_key,
                jira_dc_user_id=self.jira_dc_user.id,
            )

            await integration_store.create_conversation(jira_dc_conversation)

            return self.conversation_id
        except Exception as e:
            logger.error(
                f'[Jira DC] Failed to create conversation: {str(e)}', exc_info=True
            )
            raise StartingConvoException(f'Failed to create conversation: {str(e)}')

    def get_response_msg(self) -> str:
        """Get the response message to send back to Jira DC"""
        conversation_link = CONVERSATION_URL.format(self.conversation_id)
        return f"I'm on it! {self.job_context.display_name} can [track my progress here|{conversation_link}]."


@dataclass
class JiraDcExistingConversationView(JiraDcViewInterface):
    job_context: JobContext
    saas_user_auth: UserAuth
    jira_dc_user: JiraDcUser
    jira_dc_workspace: JiraDcWorkspace
    selected_repo: str | None
    conversation_id: str

    def _get_instructions(self, jinja_env: Environment) -> tuple[str, str]:
        """Instructions passed when conversation is first initialized"""

        user_msg_template = jinja_env.get_template('jira_dc_existing_conversation.j2')
        user_msg = user_msg_template.render(
            issue_key=self.job_context.issue_key,
            user_message=self.job_context.user_msg or '',
            issue_title=self.job_context.issue_title,
            issue_description=self.job_context.issue_description,
        )

        return '', user_msg

    async def create_or_update_conversation(self, jinja_env: Environment) -> str:
        """Update an existing Jira conversation"""

        user_id = self.jira_dc_user.keycloak_user_id

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

            _, user_msg = self._get_instructions(jinja_env)
            user_message_event = MessageAction(content=user_msg)
            await conversation_manager.send_event_to_conversation(
                self.conversation_id, event_to_dict(user_message_event)
            )

            return self.conversation_id
        except Exception as e:
            logger.error(
                f'[Jira] Failed to create conversation: {str(e)}', exc_info=True
            )
            raise StartingConvoException(f'Failed to create conversation: {str(e)}')

    def get_response_msg(self) -> str:
        """Get the response message to send back to Jira"""
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
