from dataclasses import dataclass

from integrations.jira.jira_types import JiraViewInterface, StartingConvoException
from integrations.models import JobContext, Message
from integrations.utils import CONVERSATION_URL, HOST, get_oh_labels, has_exact_mention
from jinja2 import Environment
from storage.jira_conversation import JiraConversation
from storage.jira_integration_store import JiraIntegrationStore
from storage.jira_user import JiraUser
from storage.jira_workspace import JiraWorkspace

from openhands.core.logger import openhands_logger as logger
from openhands.server.services.conversation_service import (
    create_new_conversation,
)
from openhands.server.user_auth.user_auth import UserAuth
from openhands.storage.data_models.conversation_metadata import ConversationTrigger

integration_store = JiraIntegrationStore.get_instance()

OH_LABEL, INLINE_OH_LABEL = get_oh_labels(HOST)


@dataclass
class JiraNewConversationView(JiraViewInterface):
    job_context: JobContext
    saas_user_auth: UserAuth
    jira_user: JiraUser
    jira_workspace: JiraWorkspace
    selected_repo: str | None
    conversation_id: str

    def _get_instructions(self, jinja_env: Environment) -> tuple[str, str]:
        """Instructions passed when conversation is first initialized"""

        instructions_template = jinja_env.get_template('jira_instructions.j2')
        instructions = instructions_template.render()

        user_msg_template = jinja_env.get_template('jira_new_conversation.j2')

        user_msg = user_msg_template.render(
            issue_key=self.job_context.issue_key,
            issue_title=self.job_context.issue_title,
            issue_description=self.job_context.issue_description,
            user_message=self.job_context.user_msg or '',
        )

        return instructions, user_msg

    async def create_or_update_conversation(self, jinja_env: Environment) -> str:
        """Create a new Jira conversation"""

        if not self.selected_repo:
            raise StartingConvoException('No repository selected for this conversation')

        provider_tokens = await self.saas_user_auth.get_provider_tokens()
        user_secrets = await self.saas_user_auth.get_secrets()
        instructions, user_msg = self._get_instructions(jinja_env)

        try:
            agent_loop_info = await create_new_conversation(
                user_id=self.jira_user.keycloak_user_id,
                git_provider_tokens=provider_tokens,
                selected_repository=self.selected_repo,
                selected_branch=None,
                initial_user_msg=user_msg,
                conversation_instructions=instructions,
                image_urls=None,
                replay_json=None,
                conversation_trigger=ConversationTrigger.JIRA,
                custom_secrets=user_secrets.custom_secrets if user_secrets else None,
            )

            self.conversation_id = agent_loop_info.conversation_id

            logger.info(f'[Jira] Created conversation {self.conversation_id}')

            # Store Jira conversation mapping
            jira_conversation = JiraConversation(
                conversation_id=self.conversation_id,
                issue_id=self.job_context.issue_id,
                issue_key=self.job_context.issue_key,
                jira_user_id=self.jira_user.id,
            )

            await integration_store.create_conversation(jira_conversation)

            return self.conversation_id
        except Exception as e:
            logger.error(
                f'[Jira] Failed to create conversation: {str(e)}', exc_info=True
            )
            raise StartingConvoException(f'Failed to create conversation: {str(e)}')

    def get_response_msg(self) -> str:
        """Get the response message to send back to Jira"""
        conversation_link = CONVERSATION_URL.format(self.conversation_id)
        return f"I'm on it! {self.job_context.display_name} can [track my progress here|{conversation_link}]."


class JiraFactory:
    """Factory for creating Jira views based on message content"""

    @staticmethod
    def is_labeled_ticket(message: Message) -> bool:
        payload = message.message.get('payload', {})
        event_type = payload.get('webhookEvent')

        if event_type != 'jira:issue_updated':
            return False

        changelog = payload.get('changelog', {})
        items = changelog.get('items', [])
        labels = [
            item.get('toString', '')
            for item in items
            if item.get('field') == 'labels' and 'toString' in item
        ]

        return OH_LABEL in labels

    @staticmethod
    def is_ticket_comment(message: Message) -> bool:
        payload = message.message.get('payload', {})
        event_type = payload.get('webhookEvent')

        if event_type != 'comment_created':
            return False

        comment_data = payload.get('comment', {})
        comment_body = comment_data.get('body', '')
        return has_exact_mention(comment_body, INLINE_OH_LABEL)

    @staticmethod
    async def create_jira_view_from_payload(
        job_context: JobContext,
        saas_user_auth: UserAuth,
        jira_user: JiraUser,
        jira_workspace: JiraWorkspace,
    ) -> JiraViewInterface:
        """Create appropriate Jira view based on the message and user state"""

        if not jira_user or not saas_user_auth or not jira_workspace:
            raise StartingConvoException('User not authenticated with Jira integration')

        return JiraNewConversationView(
            job_context=job_context,
            saas_user_auth=saas_user_auth,
            jira_user=jira_user,
            jira_workspace=jira_workspace,
            selected_repo=None,  # Will be set later after repo inference
            conversation_id='',  # Will be set when conversation is created
        )
