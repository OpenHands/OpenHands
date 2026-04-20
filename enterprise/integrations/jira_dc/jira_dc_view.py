from dataclasses import dataclass

from integrations.jira_dc.jira_dc_types import (
    JiraDcViewInterface,
    StartingConvoException,
)
from integrations.models import JobContext
from integrations.utils import CONVERSATION_URL
from jinja2 import Environment
from storage.jira_dc_integration_store import JiraDcIntegrationStore
from storage.jira_dc_user import JiraDcUser
from storage.jira_dc_workspace import JiraDcWorkspace

from openhands.server.user_auth.user_auth import UserAuth

integration_store = JiraDcIntegrationStore.get_instance()


@dataclass
class JiraDcNewConversationView(JiraDcViewInterface):
    job_context: JobContext
    saas_user_auth: UserAuth
    jira_dc_user: JiraDcUser
    jira_dc_workspace: JiraDcWorkspace
    selected_repo: str | None
    conversation_id: str

    async def _get_instructions(self, jinja_env: Environment) -> tuple[str, str]:
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
        """Create a new Jira DC conversation.

        Note: This functionality has been deprecated as part of the V0 to V1 migration.
        The conversation_manager has been removed and this method needs to be reimplemented
        using the V1 API.
        """
        raise NotImplementedError(
            'JiraDcNewConversationView.create_or_update_conversation is not yet '
            'implemented in V1. The V0 conversation_manager has been removed.'
        )

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

    async def _get_instructions(self, jinja_env: Environment) -> tuple[str, str]:
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
        """Update an existing Jira conversation.

        Note: This functionality has been deprecated as part of the V0 to V1 migration.
        The conversation_manager has been removed and this method needs to be reimplemented
        using the V1 API.
        """
        raise NotImplementedError(
            'JiraDcExistingConversationView.create_or_update_conversation is not yet '
            'implemented in V1. The V0 conversation_manager has been removed.'
        )

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
