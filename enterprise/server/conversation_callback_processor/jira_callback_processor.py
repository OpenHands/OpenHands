import asyncio

from integrations.jira.jira_manager import JiraManager
from integrations.utils import (
    extract_summary_from_conversation_manager,
    get_last_user_msg_from_conversation_manager,
    get_summary_instruction,
    markdown_to_jira_markup,
)
from server.auth.token_manager import TokenManager
from storage.conversation_callback import (
    ConversationCallback,
    ConversationCallbackProcessor,
)

from openhands.core.logger import openhands_logger as logger
from openhands.core.schema.agent import AgentState
from openhands.events.action import MessageAction
from openhands.events.observation.agent import AgentStateChangedObservation
from openhands.events.serialization.event import event_to_dict
from openhands.server.shared import conversation_manager

token_manager = TokenManager()
jira_manager = JiraManager(token_manager)
integration_store = jira_manager.integration_store


class JiraCallbackProcessor(ConversationCallbackProcessor):
    """
    Processor for sending conversation summaries to Jira.

    This processor is used to send summaries of conversations to Jira issues
    when agent state changes occur.
    """

    issue_key: str
    workspace_name: str

    async def _send_comment_to_jira(self, message: str) -> None:
        """Send a comment to Jira issue.

        Args:
            message: The message content to send to Jira
        """
        try:
            # Get workspace details to retrieve API credentials
            workspace = await jira_manager.integration_store.get_workspace_by_name(
                self.workspace_name
            )
            if not workspace:
                logger.error(f'[Jira] Workspace {self.workspace_name} not found')
                return

            if workspace.status != 'active':
                logger.error(f'[Jira] Workspace {workspace.id} is not active')
                return

            # Decrypt API key
            api_key = jira_manager.token_manager.decrypt_text(workspace.svc_acc_api_key)

            # Send comment directly as a string
            await jira_manager.send_message(
                message,
                issue_key=self.issue_key,
                jira_cloud_id=workspace.jira_cloud_id,
                svc_acc_email=workspace.svc_acc_email,
                svc_acc_api_key=api_key,
            )

            logger.info(
                f'[Jira] Sent summary comment to issue {self.issue_key} '
                f'(workspace {self.workspace_name})'
            )
        except Exception as e:
            logger.error(f'[Jira] Failed to send summary comment: {str(e)}')

    async def __call__(
        self,
        callback: ConversationCallback,
        observation: AgentStateChangedObservation,
    ) -> None:
        """
        Process a conversation event by sending a summary to Jira.

        Args:
            callback: The conversation callback
            observation: The AgentStateChangedObservation that triggered the callback
        """
        logger.info(f'[Jira] Callback agent state was {observation.agent_state}')
        if observation.agent_state not in (
            AgentState.AWAITING_USER_INPUT,
            AgentState.FINISHED,
        ):
            return

        conversation_id = callback.conversation_id
        try:
            logger.info(
                f'[Jira] Sending summary instruction for conversation {conversation_id}'
            )

            # Get the summary instruction
            summary_instruction = get_summary_instruction()
            summary_event = event_to_dict(MessageAction(content=summary_instruction))

            # Prevent infinite loops for summary callback that always sends instructions when agent stops
            # We should not request summary if the last message is the summary request
            last_user_msg = await get_last_user_msg_from_conversation_manager(
                conversation_manager, conversation_id
            )
            logger.info(
                'last_user_msg',
                extra={
                    'last_user_msg': [m.content for m in last_user_msg],
                    'summary_instruction': summary_instruction,
                },
            )
            if (
                len(last_user_msg) > 0
                and last_user_msg[0].content == summary_instruction
            ):
                # Extract the summary from the event store
                logger.info(
                    f'[Jira] Extracting summary for conversation {conversation_id}'
                )
                summary_markdown = await extract_summary_from_conversation_manager(
                    conversation_manager, conversation_id
                )

                summary = markdown_to_jira_markup(summary_markdown)

                asyncio.create_task(self._send_comment_to_jira(summary))

                logger.info(f'[Jira] Summary sent for conversation {conversation_id}')
                return

            # Add the summary instruction to the event stream
            logger.info(
                f'[Jira] Sending summary instruction to conversation {conversation_id} {summary_event}'
            )
            await conversation_manager.send_event_to_conversation(
                conversation_id, summary_event
            )

            logger.info(
                f'[Jira] Sent summary instruction to conversation {conversation_id} {summary_event}'
            )

        except Exception:
            logger.error(
                '[Jira] Error processing conversation callback',
                exc_info=True,
                stack_info=True,
            )
