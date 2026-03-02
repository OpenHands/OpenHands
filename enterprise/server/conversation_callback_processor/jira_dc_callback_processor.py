import asyncio

from integrations.jira_dc.jira_dc_manager import JiraDcManager
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
jira_dc_manager = JiraDcManager(token_manager)


class JiraDcCallbackProcessor(ConversationCallbackProcessor):
    """
    Processor for sending conversation summaries to Jira DC.

    This processor is used to send summaries of conversations to Jira DC issues
    when agent state changes occur.
    """

    issue_key: str
    workspace_name: str
    base_api_url: str

    async def _send_comment_to_jira_dc(self, message: str) -> None:
        """Send a comment to Jira DC issue.

        Args:
            message: The message content to send to Jira DC
        """
        try:
            # Get workspace details to retrieve API credentials
            workspace = await jira_dc_manager.integration_store.get_workspace_by_name(
                self.workspace_name
            )
            if not workspace:
                logger.error(f'[Jira DC] Workspace {self.workspace_name} not found')
                return

            if workspace.status != 'active':
                logger.error(f'[Jira DC] Workspace {workspace.id} is not active')
                return

            # Decrypt API key
            api_key = jira_dc_manager.token_manager.decrypt_text(
                workspace.svc_acc_api_key
            )

            # Send comment directly as a string
            await jira_dc_manager.send_message(
                message,
                issue_key=self.issue_key,
                base_api_url=self.base_api_url,
                svc_acc_api_key=api_key,
            )

            logger.info(
                f'[Jira DC] Sent summary comment to issue {self.issue_key} '
                f'(workspace {self.workspace_name})'
            )
        except Exception as e:
            logger.error(f'[Jira DC] Failed to send summary comment: {str(e)}')

    async def __call__(
        self,
        callback: ConversationCallback,
        observation: AgentStateChangedObservation,
    ) -> None:
        """
        Process a conversation event by sending a summary to Jira DC.

        Args:
            callback: The conversation callback
            observation: The AgentStateChangedObservation that triggered the callback
        """
        logger.info(f'[Jira DC] Callback agent state was {observation.agent_state}')
        if observation.agent_state not in (
            AgentState.AWAITING_USER_INPUT,
            AgentState.FINISHED,
        ):
            return

        conversation_id = callback.conversation_id
        try:
            logger.info(
                f'[Jira DC] Sending summary instruction for conversation {conversation_id}'
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
                    f'[Jira DC] Extracting summary for conversation {conversation_id}'
                )

                summary_markdown = await extract_summary_from_conversation_manager(
                    conversation_manager, conversation_id
                )

                summary = markdown_to_jira_markup(summary_markdown)

                asyncio.create_task(self._send_comment_to_jira_dc(summary))

                logger.info(
                    f'[Jira DC] Summary sent for conversation {conversation_id}'
                )
                return

            # Add the summary instruction to the event stream
            logger.info(
                f'[Jira DC] Sending summary instruction to conversation {conversation_id} {summary_event}'
            )
            await conversation_manager.send_event_to_conversation(
                conversation_id, summary_event
            )

            logger.info(
                f'[Jira DC] Sent summary instruction to conversation {conversation_id} {summary_event}'
            )

        except Exception:
            logger.error(
                '[Jira DC] Error processing conversation callback',
                exc_info=True,
                stack_info=True,
            )
