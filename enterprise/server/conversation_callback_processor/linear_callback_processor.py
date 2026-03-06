import asyncio

from integrations.linear.linear_manager import LinearManager
from integrations.utils import (
    extract_summary_from_conversation_manager,
    get_last_user_msg_from_conversation_manager,
    get_summary_instruction,
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
linear_manager = LinearManager(token_manager)


class LinearCallbackProcessor(ConversationCallbackProcessor):
    """
    Processor for sending conversation summaries to Linear.

    This processor is used to send summaries of conversations to Linear issues
    when agent state changes occur.
    """

    issue_id: str
    issue_key: str
    workspace_name: str

    async def _send_comment_to_linear(self, message: str) -> None:
        """Send a comment to Linear issue.

        Args:
            message: The message content to send to Linear
        """
        try:
            # Get workspace details to retrieve API key
            workspace = await linear_manager.integration_store.get_workspace_by_name(
                self.workspace_name
            )
            if not workspace:
                logger.error(f'[Linear] Workspace {self.workspace_name} not found')
                return

            if workspace.status != 'active':
                logger.error(f'[Linear] Workspace {workspace.id} is not active')
                return

            # Decrypt API key
            api_key = linear_manager.token_manager.decrypt_text(
                workspace.svc_acc_api_key
            )

            # Send comment directly as a string
            await linear_manager.send_message(
                message,
                self.issue_id,
                api_key,
            )

            logger.info(
                f'[Linear] Sent summary comment to issue {self.issue_key} '
                f'(workspace {self.workspace_name})'
            )
        except Exception as e:
            logger.error(f'[Linear] Failed to send summary comment: {str(e)}')

    async def __call__(
        self,
        callback: ConversationCallback,
        observation: AgentStateChangedObservation,
    ) -> None:
        """
        Process a conversation event by sending a summary to Linear.

        Args:
            callback: The conversation callback
            observation: The AgentStateChangedObservation that triggered the callback
        """
        logger.info(f'[Linear] Callback agent state was {observation.agent_state}')
        if observation.agent_state not in (
            AgentState.AWAITING_USER_INPUT,
            AgentState.FINISHED,
        ):
            return

        conversation_id = callback.conversation_id
        try:
            logger.info(
                f'[Linear] Sending summary instruction for conversation {conversation_id}'
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
                    f'[Linear] Extracting summary for conversation {conversation_id}'
                )
                summary = await extract_summary_from_conversation_manager(
                    conversation_manager, conversation_id
                )

                # Send the summary to Linear
                asyncio.create_task(self._send_comment_to_linear(summary))

                logger.info(f'[Linear] Summary sent for conversation {conversation_id}')
                return

            # Add the summary instruction to the event stream
            logger.info(
                f'[Linear] Sending summary instruction to conversation {conversation_id} {summary_event}'
            )
            await conversation_manager.send_event_to_conversation(
                conversation_id, summary_event
            )

            logger.info(
                f'[Linear] Sent summary instruction to conversation {conversation_id} {summary_event}'
            )

        except Exception:
            logger.error(
                '[Linear] Error processing conversation callback',
                exc_info=True,
                stack_info=True,
            )
