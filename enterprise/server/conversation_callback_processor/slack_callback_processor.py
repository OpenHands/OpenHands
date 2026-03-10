import asyncio

from integrations.models import Message, SourceType
from integrations.slack.slack_manager import SlackManager
from integrations.slack.slack_view import SlackFactory
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
slack_manager = SlackManager(token_manager)


class SlackCallbackProcessor(ConversationCallbackProcessor):
    """Processor for sending conversation summaries to Slack.

    This processor is used to send summaries of conversations to Slack channels
    when agent state changes occur.
    """

    slack_user_id: str
    channel_id: str
    message_ts: str
    thread_ts: str | None
    team_id: str
    last_user_msg_id: int | None = None

    async def _send_message_to_slack(self, message: str) -> None:
        """Send a message to Slack.

        Args:
            message: The message content to send to Slack
        """
        try:
            # Create a message object for Slack view creation (incoming message format)
            message_obj = Message(
                source=SourceType.SLACK,
                message={
                    'slack_user_id': self.slack_user_id,
                    'channel_id': self.channel_id,
                    'message_ts': self.message_ts,
                    'thread_ts': self.thread_ts,
                    'team_id': self.team_id,
                    'user_msg': message,
                },
            )

            slack_user, saas_user_auth = await slack_manager.authenticate_user(
                self.slack_user_id
            )
            slack_view = await SlackFactory.create_slack_view_from_payload(
                message_obj, slack_user, saas_user_auth
            )
            # Send the message directly as a string
            await slack_manager.send_message(message, slack_view)

            logger.info(
                f'[Slack] Sent summary message to channel {self.channel_id} '
                f'for user {self.slack_user_id}'
            )
        except Exception as e:
            logger.error(f'[Slack] Failed to send summary message: {str(e)}')

    async def __call__(
        self,
        callback: ConversationCallback,
        observation: AgentStateChangedObservation,
    ) -> None:
        """
        Process a conversation event by sending a summary to Slack.

        Args:
            conversation_id: The ID of the conversation to process
            observation: The AgentStateChangedObservation that triggered the callback
            callback: The conversation callback
        """
        logger.info(f'[Slack] Callback agent state was {observation.agent_state}')
        if observation.agent_state not in (
            AgentState.AWAITING_USER_INPUT,
            AgentState.FINISHED,
        ):
            return

        conversation_id = callback.conversation_id
        try:
            logger.info(f'[Slack] Processing conversation {conversation_id}')

            # Get the summary instruction
            summary_instruction = get_summary_instruction()
            summary_event = event_to_dict(MessageAction(content=summary_instruction))

            # Prevent infinite loops for summary callback that always sends instructions when agent stops
            # We should not request summary if the last message is the summary request
            last_user_msg = await get_last_user_msg_from_conversation_manager(
                conversation_manager, conversation_id
            )

            # Check if we have any messages
            if len(last_user_msg) == 0:
                logger.info(
                    f'[Slack] No messages found for conversation {conversation_id}'
                )
                return

            # Get the ID of the last user message
            current_msg_id = last_user_msg[0].id if last_user_msg else None

            logger.info(
                'last_user_msg',
                extra={
                    'last_user_msg': [m.content for m in last_user_msg],
                    'summary_instruction': summary_instruction,
                    'current_msg_id': current_msg_id,
                    'last_user_msg_id': self.last_user_msg_id,
                },
            )

            # Check if the message ID has changed
            if current_msg_id == self.last_user_msg_id:
                logger.info(
                    f'[Slack] Skipping processing as message ID has not changed: {current_msg_id}'
                )
                return

            # Update the last user message ID
            self.last_user_msg_id = current_msg_id

            # Update the processor in the callback and save to database
            callback.set_processor(self)

            logger.info(f'[Slack] Updated last_user_msg_id to {self.last_user_msg_id}')

            if last_user_msg[0].content == summary_instruction:
                # Extract the summary from the event store
                logger.info(
                    f'[Slack] Extracting summary for conversation {conversation_id}'
                )
                summary = await extract_summary_from_conversation_manager(
                    conversation_manager, conversation_id
                )

                # Send the summary to Slack
                asyncio.create_task(self._send_message_to_slack(summary))

                logger.info(f'[Slack] Summary sent for conversation {conversation_id}')
                return

            # Add the summary instruction to the event stream
            logger.info(
                f'[Slack] Sending summary instruction to conversation {conversation_id} {summary_event}'
            )
            await conversation_manager.send_event_to_conversation(
                conversation_id, summary_event
            )

            logger.info(
                f'[Slack] Sent summary instruction to conversation {conversation_id} {summary_event}'
            )

        except Exception:
            logger.error(
                '[Slack] Error processing conversation callback',
                exc_info=True,
                stack_info=True,
            )
