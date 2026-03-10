import asyncio
from datetime import datetime

from integrations.gitlab.gitlab_manager import GitlabManager
from integrations.gitlab.gitlab_view import GitlabViewType
from integrations.utils import (
    extract_summary_from_conversation_manager,
    get_summary_instruction,
)
from server.auth.token_manager import TokenManager
from storage.conversation_callback import (
    CallbackStatus,
    ConversationCallback,
    ConversationCallbackProcessor,
)
from storage.database import a_session_maker

from openhands.core.logger import openhands_logger as logger
from openhands.core.schema.agent import AgentState
from openhands.events.action import MessageAction
from openhands.events.observation.agent import AgentStateChangedObservation
from openhands.events.serialization.event import event_to_dict
from openhands.server.shared import conversation_manager

token_manager = TokenManager()
gitlab_manager = GitlabManager(token_manager)


class GitlabCallbackProcessor(ConversationCallbackProcessor):
    """Processor for sending conversation summaries to GitLab.

    This processor is used to send summaries of conversations to GitLab
    when agent state changes occur.
    """

    gitlab_view: GitlabViewType
    send_summary_instruction: bool = True

    async def _send_message_to_gitlab(self, message: str) -> None:
        """Send a message to GitLab.

        Args:
            message: The message content to send to GitLab
        """
        try:
            # Get the token manager
            token_manager = TokenManager()
            gitlab_manager = GitlabManager(token_manager)

            # Send the message directly as a string
            await gitlab_manager.send_message(message, self.gitlab_view)

            logger.info(
                f'[GitLab] Sent summary message to {self.gitlab_view.full_repo_name}#{self.gitlab_view.issue_number}'
            )
        except Exception as e:
            logger.exception(f'[GitLab] Failed to send summary message: {str(e)}')

    async def __call__(
        self,
        callback: ConversationCallback,
        observation: AgentStateChangedObservation,
    ) -> None:
        """
        Process a conversation event by sending a summary to GitLab.

        Args:
            callback: The conversation callback
            observation: The AgentStateChangedObservation that triggered the callback
        """
        logger.info(f'[GitLab] Callback agent state was {observation.agent_state}')
        if observation.agent_state not in (
            AgentState.AWAITING_USER_INPUT,
            AgentState.FINISHED,
        ):
            return

        conversation_id = callback.conversation_id
        try:
            # If we need to send a summary instruction first
            if self.send_summary_instruction:
                logger.info(
                    f'[GitLab] Sending summary instruction for conversation {conversation_id}'
                )

                # Get the summary instruction
                summary_instruction = get_summary_instruction()
                summary_event = event_to_dict(
                    MessageAction(content=summary_instruction)
                )

                # Add the summary instruction to the event stream
                logger.info(
                    f'[GitLab] Sending summary instruction to conversation {conversation_id} {summary_event}'
                )
                await conversation_manager.send_event_to_conversation(
                    conversation_id, summary_event
                )

                logger.info(
                    f'[GitLab] Sent summary instruction to conversation {conversation_id} {summary_event}'
                )

                # Update the processor state
                self.send_summary_instruction = False
                callback.set_processor(self)
                callback.updated_at = datetime.now()
                async with a_session_maker() as session:
                    session.merge(callback)
                    await session.commit()
                return

            # Extract the summary from the event store
            logger.info(
                f'[GitLab] Extracting summary for conversation {conversation_id}'
            )
            summary = await extract_summary_from_conversation_manager(
                conversation_manager, conversation_id
            )

            # Send the summary to GitLab
            asyncio.create_task(self._send_message_to_gitlab(summary))

            logger.info(f'[GitLab] Summary sent for conversation {conversation_id}')

            # Mark callback as completed status
            callback.status = CallbackStatus.COMPLETED
            callback.updated_at = datetime.now()
            async with a_session_maker() as session:
                session.merge(callback)
                await session.commit()

        except Exception as e:
            logger.exception(
                f'[GitLab] Error processing conversation callback: {str(e)}'
            )
