import asyncio
from datetime import datetime

from integrations.github.github_manager import GithubManager
from integrations.github.github_view import GithubViewType
from integrations.models import Message, SourceType
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

from openhands.core.logger import openhands_logger as logger
from openhands.core.schema.agent import AgentState
from openhands.events.action import MessageAction
from openhands.events.observation.agent import AgentStateChangedObservation
from openhands.events.serialization.event import event_to_dict
from openhands.server.shared import conversation_manager


class GithubCallbackProcessor(ConversationCallbackProcessor):
    """
    Processor for sending conversation summaries to GitHub.

    This processor is used to send summaries of conversations to GitHub issues/PRs
    when agent state changes occur.
    """

    github_view: GithubViewType
    send_summary_instruction: bool = True

    async def _send_message_to_github(self, message: str) -> None:
        """
        Send a message to GitHub.

        Args:
            message: The message content to send to GitHub
        """
        try:
            # Create a message object for GitHub
            message_obj = Message(source=SourceType.OPENHANDS, message=message)

            # Get the token manager
            token_manager = TokenManager()

            # Create GitHub manager
            from integrations.github.data_collector import GitHubDataCollector

            github_manager = GithubManager(token_manager, GitHubDataCollector())

            # Send the message
            await github_manager.send_message(message_obj, self.github_view)

            logger.info(
                f'[GitHub] Sent summary message to {self.github_view.full_repo_name}#{self.github_view.issue_number}'
            )
        except Exception as e:
            logger.exception(f'[GitHub] Failed to send summary message: {str(e)}')

    async def __call__(
        self,
        callback: ConversationCallback,
        observation: AgentStateChangedObservation,
    ) -> None:
        """
        Process a conversation event by sending a summary to GitHub.

        Args:
            callback: The conversation callback
            observation: The AgentStateChangedObservation that triggered the callback
        """
        logger.info(f'[GitHub] Callback agent state was {observation.agent_state}')
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
                    f'[GitHub] Sending summary instruction for conversation {conversation_id}'
                )

                # Get the summary instruction
                summary_instruction = get_summary_instruction()
                summary_event = event_to_dict(
                    MessageAction(content=summary_instruction)
                )

                # Add the summary instruction to the event stream
                logger.info(
                    f'[GitHub] Sending summary instruction to conversation {conversation_id} {summary_event}'
                )
                await conversation_manager.send_event_to_conversation(
                    conversation_id, summary_event
                )

                logger.info(
                    f'[GitHub] Sent summary instruction to conversation {conversation_id} {summary_event}'
                )

                # Update the processor state - the outer session will commit this
                self.send_summary_instruction = False
                callback.set_processor(self)
                callback.updated_at = datetime.now()
                return

            # Extract the summary from the event store
            logger.info(
                f'[GitHub] Extracting summary for conversation {conversation_id}'
            )
            summary = await extract_summary_from_conversation_manager(
                conversation_manager, conversation_id
            )

            # Send the summary to GitHub
            asyncio.create_task(self._send_message_to_github(summary))

            logger.info(f'[GitHub] Summary sent for conversation {conversation_id}')

            # Mark callback as completed status - the outer session will commit this
            callback.status = CallbackStatus.COMPLETED
            callback.updated_at = datetime.now()

        except Exception as e:
            logger.exception(
                f'[GitHub] Error processing conversation callback: {str(e)}'
            )
            # Mark callback as error to prevent infinite re-invocation
            # The outer session will commit this
            callback.status = CallbackStatus.ERROR
            callback.updated_at = datetime.now()
