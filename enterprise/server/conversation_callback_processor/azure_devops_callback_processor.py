import asyncio
from datetime import datetime

from integrations.azure_devops.azure_devops_manager import AzureDevOpsManager
from integrations.azure_devops.azure_devops_view_classes import AzureDevOpsViewType
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
from storage.database import session_maker

from openhands.core.logger import openhands_logger as logger
from openhands.core.schema.agent import AgentState
from openhands.events.action import MessageAction
from openhands.events.observation.agent import AgentStateChangedObservation
from openhands.events.serialization.event import event_to_dict
from openhands.server.shared import conversation_manager


class AzureDevOpsCallbackProcessor(ConversationCallbackProcessor):
    """Processor for sending conversation summaries to Azure DevOps.

    This processor is used to send summaries of conversations to Azure DevOps work items/PRs
    when agent state changes occur.
    """

    azure_devops_view: AzureDevOpsViewType
    send_summary_instruction: bool = True

    async def _send_message_to_azure_devops(self, message: str) -> None:
        """Send a message to Azure DevOps.

        Args:
            message: The message content to send to Azure DevOps
        """
        try:
            # Create a message object for Azure DevOps
            message_obj = Message(source=SourceType.OPENHANDS, message=message)

            # Get the token manager
            token_manager = TokenManager()

            # Create Azure DevOps manager
            from integrations.azure_devops.data_collector import (
                AzureDevOpsDataCollector,
            )

            azure_devops_manager = AzureDevOpsManager(
                token_manager, AzureDevOpsDataCollector()
            )

            # Send the message
            await azure_devops_manager.send_message(message_obj, self.azure_devops_view)

            logger.info(
                f'[Azure DevOps] Sent summary message to {self.azure_devops_view.full_repo_name}'
            )
        except Exception as e:
            logger.exception(f'[Azure DevOps] Failed to send summary message: {str(e)}')

    async def __call__(
        self,
        callback: ConversationCallback,
        observation: AgentStateChangedObservation,
    ) -> None:
        """Process a conversation event by sending a summary to Azure DevOps.

        Args:
            callback: The conversation callback
            observation: The AgentStateChangedObservation that triggered the callback
        """
        logger.info(
            f'[Azure DevOps] Callback agent state was {observation.agent_state}'
        )
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
                    f'[Azure DevOps] Sending summary instruction for conversation {conversation_id}'
                )

                # Get the summary instruction
                summary_instruction = get_summary_instruction()
                summary_event = event_to_dict(
                    MessageAction(content=summary_instruction)
                )

                # Add the summary instruction to the event stream
                logger.info(
                    f'[Azure DevOps] Sending summary instruction to conversation {conversation_id} {summary_event}'
                )
                await conversation_manager.send_event_to_conversation(
                    conversation_id, summary_event
                )

                logger.info(
                    f'[Azure DevOps] Sent summary instruction to conversation {conversation_id} {summary_event}'
                )

                # Update the processor state
                self.send_summary_instruction = False
                callback.set_processor(self)
                callback.updated_at = datetime.now()
                with session_maker() as session:
                    session.merge(callback)
                    session.commit()
                return

            # Extract the summary from the event store
            logger.info(
                f'[Azure DevOps] Extracting summary for conversation {conversation_id}'
            )
            summary = await extract_summary_from_conversation_manager(
                conversation_manager, conversation_id
            )

            # Send the summary to Azure DevOps
            asyncio.create_task(self._send_message_to_azure_devops(summary))

            logger.info(
                f'[Azure DevOps] Summary sent for conversation {conversation_id}'
            )

            # Mark callback as completed status
            callback.status = CallbackStatus.COMPLETED
            callback.updated_at = datetime.now()
            with session_maker() as session:
                session.merge(callback)
                session.commit()

        except Exception as e:
            logger.exception(
                f'[Azure DevOps] Error processing conversation callback: {str(e)}'
            )
