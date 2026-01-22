from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openhands.events.observation.agent import AgentStateChangedObservation

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Type

from pydantic import BaseModel, ConfigDict
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, text
from sqlalchemy import Enum as SQLEnum
from storage.base import Base

from openhands.utils.import_utils import get_impl


class ConversationCallbackProcessor(BaseModel, ABC):
    """
    Abstract base class for conversation callback processors.

    Conversation processors are invoked when events occur in a conversation
    to perform additional processing, notifications, or integrations.
    """

    model_config = ConfigDict(
        # Allow extra fields for flexibility
        extra='allow',
        # Allow arbitrary types
        arbitrary_types_allowed=True,
    )

    @abstractmethod
    async def __call__(
        self,
        callback: ConversationCallback,
        observation: 'AgentStateChangedObservation',
    ) -> None:
        """
        Process a conversation event.

        Args:
            conversation_id: The ID of the conversation to process
            observation: The AgentStateChangedObservation that triggered the callback
            callback: The conversation callback
        """


class CallbackStatus(Enum):
    """Status of a conversation callback."""

    ACTIVE = 'ACTIVE'
    COMPLETED = 'COMPLETED'
    ERROR = 'ERROR'


class ConversationCallback(Base):  # type: ignore
    """
    Model for storing conversation callbacks that process conversation events.
    """

    __tablename__ = 'conversation_callbacks'

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(
        String,
        ForeignKey('conversation_metadata.conversation_id'),
        nullable=False,
        index=True,
    )
    status = Column(
        SQLEnum(CallbackStatus), nullable=False, default=CallbackStatus.ACTIVE
    )
    processor_type = Column(String, nullable=False)
    processor_json = Column(Text, nullable=False)
    created_at = Column(
        DateTime,
        server_default=text('CURRENT_TIMESTAMP'),
        nullable=False,
    )
    updated_at = Column(
        DateTime,
        server_default=text('CURRENT_TIMESTAMP'),
        onupdate=datetime.now,
        nullable=False,
    )

    def get_processor(self) -> ConversationCallbackProcessor:
        """
        Get the processor instance from the stored processor type and JSON data.

        Returns:
            ConversationCallbackProcessor: The processor instance
        """
        # Import the processor class dynamically
        processor_type: Type[ConversationCallbackProcessor] = get_impl(
            ConversationCallbackProcessor, self.processor_type
        )
        processor = processor_type.model_validate_json(self.processor_json)
        return processor

    def set_processor(self, processor: ConversationCallbackProcessor) -> None:
        """
        Set the processor instance, storing its type and JSON representation.

        Args:
            processor: The ConversationCallbackProcessor instance to store
        """
        self.processor_type = (
            f'{processor.__class__.__module__}.{processor.__class__.__name__}'
        )
        self.processor_json = processor.model_dump_json()
