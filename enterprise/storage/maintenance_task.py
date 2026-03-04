from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Type

from pydantic import BaseModel, ConfigDict
from sqlalchemy import Column, DateTime, Integer, String, Text, text
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.dialects.postgresql import JSON
from storage.base import Base

from openhands.utils.import_utils import get_impl


class MaintenanceTaskProcessor(BaseModel, ABC):
    """Abstract base class for maintenance task processors.

    Maintenance processors are invoked to perform background maintenance
    tasks such as upgrading user settings, cleaning up data, etc.
    """

    model_config = ConfigDict(
        # Allow extra fields for flexibility
        extra='allow',
        # Allow arbitrary types
        arbitrary_types_allowed=True,
    )

    @abstractmethod
    async def __call__(self, task: MaintenanceTask) -> dict:
        """Process a maintenance task.

        Args:
            task: The maintenance task to process

        Returns:
            dict: Information about the task execution to store in the info column
        """


class MaintenanceTaskStatus(Enum):
    """Status of a maintenance task."""

    INACTIVE = 'INACTIVE'
    PENDING = 'PENDING'
    WORKING = 'WORKING'
    COMPLETED = 'COMPLETED'
    ERROR = 'ERROR'


class MaintenanceTask(Base):  # type: ignore
    """Model for storing maintenance tasks that perform background operations.
    """

    __tablename__ = 'maintenance_tasks'

    id = Column(Integer, primary_key=True, autoincrement=True)
    status = Column(
        SQLEnum(MaintenanceTaskStatus),
        nullable=False,
        default=MaintenanceTaskStatus.INACTIVE,
    )
    processor_type = Column(String, nullable=False)
    processor_json = Column(Text, nullable=False)
    delay = Column(Integer, server_default='0')
    started_at = Column(DateTime, nullable=True)
    info = Column(JSON, nullable=True)
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

    def get_processor(self) -> MaintenanceTaskProcessor:
        """Get the processor instance from the stored processor type and JSON data.

        Returns:
            MaintenanceTaskProcessor: The processor instance
        """
        # Import the processor class dynamically
        processor_type: Type[MaintenanceTaskProcessor] = get_impl(
            MaintenanceTaskProcessor, self.processor_type
        )
        processor = processor_type.model_validate_json(self.processor_json)
        return processor

    def set_processor(self, processor: MaintenanceTaskProcessor) -> None:
        """Set the processor instance, storing its type and JSON representation.

        Args:
            processor: The MaintenanceTaskProcessor instance to store
        """
        self.processor_type = (
            f'{processor.__class__.__module__}.{processor.__class__.__name__}'
        )
        self.processor_json = processor.model_dump_json()
