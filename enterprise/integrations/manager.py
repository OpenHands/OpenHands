from abc import ABC, abstractmethod
from typing import Any

from integrations.models import Message, SourceType


class Manager(ABC):
    manager_type: SourceType

    @abstractmethod
    async def receive_message(self, message: Message):
        "Receive message from integration"
        raise NotImplementedError

    @abstractmethod
    def send_message(self, message: str, *args: Any, **kwargs: Any):
        """Send message to integration from OpenHands server.

        Args:
            message: The message content to send (plain text string).
        """
        raise NotImplementedError

    @abstractmethod
    def start_job(self):
        "Kick off a job with openhands agent"
        raise NotImplementedError
