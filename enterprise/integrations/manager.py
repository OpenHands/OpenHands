from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from integrations.models import Message, SourceType

# TypeVar for view types - each manager subclass specifies its own view type
ViewT = TypeVar('ViewT')


class Manager(ABC, Generic[ViewT]):
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
    async def start_job(self, view: ViewT) -> None:
        """Kick off a job with openhands agent.

        Args:
            view: Integration-specific view object containing job context.
                  Each manager subclass accepts its own view type
                  (e.g., SlackViewInterface, JiraViewInterface, etc.)
        """
        raise NotImplementedError
