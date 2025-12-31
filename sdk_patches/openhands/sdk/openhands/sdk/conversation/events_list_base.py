from abc import ABC, abstractmethod
from collections.abc import Sequence

from openhands.sdk.event import Event


class EventsListBase(Sequence[Event], ABC):
    """Abstract base class for event lists that can be appended to.

    This provides a common interface for both local EventLog and remote
    RemoteEventsList implementations, avoiding circular imports in protocols.
    """

    @abstractmethod
    def append(self, event: Event) -> None:
        """Add a new event to the list."""
        ...
