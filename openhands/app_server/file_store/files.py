from abc import ABC, abstractmethod

from pydantic import ConfigDict

from openhands.sdk.utils.models import DiscriminatedUnionMixin


class FileStore(DiscriminatedUnionMixin, ABC):
    """Base class for file storage implementations.

    Uses DiscriminatedUnionMixin for automatic `kind` field based on class name.
    """

    model_config = ConfigDict(extra='forbid', arbitrary_types_allowed=True)

    @abstractmethod
    def write(self, path: str, contents: str | bytes) -> None:
        pass

    @abstractmethod
    def read(self, path: str) -> str:
        pass

    @abstractmethod
    def list(self, path: str) -> list[str]:
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        pass
