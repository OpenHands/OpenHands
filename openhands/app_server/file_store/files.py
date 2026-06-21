from abc import ABC, abstractmethod

from pydantic import ConfigDict

from openhands.sdk.utils.models import DiscriminatedUnionMixin

# Single source of truth for the encoding used by all FileStore
# implementations. Text contents are always persisted and read back as UTF-8,
# regardless of the host platform's default encoding (e.g. Windows cp1252 or a
# POSIX/C 'ascii' locale), so files round-trip consistently across backends.
TEXT_ENCODING = 'utf-8'


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
