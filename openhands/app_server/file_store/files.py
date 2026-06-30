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

    def write_from_path(self, path: str, source_path: str) -> None:
        """Store the object at ``path`` from a local file at ``source_path``.

        The default reads the whole file into memory and delegates to ``write``;
        backends that can stream from disk override this to bound peak memory.
        Callers uploading large blobs (e.g. multi-GB workspace archives, where
        buffering one whole copy in RAM under concurrent deletes risks
        OOM-killing the pod) should prefer this over ``write(path, f.read())``.
        """
        with open(source_path, 'rb') as f:
            self.write(path, f.read())

    @abstractmethod
    def read(self, path: str) -> str:
        pass

    @abstractmethod
    def list(self, path: str) -> list[str]:
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        pass
