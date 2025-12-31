import os

from openhands.sdk.io.base import FileStore
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)


class InMemoryFileStore(FileStore):
    files: dict[str, str]

    def __init__(self, files: dict[str, str] | None = None) -> None:
        self.files = {}
        if files is not None:
            self.files = files

    def write(self, path: str, contents: str | bytes) -> None:
        if isinstance(contents, bytes):
            contents = contents.decode("utf-8")
        self.files[path] = contents

    def read(self, path: str) -> str:
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    def list(self, path: str) -> list[str]:
        files = []
        for file in self.files:
            if not file.startswith(path):
                continue
            suffix = file.removeprefix(path)
            parts = suffix.split("/")
            if parts[0] == "":
                parts.pop(0)
            if len(parts) == 1:
                files.append(file)
            else:
                dir_path = os.path.join(path, parts[0])
                if not dir_path.endswith("/"):
                    dir_path += "/"
                if dir_path not in files:
                    files.append(dir_path)
        return files

    def delete(self, path: str) -> None:
        try:
            keys_to_delete = [key for key in self.files.keys() if key.startswith(path)]
            for key in keys_to_delete:
                del self.files[key]
            logger.debug(f"Cleared in-memory file store: {path}")
        except Exception as e:
            logger.error(f"Error clearing in-memory file store: {str(e)}")
