import os

from pydantic import Field

from openhands.app_server.file_store.files import FileStore
from openhands.app_server.utils.logger import openhands_logger as logger


class InMemoryFileStore(FileStore):
    # Text-only by design: this store is part of the env-parsed FileStore config
    # union (so the value type must stay a primitive) and read() returns str, so
    # it cannot round-trip a binary archive. Not a valid RUNTIME_FILE_ARCHIVE
    # store — see workspace_archive._archive_store_type.
    files: dict[str, str] = Field(default_factory=dict)

    def write(self, path: str, contents: str | bytes) -> None:
        if isinstance(contents, bytes):
            contents = contents.decode('utf-8')
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
            parts = suffix.split('/')
            if parts[0] == '':
                parts.pop(0)
            if len(parts) == 1:
                files.append(file)
            else:
                dir_path = os.path.join(path, parts[0])
                if not dir_path.endswith('/'):
                    dir_path += '/'
                if dir_path not in files:
                    files.append(dir_path)
        return files

    def delete(self, path: str) -> None:
        try:
            keys_to_delete = [key for key in self.files.keys() if key.startswith(path)]
            for key in keys_to_delete:
                del self.files[key]
            logger.debug(f'Cleared in-memory file store: {path}')
        except Exception:
            logger.exception('Error clearing in-memory file store', stack_info=True)
