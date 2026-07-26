import errno
import importlib
import os
import shutil
import sys
import threading
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import model_validator

from openhands.app_server.file_store.files import FileStore
from openhands.app_server.utils.logger import openhands_logger as logger

fcntl: Any = None
msvcrt: Any = None
if sys.platform == 'win32':
    msvcrt = importlib.import_module('msvcrt')
else:
    fcntl = importlib.import_module('fcntl')

_T = TypeVar('_T')


class LocalFileStore(FileStore):
    root: str

    @model_validator(mode='after')
    def _setup_root(self) -> 'LocalFileStore':
        if self.root.startswith('~'):
            self.root = os.path.expanduser(self.root)
        os.makedirs(self.root, exist_ok=True)
        return self

    def get_full_path(self, path: str) -> str:
        if path.startswith('/'):
            path = path[1:]
        return os.path.join(self.root, path)

    @property
    def supports_locked_update(self) -> bool:
        return True

    def locked_update(self, path: str, update: Callable[[], _T]) -> _T:
        lock_path = self.get_full_path(f'{path}.lock')
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        locked = False
        try:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                locked = True
            elif msvcrt is not None:
                os.lseek(descriptor, 0, os.SEEK_SET)
                while True:
                    try:
                        msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
                        break
                    except OSError as exc:
                        if exc.errno not in (
                            errno.EACCES,
                            errno.EAGAIN,
                            errno.EDEADLK,
                        ):
                            raise
                locked = True
            return update()
        finally:
            try:
                if locked and fcntl is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                elif locked and msvcrt is not None:
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            finally:
                os.close(descriptor)

    def write(self, path: str, contents: str | bytes) -> None:
        full_path = self.get_full_path(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        mode = 'w' if isinstance(contents, str) else 'wb'

        # Use atomic write: write to temp file, then rename
        # This prevents race conditions where concurrent writes could corrupt the file
        temp_path = f'{full_path}.tmp.{os.getpid()}.{threading.get_ident()}'
        try:
            with open(temp_path, mode) as f:
                f.write(contents)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, full_path)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def write_from_path(self, path: str, source_path: str) -> None:
        # shutil.copyfile streams in chunks (never the whole file in RAM); keep
        # the same write-temp-then-atomic-rename to avoid torn concurrent writes.
        full_path = self.get_full_path(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        temp_path = f'{full_path}.tmp.{os.getpid()}.{threading.get_ident()}'
        try:
            shutil.copyfile(source_path, temp_path)
            os.replace(temp_path, full_path)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def read(self, path: str) -> str:
        full_path = self.get_full_path(path)
        with open(full_path, 'r') as f:
            return f.read()

    def list(self, path: str) -> list[str]:
        full_path = self.get_full_path(path)
        files = [os.path.join(path, f) for f in os.listdir(full_path)]
        files = [f + '/' if os.path.isdir(self.get_full_path(f)) else f for f in files]
        return files

    def delete(self, path: str) -> None:
        try:
            full_path = self.get_full_path(path)
            if not os.path.exists(full_path):
                logger.debug(f'Local path does not exist: {full_path}')
                return
            if os.path.isfile(full_path):
                os.remove(full_path)
                logger.debug(f'Removed local file: {full_path}')
            elif os.path.isdir(full_path):
                shutil.rmtree(full_path)
                logger.debug(f'Removed local directory: {full_path}')
        except Exception:
            logger.exception('Error clearing local file store', stack_info=True)
