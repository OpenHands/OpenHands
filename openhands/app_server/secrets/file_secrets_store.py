from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

from openhands.app_server.secrets.secrets_store import SecretsStore
from openhands.storage.data_models.secrets import Secrets
from openhands.utils.async_utils import call_sync_from_async


@dataclass
class FileSecretsStore(SecretsStore):
    root_dir: Path
    filename: str = 'secrets.json'

    @property
    def file_path(self) -> Path:
        return self.root_dir / self.filename

    def _read_file(self) -> str:
        with open(self.file_path, 'r') as f:
            return f.read()

    def _write_file(self, contents: str) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        # Use atomic write: write to temp file, then rename
        # This prevents race conditions where concurrent writes could corrupt the file
        temp_path = f'{self.file_path}.tmp.{os.getpid()}.{threading.get_ident()}'
        try:
            with open(temp_path, 'w') as f:
                f.write(contents)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, self.file_path)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    async def load(self) -> Secrets | None:
        try:
            json_str = await call_sync_from_async(self._read_file)
            kwargs = json.loads(json_str)
            provider_tokens = {
                k: v
                for k, v in (kwargs.get('provider_tokens') or {}).items()
                if v.get('token')
            }
            kwargs['provider_tokens'] = provider_tokens
            secrets = Secrets(**kwargs)
            return secrets
        except FileNotFoundError:
            return None

    async def store(self, secrets: Secrets) -> None:
        json_str = secrets.model_dump_json(context={'expose_secrets': True})
        await call_sync_from_async(self._write_file, json_str)

    @classmethod
    async def get_instance(
        cls, config: object | None = None, user_id: str | None = None
    ) -> FileSecretsStore:
        # Import locally to avoid circular imports
        from openhands.app_server.config import get_global_config

        app_config = get_global_config()
        root_dir = app_config.persistence_dir
        return FileSecretsStore(root_dir=root_dir)
