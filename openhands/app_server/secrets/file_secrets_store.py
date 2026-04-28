from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from openhands.app_server.secrets.secrets_models import Secrets
from openhands.app_server.secrets.secrets_store import SecretsStore
from openhands.app_server.utils.io_utils import write_file_atomic
from openhands.core.config.openhands_config import OpenHandsConfig
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
        await write_file_atomic(self.file_path, json_str)

    @classmethod
    async def get_instance(
        cls, config: OpenHandsConfig, user_id: str | None
    ) -> FileSecretsStore:
        root_dir = Path(config.file_store_path)
        if str(root_dir).startswith('~'):
            root_dir = root_dir.expanduser()
        return FileSecretsStore(root_dir=root_dir)
