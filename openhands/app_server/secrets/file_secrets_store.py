from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from openhands.app_server.secrets.secrets_models import Secrets
from openhands.app_server.secrets.secrets_store import SecretsStore
from openhands.app_server.utils.file_store_mixin import FileStoreMixin
from openhands.core.config.openhands_config import OpenHandsConfig


@dataclass
class FileSecretsStore(FileStoreMixin, SecretsStore):
    root_dir: Path
    filename: str = 'secrets.json'

    async def load(self) -> Secrets | None:
        try:
            json_str = await self._read_file_async()
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
        await self._write_file_async(json_str)

    @classmethod
    async def get_instance(
        cls, config: OpenHandsConfig, user_id: str | None
    ) -> FileSecretsStore:
        root_dir = cls._resolve_root_dir(config.file_store_path)
        return FileSecretsStore(root_dir=root_dir)
