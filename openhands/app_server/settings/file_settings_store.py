from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from openhands.app_server.settings.settings_models import Settings
from openhands.app_server.settings.settings_store import SettingsStore
from openhands.app_server.utils.file_store_mixin import FileStoreMixin
from openhands.core.config.openhands_config import OpenHandsConfig


@dataclass
class FileSettingsStore(FileStoreMixin, SettingsStore):
    root_dir: Path
    filename: str = 'settings.json'

    async def load(self) -> Settings | None:
        try:
            json_str = await self._read_file_async()
            kwargs = json.loads(json_str)
            settings = Settings(**kwargs)

            # Turn on V1 in OpenHands
            # We can simplify / remove this as part of V0 removal
            settings.v1_enabled = True

            return settings
        except FileNotFoundError:
            return None

    async def store(self, settings: Settings) -> None:
        json_str = settings.model_dump_json(
            context={'expose_secrets': True, 'persist_settings': True}
        )
        await self._write_file_async(json_str)

    @classmethod
    async def get_instance(
        cls, config: OpenHandsConfig, user_id: str | None
    ) -> FileSettingsStore:
        root_dir = cls._resolve_root_dir(config.file_store_path)
        return FileSettingsStore(root_dir=root_dir)
