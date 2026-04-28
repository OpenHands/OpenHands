from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from openhands.app_server.settings.settings_models import Settings
from openhands.app_server.settings.settings_store import SettingsStore
from openhands.app_server.utils.io_utils import write_file_atomic
from openhands.core.config.openhands_config import OpenHandsConfig
from openhands.utils.async_utils import call_sync_from_async


@dataclass
class FileSettingsStore(SettingsStore):
    root_dir: Path
    filename: str = 'settings.json'

    @property
    def file_path(self) -> Path:
        return self.root_dir / self.filename

    def _read_file(self) -> str:
        with open(self.file_path, 'r') as f:
            return f.read()

    async def load(self) -> Settings | None:
        try:
            json_str = await call_sync_from_async(self._read_file)
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
        await write_file_atomic(self.file_path, json_str)

    @classmethod
    async def get_instance(
        cls, config: OpenHandsConfig, user_id: str | None
    ) -> FileSettingsStore:
        root_dir = Path(config.file_store_path)
        if str(root_dir).startswith('~'):
            root_dir = root_dir.expanduser()
        return FileSettingsStore(root_dir=root_dir)
