from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

from openhands.app_server.settings.settings_models import Settings
from openhands.app_server.settings.settings_store import SettingsStore
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
        await call_sync_from_async(self._write_file, json_str)

    @classmethod
    async def get_instance(
        cls, config: OpenHandsConfig, user_id: str | None
    ) -> FileSettingsStore:
        root_dir = Path(config.file_store_path)
        if str(root_dir).startswith('~'):
            root_dir = root_dir.expanduser()
        return FileSettingsStore(root_dir=root_dir)
