from __future__ import annotations

import os
from pathlib import Path

from alembic import command
from alembic.config import Config

from openhands.app_server.app_lifespan.app_lifespan_service import AppLifespanService


class OssAppLifespanService(AppLifespanService):
    run_alembic_on_startup: bool = True

    async def __aenter__(self):
        if self.run_alembic_on_startup:
            self.run_alembic()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        pass

    def run_alembic(self):
        # Run alembic upgrade head to ensure database is up to date
        alembic_dir = Path(__file__).parent / 'alembic'
        alembic_ini = alembic_dir / 'alembic.ini'

        # Create alembic config with absolute paths
        alembic_cfg = Config(str(alembic_ini))
        alembic_cfg.set_main_option('script_location', str(alembic_dir))

        # Change to alembic directory for the command execution
        original_cwd = os.getcwd()
        try:
            os.chdir(str(alembic_dir.parent))
            command.upgrade(alembic_cfg, 'head')
        finally:
            os.chdir(original_cwd)
