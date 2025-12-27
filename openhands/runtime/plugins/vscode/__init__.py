import asyncio
import os
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from openhands.core.logger import openhands_logger as logger
from openhands.events.action import Action
from openhands.events.observation import Observation
from openhands.runtime.plugins.requirement import Plugin, PluginRequirement
from openhands.runtime.utils.system import check_port_available
from openhands.utils.shutdown_listener import should_continue

RUNTIME_USERNAME = os.getenv('RUNTIME_USERNAME')
SU_TO_USER = os.getenv('SU_TO_USER', 'true').lower() in (
    '1',
    'true',
    't',
    'yes',
    'y',
    'on',
)


@dataclass
class VSCodeRequirement(PluginRequirement):
    name: str = 'vscode'


class VSCodePlugin(Plugin):
    name: str = 'vscode'
    vscode_port: Optional[int] = None
    vscode_connection_token: Optional[str] = None
    gateway_process: asyncio.subprocess.Process

    async def initialize(self, username: str, runtime_id: str | None = None) -> None:
        # Check if we're on Windows - VSCode plugin is not supported on Windows
        if os.name == 'nt' or sys.platform == 'win32':
            self.vscode_port = None
            self.vscode_connection_token = None
            logger.warning(
                'VSCode plugin is not supported on Windows. Plugin will be disabled.'
            )
            return

        if username not in filter(None, [RUNTIME_USERNAME, 'root', 'openhands']):
            self.vscode_port = None
            self.vscode_connection_token = None
            logger.warning(
                'VSCodePlugin is only supported for root or openhands user. '
                'It is not yet supported for other users (i.e., when running LocalRuntime).'
            )
            return

        # Set up VSCode settings.json
        self._setup_vscode_settings()

        try:
            self.vscode_port = int(os.environ['VSCODE_PORT'])
        except (KeyError, ValueError):
            logger.warning(
                'VSCODE_PORT environment variable not set or invalid. VSCode plugin will be disabled.'
            )
            return

        self.vscode_connection_token = str(uuid.uuid4())
        if not check_port_available(self.vscode_port):
            logger.warning(
                f'Port {self.vscode_port} is not available. VSCode plugin will be disabled.'
            )
            return
        workspace_path = os.getenv('WORKSPACE_MOUNT_PATH_IN_SANDBOX', '/workspace')
        # Compute base path for OpenVSCode Server when running behind a path-based router
        base_path_flag = ''
        # Allow explicit override via environment
        explicit_base = os.getenv('OPENVSCODE_SERVER_BASE_PATH')
        if explicit_base:
            explicit_base = (
                explicit_base if explicit_base.startswith('/') else f'/{explicit_base}'
            )
            base_path_flag = f' --server-base-path {explicit_base.rstrip("/")}'
        else:
            # If runtime_id passed explicitly (preferred), use it
            runtime_url = os.getenv('RUNTIME_URL', '')
            if runtime_url and runtime_id:
                parsed = urlparse(runtime_url)
                path = parsed.path or '/'
                path_mode = path.startswith(f'/{runtime_id}')
                if path_mode:
                    base_path_flag = f' --server-base-path /{runtime_id}/vscode'

            cmd = (
                (
                    f"su - {username} -s /bin/bash << 'EOF'\n"
                    if SU_TO_USER
                    else "/bin/bash << 'EOF'\n"
                )
                + f'sudo chown -R {username}:{username} /openhands/.openvscode-server\n'
                + f'cd {workspace_path}\n'
                + 'exec /openhands/.openvscode-server/bin/openvscode-server '
                + f'--host 0.0.0.0 --connection-token {self.vscode_connection_token} '
                + f'--port {self.vscode_port} --disable-workspace-trust{base_path_flag}\n'
                + 'EOF'
            )

        # Using asyncio.create_subprocess_shell instead of subprocess.Popen
        # to avoid ASYNC101 linting error
        self.gateway_process = await asyncio.create_subprocess_shell(
            cmd,
            stderr=asyncio.subprocess.STDOUT,
            stdout=asyncio.subprocess.PIPE,
        )
        # read stdout until the kernel gateway is ready
        output = ''
        while should_continue() and self.gateway_process.stdout is not None:
            line_bytes = await self.gateway_process.stdout.readline()
            line = line_bytes.decode('utf-8')
            print(line)
            output += line
            if 'at' in line:
                break
            await asyncio.sleep(1)
            logger.debug('Waiting for VSCode server to start...')

        logger.debug(
            f'VSCode server started at port {self.vscode_port}. Output: {output}'
        )

    def _setup_vscode_settings(self) -> None:
        """Set up VSCode settings by creating the .vscode directory in the workspace
        and copying the settings.json file there.
        """
        # Get the path to the settings.json file in the plugin directory
        current_dir = Path(__file__).parent
        settings_path = current_dir / 'settings.json'

        # Create the .vscode directory in the workspace if it doesn't exist
        workspace_dir = Path(os.getenv('WORKSPACE_BASE', '/workspace'))
        vscode_dir = workspace_dir / '.vscode'
        vscode_dir.mkdir(parents=True, exist_ok=True)

        # Copy the settings.json file to the .vscode directory
        target_path = vscode_dir / 'settings.json'
        shutil.copy(settings_path, target_path)

        # Make sure the settings file is readable and writable by all users
        os.chmod(target_path, 0o666)

        logger.debug(f'VSCode settings copied to {target_path}')

    async def run(self, action: Action) -> Observation:
        """Run the plugin for a given action."""
        raise NotImplementedError('VSCodePlugin does not support run method')
