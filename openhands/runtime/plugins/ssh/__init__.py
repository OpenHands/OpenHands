import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from openhands.core.logger import openhands_logger as logger
from openhands.events.action import Action
from openhands.events.observation import Observation
from openhands.runtime.plugins.requirement import Plugin, PluginRequirement
from openhands.runtime.utils.system import check_port_available

SSH_PORT = 2222
SSH_PUBLIC_KEYS_ENV = 'OH_SSH_PUBLIC_KEYS'


@dataclass
class SSHRequirement(PluginRequirement):
    name: str = 'ssh'


class SSHPlugin(Plugin):
    """Plugin to start SSH server for local VSCode Remote-SSH access."""

    name: str = 'ssh'
    ssh_port: Optional[int] = None
    sshd_process: Optional[asyncio.subprocess.Process] = None

    def _setup_authorized_keys(self, username: str) -> int:
        """Set up authorized_keys file from environment variable.

        Returns the number of keys added.
        """
        ssh_keys_str = os.environ.get(SSH_PUBLIC_KEYS_ENV, '')
        if not ssh_keys_str:
            return 0

        # SSH keys are passed as newline-separated values
        ssh_keys = [k.strip() for k in ssh_keys_str.split('\n') if k.strip()]
        if not ssh_keys:
            return 0

        # Determine the SSH directory based on the username
        if username == 'root':
            ssh_dir = Path('/root/.ssh')
        else:
            ssh_dir = Path(f'/home/{username}/.ssh')

        # Create .ssh directory if it doesn't exist
        ssh_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(ssh_dir, 0o700)

        # Write authorized_keys file
        authorized_keys_path = ssh_dir / 'authorized_keys'
        with open(authorized_keys_path, 'w') as f:
            for key in ssh_keys:
                f.write(f'{key}\n')

        os.chmod(authorized_keys_path, 0o600)

        # Try to set correct ownership (may fail if not root)
        try:
            import pwd

            pw = pwd.getpwnam(username)
            os.chown(ssh_dir, pw.pw_uid, pw.pw_gid)
            os.chown(authorized_keys_path, pw.pw_uid, pw.pw_gid)
        except (KeyError, PermissionError):
            pass

        logger.info(f'Added {len(ssh_keys)} SSH public key(s) to authorized_keys')
        return len(ssh_keys)

    async def initialize(self, username: str, runtime_id: str | None = None) -> None:
        # Check if we're on Windows - SSH plugin is not supported on Windows
        if os.name == 'nt' or sys.platform == 'win32':
            self.ssh_port = None
            logger.warning(
                'SSH plugin is not supported on Windows. Plugin will be disabled.'
            )
            return

        try:
            self.ssh_port = int(os.environ.get('SSH_PORT', SSH_PORT))
        except (KeyError, ValueError):
            self.ssh_port = SSH_PORT
            logger.debug(
                f'SSH_PORT environment variable not set, using default port {SSH_PORT}'
            )

        if not check_port_available(self.ssh_port):
            logger.warning(
                f'Port {self.ssh_port} is not available. SSH plugin will be disabled.'
            )
            self.ssh_port = None
            return

        # Set up authorized_keys from environment variable
        num_keys = self._setup_authorized_keys('openhands')

        # Start sshd in the foreground (will be managed by this process)
        cmd = f'/usr/sbin/sshd -D -p {self.ssh_port}'

        logger.debug(f'Starting SSH server on port {self.ssh_port}')

        self.sshd_process = await asyncio.create_subprocess_shell(
            cmd,
            stderr=asyncio.subprocess.STDOUT,
            stdout=asyncio.subprocess.PIPE,
        )

        # Give sshd a moment to start
        await asyncio.sleep(0.5)

        # Check if sshd is running
        if self.sshd_process.returncode is not None:
            logger.warning(
                f'SSH server failed to start (exit code: {self.sshd_process.returncode})'
            )
            self.ssh_port = None
            return

        logger.info(f'SSH server started on port {self.ssh_port}')
        if num_keys > 0:
            logger.info(
                f'SSH key authentication enabled with {num_keys} key(s). '
                f'Connect using: ssh -p {self.ssh_port} openhands@<host>'
            )
        else:
            logger.info(
                f'Connect using: ssh -p {self.ssh_port} openhands@<host> (password: openhands)'
            )

    async def run(self, action: Action) -> Observation:
        """Run the plugin for a given action."""
        raise NotImplementedError('SSHPlugin does not support run method')
