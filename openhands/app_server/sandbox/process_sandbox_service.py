"""Process-based sandbox service implementation.

This service creates sandboxes by spawning separate agent server processes,
each running within a dedicated directory.
"""

import asyncio
import json
import logging
import os
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import AsyncGenerator

import base62
import httpx
import psutil
from fastapi import Request
from pydantic import BaseModel, ConfigDict, Field

from openhands.agent_server.utils import utc_now
from openhands.app_server.errors import SandboxError
from openhands.app_server.sandbox.sandbox_models import (
    AGENT_SERVER,
    ExposedUrl,
    SandboxInfo,
    SandboxPage,
    SandboxStatus,
)
from openhands.app_server.sandbox.sandbox_service import (
    SandboxService,
    SandboxServiceInjector,
)
from openhands.app_server.sandbox.sandbox_spec_models import SandboxSpecInfo
from openhands.app_server.sandbox.sandbox_spec_service import SandboxSpecService
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.utils.docker_utils import (
    replace_localhost_hostname_for_docker,
)

_logger = logging.getLogger(__name__)


class ProcessInfo(BaseModel):
    """Information about a running process."""

    pid: int
    port: int
    user_id: str | None
    working_dir: str
    session_api_key: str
    created_at: datetime
    sandbox_spec_id: str

    model_config = ConfigDict(frozen=True)


PROCESS_INFO_FILENAME = '.openhands-process.json'


# Global store
_processes: dict[str, ProcessInfo] = {}


@dataclass
class ProcessSandboxService(SandboxService):
    """Sandbox service that spawns separate agent server processes.

    Each sandbox is implemented as a separate Python process running the
    action execution server, with each process:
    - Operating in a dedicated directory
    - Listening on a unique port
    - Having its own session API key
    """

    user_id: str | None
    sandbox_spec_service: SandboxSpecService
    base_working_dir: str
    base_port: int
    python_executable: str
    agent_server_module: str
    health_check_path: str
    httpx_client: httpx.AsyncClient

    def __post_init__(self):
        """Initialize the service after dataclass creation."""
        os.makedirs(self.base_working_dir, exist_ok=True)

    def _metadata_path(self, working_dir: str | Path) -> Path:
        return Path(working_dir) / PROCESS_INFO_FILENAME

    def _working_dir_for_sandbox(self, sandbox_id: str) -> str:
        return os.path.join(self.base_working_dir, sandbox_id)

    def _is_within_base_working_dir(self, working_dir: str) -> bool:
        try:
            base = Path(self.base_working_dir).resolve()
            candidate = Path(working_dir).resolve()
            return os.path.commonpath([str(base), str(candidate)]) == str(base)
        except (OSError, ValueError):
            return False

    def _is_restorable_sandbox_dir(self, sandbox_id: str) -> bool:
        working_dir = Path(self._working_dir_for_sandbox(sandbox_id))
        if not working_dir.is_dir():
            return False
        if self._metadata_path(working_dir).exists():
            return True
        conversations_dir = working_dir / 'workspace' / 'conversations'
        return conversations_dir.is_dir() and any(
            (child / 'meta.json').is_file()
            for child in conversations_dir.iterdir()
            if child.is_dir()
        )

    def _save_process_info(self, sandbox_id: str, process_info: ProcessInfo) -> None:
        if not self._is_within_base_working_dir(process_info.working_dir):
            raise SandboxError(
                f'Refusing to persist sandbox metadata outside base dir: {sandbox_id}'
            )
        metadata_path = self._metadata_path(process_info.working_dir)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        payload = process_info.model_dump(mode='json')
        payload['sandbox_id'] = sandbox_id
        tmp_path = metadata_path.with_suffix('.json.tmp')
        with open(tmp_path, 'w') as f:
            json.dump(payload, f)
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, metadata_path)

    def _load_process_info_file(self, sandbox_id: str) -> ProcessInfo | None:
        metadata_path = self._metadata_path(self._working_dir_for_sandbox(sandbox_id))
        if not metadata_path.exists():
            return None
        try:
            payload = json.loads(metadata_path.read_text())
            stored_sandbox_id = payload.pop('sandbox_id', sandbox_id)
            if stored_sandbox_id != sandbox_id:
                _logger.warning(
                    'Ignoring sandbox metadata %s with mismatched sandbox_id %s',
                    metadata_path,
                    stored_sandbox_id,
                )
                return None
            process_info = ProcessInfo.model_validate(payload)
            if not self._is_within_base_working_dir(process_info.working_dir):
                _logger.warning(
                    'Ignoring sandbox metadata %s outside base working dir',
                    metadata_path,
                )
                return None
            return process_info
        except Exception:
            _logger.exception('Failed to load sandbox metadata: %s', metadata_path)
            return None

    async def _load_or_create_process_info(self, sandbox_id: str) -> ProcessInfo | None:
        process_info = _processes.get(sandbox_id)
        if process_info and self._is_within_base_working_dir(process_info.working_dir):
            return process_info

        process_info = self._load_process_info_file(sandbox_id)
        if process_info is not None:
            _processes[sandbox_id] = process_info
            return process_info

        if not self._is_restorable_sandbox_dir(sandbox_id):
            return None

        sandbox_spec = await self.sandbox_spec_service.get_default_sandbox_spec()
        working_dir = self._working_dir_for_sandbox(sandbox_id)
        created_at = datetime.fromtimestamp(Path(working_dir).stat().st_ctime, UTC)
        process_info = ProcessInfo(
            pid=-1,
            port=-1,
            user_id=self.user_id,
            working_dir=working_dir,
            session_api_key=base62.encodebytes(os.urandom(32)),
            created_at=created_at,
            sandbox_spec_id=sandbox_spec.id,
        )
        _processes[sandbox_id] = process_info
        self._save_process_info(sandbox_id, process_info)
        _logger.info('Recovered persisted process sandbox metadata for %s', sandbox_id)
        return process_info

    async def _restart_agent_process(
        self, sandbox_id: str, process_info: ProcessInfo
    ) -> ProcessInfo | None:
        sandbox_spec = await self.sandbox_spec_service.get_sandbox_spec(
            process_info.sandbox_spec_id
        )
        if sandbox_spec is None:
            sandbox_spec = await self.sandbox_spec_service.get_default_sandbox_spec()

        try:
            if process_info.pid > 0:
                process = psutil.Process(process_info.pid)
                if process.is_running():
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except psutil.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            pass

        port = self._find_unused_port()
        working_dir = process_info.working_dir
        Path(working_dir).mkdir(parents=True, exist_ok=True)
        process = await self._start_agent_process(
            sandbox_id=sandbox_id,
            port=port,
            working_dir=working_dir,
            session_api_key=process_info.session_api_key,
            sandbox_spec=sandbox_spec,
        )
        restarted = ProcessInfo(
            pid=process.pid,
            port=port,
            user_id=process_info.user_id or self.user_id,
            working_dir=working_dir,
            session_api_key=process_info.session_api_key,
            created_at=process_info.created_at,
            sandbox_spec_id=sandbox_spec.id,
        )
        _processes[sandbox_id] = restarted
        self._save_process_info(sandbox_id, restarted)
        if not await self._wait_for_server_ready(port):
            _logger.warning(
                'Restarted agent server for %s did not become ready', sandbox_id
            )
            return None
        return restarted

    def _find_unused_port(self) -> int:
        """Find an unused port starting from base_port."""
        port = self.base_port
        while port < self.base_port + 10000:  # Try up to 10000 ports
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    return port
            except OSError:
                port += 1
        raise SandboxError('No available ports found')

    def _create_sandbox_directory(self, sandbox_id: str) -> str:
        """Create a dedicated directory for the sandbox."""
        sandbox_dir = os.path.join(self.base_working_dir, sandbox_id)
        os.makedirs(sandbox_dir, exist_ok=True)
        return sandbox_dir

    async def _start_agent_process(
        self,
        sandbox_id: str,
        port: int,
        working_dir: str,
        session_api_key: str,
        sandbox_spec: SandboxSpecInfo,
    ) -> subprocess.Popen:
        """Start the agent server process."""

        # Prepare environment variables
        env = os.environ.copy()
        env.update(sandbox_spec.initial_env)
        env['SESSION_API_KEY'] = session_api_key
        env['OH_SESSION_API_KEYS_0'] = session_api_key

        # Prepare command arguments
        cmd = [
            self.python_executable,
            '-m',
            self.agent_server_module,
            '--port',
            str(port),
        ]

        _logger.info(
            f'Starting agent process for sandbox {sandbox_id}: {" ".join(cmd)}'
        )

        try:
            # Start the process, directing output to a log file to avoid pipe-buffer deadlocks
            log_path = os.path.join(working_dir, '.openhands-agent-server.log')
            with open(log_path, 'a', buffering=1) as log_handle:
                process = subprocess.Popen(
                    cmd, env=env, cwd=working_dir, stdout=log_handle, stderr=log_handle
                )

            # Wait a moment for the process to start
            await asyncio.sleep(1)

            # Check if process is still running
            if process.poll() is not None:
                raise SandboxError(
                    f'Agent process failed to start (exit code {process.returncode}). '
                    f'See {log_path} for details.'
                )

            return process

        except Exception as e:
            raise SandboxError(f'Failed to start agent process: {e}')

    async def _wait_for_server_ready(self, port: int, timeout: int = 30) -> bool:
        """Wait for the agent server to be ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                url = replace_localhost_hostname_for_docker(
                    f'http://localhost:{port}/alive'
                )
                response = await self.httpx_client.get(url, timeout=5.0)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'ok':
                        return True
            except Exception:
                pass
            await asyncio.sleep(1)
        return False

    def _get_process_status(self, process_info: ProcessInfo) -> SandboxStatus:
        """Get the status of a process."""
        if process_info.pid <= 0:
            return SandboxStatus.MISSING
        try:
            process = psutil.Process(process_info.pid)
            if process.is_running():
                status = process.status()
                if status == psutil.STATUS_STOPPED:
                    return SandboxStatus.PAUSED
                elif status in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
                    return SandboxStatus.MISSING
                else:
                    # RUNNING covers psutil 'running' AND 'sleeping' (typical for
                    # an asyncio uvicorn process waiting on epoll). The /alive
                    # check in _process_to_sandbox_info verifies real readiness.
                    return SandboxStatus.RUNNING
            else:
                return SandboxStatus.MISSING
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return SandboxStatus.MISSING

    async def _process_to_sandbox_info(
        self, sandbox_id: str, process_info: ProcessInfo
    ) -> SandboxInfo:
        """Convert process info to sandbox info."""
        status = self._get_process_status(process_info)
        if (
            status == SandboxStatus.MISSING
            and Path(process_info.working_dir).is_dir()
            and self._is_within_base_working_dir(process_info.working_dir)
        ):
            # The agent-server process is gone, but the sandbox directory and
            # conversation files are still restorable. Treat it as PAUSED so the
            # UI does not mark the conversation archived; resume_sandbox() will
            # spawn a new agent-server in the same directory.
            status = SandboxStatus.PAUSED

        exposed_urls = None
        session_api_key = None

        if status == SandboxStatus.RUNNING:
            # Check if server is actually responding
            try:
                url = replace_localhost_hostname_for_docker(
                    f'http://localhost:{process_info.port}{self.health_check_path}'
                )
                response = await self.httpx_client.get(url, timeout=5.0)
                if response.status_code == 200:
                    exposed_urls = [
                        ExposedUrl(
                            name=AGENT_SERVER,
                            url=f'http://localhost:{process_info.port}',
                            port=process_info.port,
                        ),
                    ]
                    session_api_key = process_info.session_api_key
                else:
                    status = SandboxStatus.ERROR
            except Exception:
                status = SandboxStatus.ERROR

        return SandboxInfo(
            id=sandbox_id,
            created_by_user_id=process_info.user_id,
            sandbox_spec_id=process_info.sandbox_spec_id,
            status=status,
            session_api_key=session_api_key,
            exposed_urls=exposed_urls,
            created_at=process_info.created_at,
        )

    async def search_sandboxes(
        self,
        page_id: str | None = None,
        limit: int = 100,
    ) -> SandboxPage:
        """Search for sandboxes."""
        for child in Path(self.base_working_dir).iterdir():
            if child.is_dir() and child.name not in _processes:
                await self._load_or_create_process_info(child.name)

        all_processes = [
            item
            for item in _processes.items()
            if self._is_within_base_working_dir(item[1].working_dir)
        ]
        all_processes.sort(key=lambda x: x[1].created_at, reverse=True)

        start_idx = 0
        if page_id:
            try:
                start_idx = int(page_id)
            except ValueError:
                start_idx = 0

        end_idx = start_idx + limit
        paginated_processes = all_processes[start_idx:end_idx]

        items = []
        for sandbox_id, process_info in paginated_processes:
            sandbox_info = await self._process_to_sandbox_info(sandbox_id, process_info)
            items.append(sandbox_info)

        next_page_id = None
        if end_idx < len(all_processes):
            next_page_id = str(end_idx)

        return SandboxPage(items=items, next_page_id=next_page_id)

    async def get_sandbox(self, sandbox_id: str) -> SandboxInfo | None:
        """Get a single sandbox."""
        process_info = await self._load_or_create_process_info(sandbox_id)
        if process_info is None:
            return None

        return await self._process_to_sandbox_info(sandbox_id, process_info)

    async def get_sandbox_by_session_api_key(
        self, session_api_key: str
    ) -> SandboxInfo | None:
        """Get a single sandbox by session API key."""
        for sandbox_id, process_info in list(_processes.items()):
            if (
                self._is_within_base_working_dir(process_info.working_dir)
                and process_info.session_api_key == session_api_key
            ):
                return await self._process_to_sandbox_info(sandbox_id, process_info)

        return None

    async def start_sandbox(
        self, sandbox_spec_id: str | None = None, sandbox_id: str | None = None
    ) -> SandboxInfo:
        """Start a new sandbox."""
        if sandbox_spec_id is None:
            sandbox_spec = await self.sandbox_spec_service.get_default_sandbox_spec()
        else:
            sandbox_spec_maybe = await self.sandbox_spec_service.get_sandbox_spec(
                sandbox_spec_id
            )
            if sandbox_spec_maybe is None:
                raise ValueError('Sandbox Spec not found')
            sandbox_spec = sandbox_spec_maybe

        if sandbox_id is None:
            sandbox_id = base62.encodebytes(os.urandom(16))
        session_api_key = base62.encodebytes(os.urandom(32))
        port = self._find_unused_port()
        working_dir = self._create_sandbox_directory(sandbox_id)

        process = await self._start_agent_process(
            sandbox_id=sandbox_id,
            port=port,
            working_dir=working_dir,
            session_api_key=session_api_key,
            sandbox_spec=sandbox_spec,
        )

        process_info = ProcessInfo(
            pid=process.pid,
            port=port,
            user_id=self.user_id,
            working_dir=working_dir,
            session_api_key=session_api_key,
            created_at=utc_now(),
            sandbox_spec_id=sandbox_spec.id,
        )
        _processes[sandbox_id] = process_info
        self._save_process_info(sandbox_id, process_info)

        if not await self._wait_for_server_ready(port):
            await self.delete_sandbox(sandbox_id)
            raise SandboxError('Agent Server Failed to start properly')

        return await self._process_to_sandbox_info(sandbox_id, process_info)

    async def resume_sandbox(self, sandbox_id: str) -> bool:
        """Resume a paused or persisted process sandbox."""
        process_info = await self._load_or_create_process_info(sandbox_id)
        if process_info is None:
            return False

        try:
            process = psutil.Process(process_info.pid)
            if process.status() == psutil.STATUS_STOPPED:
                process.resume()
                return True
            sandbox_info = await self._process_to_sandbox_info(sandbox_id, process_info)
            if sandbox_info.status == SandboxStatus.RUNNING:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
            pass

        restarted = await self._restart_agent_process(sandbox_id, process_info)
        return restarted is not None

    async def pause_sandbox(self, sandbox_id: str) -> bool:
        """Pause a running sandbox."""
        process_info = await self._load_or_create_process_info(sandbox_id)
        if process_info is None:
            return False

        try:
            process = psutil.Process(process_info.pid)
            if process.is_running():
                process.suspend()
            return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
            return Path(process_info.working_dir).is_dir()

    async def delete_sandbox(self, sandbox_id: str) -> bool:
        """Delete a sandbox."""
        process_info = await self._load_or_create_process_info(sandbox_id)
        if process_info is None:
            return False

        try:
            if process_info.pid > 0:
                process = psutil.Process(process_info.pid)
                if process.is_running():
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except psutil.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError, ValueError) as e:
            _logger.warning(f'Error terminating sandbox process {sandbox_id}: {e}')

        import shutil

        if os.path.exists(process_info.working_dir):
            shutil.rmtree(process_info.working_dir, ignore_errors=True)

        _processes.pop(sandbox_id, None)
        return True


class ProcessSandboxServiceInjector(SandboxServiceInjector):
    """Dependency injector for process sandbox services."""

    base_working_dir: str = Field(
        default_factory=lambda: os.path.join(
            tempfile.gettempdir(), 'openhands-sandboxes'
        ),
        description='Base directory for sandbox working directories',
    )
    base_port: int = Field(
        default=18000, description='Base port number for agent servers'
    )
    python_executable: str = Field(
        default=sys.executable,
        description='Python executable to use for agent processes',
    )
    agent_server_module: str = Field(
        default='openhands.agent_server',
        description='Python module for the agent server',
    )
    health_check_path: str = Field(
        default='/alive', description='Health check endpoint path'
    )

    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[SandboxService, None]:
        # Define inline to prevent circular lookup
        from openhands.app_server.config import (
            get_httpx_client,
            get_sandbox_spec_service,
            get_user_context,
        )

        async with (
            get_httpx_client(state, request) as httpx_client,
            get_sandbox_spec_service(state, request) as sandbox_spec_service,
            get_user_context(state, request) as user_context,
        ):
            user_id = await user_context.get_user_id()
            yield ProcessSandboxService(
                user_id=user_id,
                sandbox_spec_service=sandbox_spec_service,
                base_working_dir=self.base_working_dir,
                base_port=self.base_port,
                python_executable=self.python_executable,
                agent_server_module=self.agent_server_module,
                health_check_path=self.health_check_path,
                httpx_client=httpx_client,
            )
