import asyncio
import logging
import os
from typing import AsyncGenerator

import docker
from fastapi import Request
from pydantic import Field

from openhands.app_server.errors import SandboxError
from openhands.app_server.sandbox.preset_sandbox_spec_service import (
    PresetSandboxSpecService,
)
from openhands.app_server.sandbox.sandbox_spec_models import (
    SandboxSpecInfo,
)
from openhands.app_server.sandbox.sandbox_spec_service import (
    SandboxSpecService,
    SandboxSpecServiceInjector,
    get_agent_server_env,
    get_agent_server_image,
    get_forwarded_env,
)
from openhands.app_server.services.injector import InjectorState

_global_docker_client: docker.DockerClient | None = None
_logger = logging.getLogger(__name__)


def get_docker_client() -> docker.DockerClient:
    global _global_docker_client
    if _global_docker_client is None:
        _global_docker_client = docker.from_env()
    return _global_docker_client


_SANDBOX_WORKING_DIR = '/workspace/project'


def get_default_sandbox_specs():
    return [
        SandboxSpecInfo(
            id=get_agent_server_image(),
            command=['--port', '8000'],
            initial_env={
                'OPENVSCODE_SERVER_ROOT': '/openhands/.openvscode-server',
                'OH_ENABLE_VNC': '0',
                'LOG_JSON': 'false',
                'OH_CONVERSATIONS_PATH': '/workspace/conversations',
                'OH_BASH_EVENTS_DIR': '/workspace/bash_events',
                'PYTHONUNBUFFERED': '1',
                'LOG_LEVEL': 'WARNING',
                'ENV_LOG_LEVEL': os.getenv('OH_SANDBOX_LOG_LEVEL', '30'),
                'USER': 'openhands',
                'WORKSPACE_ROOT': _SANDBOX_WORKING_DIR,
                'EXTRA_PATH_PREFIX': f'{_SANDBOX_WORKING_DIR}/node_modules/.bin',
                'YARN_CACHE_FOLDER': '/opt/package-cache/yarn',
                'NPM_CONFIG_CACHE': '/opt/package-cache/npm',
                'PIP_CACHE_DIR': '/opt/package-cache/pip',
                'COREPACK_ENABLE_DOWNLOAD_PROMPT': '0',
                'AUTHENTICATION_SERVER_TYPE': 'remote',
                'DATABASE_SERVER_TYPE': 'neon',
                **get_forwarded_env(),
                **get_agent_server_env(),
            },
            working_dir=_SANDBOX_WORKING_DIR,
        )
    ]


class DockerSandboxSpecServiceInjector(SandboxSpecServiceInjector):
    specs: list[SandboxSpecInfo] = Field(
        default_factory=get_default_sandbox_specs,
        description='Preset list of sandbox specs',
    )
    pull_if_missing: bool = Field(
        default=True,
        description=(
            'Flag indicating that any missing specs should be pulled from '
            'remote repositories.'
        ),
    )
    _pull_task: asyncio.Task | None = None

    model_config = {'arbitrary_types_allowed': True}

    def start_background_pull(self) -> asyncio.Task:
        """Start pulling sandbox images in the background.

        Returns the created asyncio.Task so callers can optionally await it.
        """
        _logger.info('Starting background pull of sandbox images...')
        self._pull_task = asyncio.create_task(self._background_pull())
        return self._pull_task

    async def _background_pull(self):
        """Background pull wrapper that logs completion and errors."""
        try:
            await self.pull_missing_specs()
            _logger.info('Background pull of sandbox images completed successfully')
        except Exception:
            _logger.warning('Background pull of sandbox images failed', exc_info=True)
            raise

    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[SandboxSpecService, None]:
        if self.pull_if_missing:
            if self._pull_task is not None:
                # A background pull was started — wait for it instead of
                # starting a duplicate pull.
                if not self._pull_task.done():
                    _logger.info(
                        'Waiting for background sandbox image pull to complete...'
                    )
                await self._pull_task  # re-raises if the task failed
            else:
                # No background pull was started — pull inline (original behavior)
                await self.pull_missing_specs()
            # Prevent repeated checks - more efficient but it does mean if you
            # delete a docker image outside the app you need to restart
            self.pull_if_missing = False
        yield PresetSandboxSpecService(specs=self.specs)

    async def pull_missing_specs(self):
        await asyncio.gather(*[self.pull_spec_if_missing(spec) for spec in self.specs])

    async def pull_spec_if_missing(self, spec: SandboxSpecInfo):
        _logger.debug(f'Checking Docker Image: {spec.id}')
        try:
            docker_client = get_docker_client()
            try:
                docker_client.images.get(spec.id)
            except docker.errors.ImageNotFound:
                _logger.info(f'⬇️  Pulling Docker Image: {spec.id}')
                await self._pull_with_progress_logging(docker_client, spec.id)
                _logger.info(f'⬇️  Finished Pulling Docker Image: {spec.id}')
        except docker.errors.APIError as exc:
            raise SandboxError(f'Error Getting Docker Image: {spec.id}') from exc

    async def _pull_with_progress_logging(
        self, docker_client: docker.DockerClient, image_id: str
    ):
        """Pull Docker image with periodic progress logging every 5 seconds."""
        # Event to signal when pull is complete
        pull_complete = asyncio.Event()

        async def periodic_logger():
            """Log progress message every 5 seconds until pull is complete."""
            while not pull_complete.is_set():
                try:
                    await asyncio.wait_for(pull_complete.wait(), timeout=5.0)
                    break  # Pull completed
                except asyncio.TimeoutError:
                    # 5 seconds elapsed, log progress message
                    _logger.info(f'🔄 Downloading Docker Image: {image_id}...')

        async def pull_image():
            """Perform the actual Docker image pull."""
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, docker_client.images.pull, image_id)
            finally:
                pull_complete.set()

        # Run both tasks concurrently
        logger_task = asyncio.create_task(periodic_logger())
        pull_task = asyncio.create_task(pull_image())

        try:
            # Wait for pull to complete
            await pull_task
        finally:
            # Ensure logger task is cancelled if still running
            if not logger_task.done():
                logger_task.cancel()
                try:
                    await logger_task
                except asyncio.CancelledError:
                    pass
