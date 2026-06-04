import asyncio
import logging
from typing import AsyncGenerator

import podman
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
)
from openhands.app_server.services.injector import InjectorState

_global_podman_client: podman.PodmanClient | None = None
_logger = logging.getLogger(__name__)


def get_podman_client() -> podman.PodmanClient:
    global _global_podman_client
    if _global_podman_client is None:
        _global_podman_client = podman.from_env()
    return _global_podman_client


def get_default_sandbox_specs():
    return [
        SandboxSpecInfo(
            id=get_agent_server_image(),
            command=['--port', '8000'],
            initial_env={
                'OPENVSCODE_SERVER_ROOT': '/openhands/.openvscode-server',
                'OH_ENABLE_VNC': '0',
                'LOG_JSON': 'true',
                'OH_CONVERSATIONS_PATH': '/workspace/conversations',
                'OH_BASH_EVENTS_DIR': '/workspace/bash_events',
                'PYTHONUNBUFFERED': '1',
                'ENV_LOG_LEVEL': '20',
                **get_agent_server_env(),
            },
            working_dir='/workspace/project',
        )
    ]


class PodmanSandboxSpecServiceInjector(SandboxSpecServiceInjector):
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

    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[SandboxSpecService, None]:
        if self.pull_if_missing:
            await self.pull_missing_specs()
            # Prevent repeated checks - more efficient but it does mean if you
            # delete a podman image outside the app you need to restart
            self.pull_if_missing = False
        yield PresetSandboxSpecService(specs=self.specs)

    async def pull_missing_specs(self):
        await asyncio.gather(*[self.pull_spec_if_missing(spec) for spec in self.specs])

    async def pull_spec_if_missing(self, spec: SandboxSpecInfo):
        _logger.debug(f'Checking Podman Image: {spec.id}')
        try:
            podman_client = get_podman_client()
            try:
                podman_client.images.get(spec.id)
            except podman.errors.ImageNotFound:
                _logger.info(f'⬇️  Pulling Podman Image: {spec.id}')
                await self._pull_with_progress_logging(podman_client, spec.id)
                _logger.info(f'⬇️  Finished Pulling Podman Image: {spec.id}')
        except podman.errors.APIError as exc:
            raise SandboxError(f'Error Getting Podman Image: {spec.id}') from exc

    async def _pull_with_progress_logging(
        self, podman_client: podman.PodmanClient, image_id: str
    ):
        """Pull Podman image with periodic progress logging every 5 seconds."""
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
                    _logger.info(f'🔄 Downloading Podman Image: {image_id}...')

        async def pull_image():
            """Perform the actual Podman image pull."""
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, podman_client.images.pull, image_id)
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
