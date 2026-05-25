import os
from typing import AsyncGenerator

from fastapi import Request
from pydantic import Field

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


def _get_local_working_dir() -> str:
    """Return the agent workspace directory for local (non-Docker) process mode.

    Priority:
    1. OPENHANDS_WORK_DIR env var (used by run_local.sh and the CLI)
    2. /tmp/openhands-workspace as a safe fallback
    """
    work_dir = os.environ.get('OPENHANDS_WORK_DIR', '')
    if work_dir:
        work_dir = os.path.expanduser(work_dir)
    else:
        work_dir = '/tmp/openhands-workspace'
    os.makedirs(work_dir, exist_ok=True)
    return work_dir


def get_default_sandbox_specs():
    return [
        SandboxSpecInfo(
            id=get_agent_server_image(),
            command=['python', '-m', 'openhands.agent_server'],
            initial_env={
                # Keep tmux sockets on a short path; macOS default temp dirs can
                # exceed Unix socket path limits once libtmux appends tmux-UID.
                'TMUX_TMPDIR': '/tmp/openhands-tmux',
                # VSCode disabled for now
                'OH_ENABLE_VS_CODE': '0',
                **get_agent_server_env(),
            },
            working_dir=_get_local_working_dir(),
        )
    ]


class ProcessSandboxSpecServiceInjector(SandboxSpecServiceInjector):
    specs: list[SandboxSpecInfo] = Field(
        default_factory=get_default_sandbox_specs,
        description='Preset list of sandbox specs',
    )

    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[SandboxSpecService, None]:
        yield PresetSandboxSpecService(specs=self.specs)
