from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from openhands.agent_server.utils import utc_now


class SandboxStatus(Enum):
    STARTING = 'STARTING'
    RUNNING = 'RUNNING'
    PAUSED = 'PAUSED'
    ERROR = 'ERROR'
    MISSING = 'MISSING'
    """Missing - possibly deleted"""


class ExposedUrl(BaseModel):
    """URL to access some named service within the container."""

    name: str
    url: str
    port: int


# Standard names
AGENT_SERVER = 'AGENT_SERVER'
VSCODE = 'VSCODE'
WORKER_1 = 'WORKER_1'
WORKER_2 = 'WORKER_2'


class SandboxInfo(BaseModel):
    """Information about a sandbox."""

    id: str
    created_by_user_id: str | None
    sandbox_spec_id: str
    status: SandboxStatus
    session_api_key: str | None = Field(
        description=(
            'Key to access sandbox, to be added as an `X-Session-API-Key` header '
            'in each request. In cases where the sandbox statues is STARTING or '
            'PAUSED, or the current user does not have full access '
            'the session_api_key will be None.'
        )
    )
    exposed_urls: list[ExposedUrl] | None = Field(
        default_factory=lambda: [],
        description=(
            'URLs exposed by the sandbox (App server, Vscode, etc...)'
            'Sandboxes with a status STARTING / PAUSED / ERROR may '
            'not return urls.'
        ),
    )
    created_at: datetime = Field(default_factory=utc_now)


class SandboxPage(BaseModel):
    items: list[SandboxInfo]
    next_page_id: str | None = None
