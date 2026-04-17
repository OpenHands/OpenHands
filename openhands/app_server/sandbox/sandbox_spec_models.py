from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from openhands.agent_server.utils import utc_now


class ExposedPort(BaseModel):
    """Exposed port for a service running in the sandbox.

    This defines a service that should be exposed and accessible from outside
    the sandbox. Used by both Docker and Firecracker sandbox implementations.
    """

    name: str = Field(description='Service name (e.g., SSH, VSCODE, AGENT_SERVER)')
    port: int = Field(description='Port number the service listens on')
    url_template: str | None = Field(
        default=None,
        description=(
            'URL template for the service. Supports {host} and {port} placeholders. '
            'Example: "ssh://{host}:{port}" for SSH, or None to use default http:// pattern.'
        ),
    )
    description: str | None = Field(
        default=None,
        description='Human-readable description of the service',
    )

    model_config = ConfigDict(frozen=True)


class SandboxSpecInfo(BaseModel):
    """A template for creating a Sandbox (e.g: A Docker Image vs Container)."""

    id: str
    command: list[str] | None
    created_at: datetime = Field(default_factory=utc_now)
    initial_env: dict[str, str] = Field(
        default_factory=dict, description='Initial Environment Variables'
    )
    working_dir: str = '/home/openhands/workspace'


class SandboxSpecInfoPage(BaseModel):
    items: list[SandboxSpecInfo]
    next_page_id: str | None = None
