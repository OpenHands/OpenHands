from datetime import datetime, timezone

from pydantic import BaseModel, Field


# Replacement for utc_now
def utc_now():
    return datetime.now(timezone.utc)


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
