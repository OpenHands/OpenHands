from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field

from openhands.agent_server.utils import OpenHandsUUID, utc_now


class RecaptchaLog(BaseModel):
    """Model for logging reCAPTCHA verification results."""

    id: OpenHandsUUID = Field(default_factory=uuid4)
    user_id: str | None = None
    score: float | None = None
    token: str | None = None
    action: str | None = None
    hostname: str | None = None
    apk_package_name: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class RecaptchaLogRequest(BaseModel):
    """Request model for logging reCAPTCHA verification results."""

    score: float | None = None
    token: str | None = None
    action: str | None = None
    hostname: str | None = None
    apk_package_name: str | None = None
