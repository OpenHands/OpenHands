from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EngagementStatus = Literal["draft", "active", "paused", "completed", "archived"]
AutonomyMode = Literal["manual", "semi_autonomous", "autonomous"]
RuntimeProfile = Literal["web", "network", "mobile", "sast"]
SandboxStatus = Literal["stopped", "provisioning", "running", "error"]


class EngagementCreate(BaseModel):
    name: str
    client_name: str
    description: str | None = None
    runtime_profile: RuntimeProfile = "web"
    autonomy_mode: AutonomyMode = "semi_autonomous"


class EngagementUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    status: EngagementStatus | None = None
    autonomy_mode: AutonomyMode | None = None
    runtime_profile: RuntimeProfile | None = None


class EngagementOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    client_name: str
    description: str | None
    status: str
    scope_authorized_at: datetime | None
    scope_document_url: str | None
    autonomy_mode: str
    runtime_profile: str
    sandbox_status: str | None
    sandbox_compose_project: str | None
    defectdojo_engagement_id: int | None
    created_by: str
    created_at: datetime
    updated_at: datetime


class EngagementListResponse(BaseModel):
    items: list[EngagementOut]
    total: int


class ProvisionResponse(BaseModel):
    job_id: uuid.UUID
    status: SandboxStatus
    sandbox_compose_project: str
