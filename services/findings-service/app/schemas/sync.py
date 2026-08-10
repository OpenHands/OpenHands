from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field

FindingStatus = Literal[
    "new",
    "triaging",
    "confirmed",
    "false_positive",
    "duplicate",
    "risk_accepted",
]


class SyncDefectDojoRequest(BaseModel):
    engagement_id: uuid.UUID
    status_filter: list[FindingStatus] = Field(default_factory=lambda: ["confirmed"])


class SyncJobResponse(BaseModel):
    job_id: uuid.UUID
    status: Literal["queued", "running", "completed", "failed"]


class SyncResult(BaseModel):
    synced: int
    failed: int
    finding_ids: list[uuid.UUID]
