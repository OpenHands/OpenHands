from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

FindingSeverity = Literal["critical", "high", "medium", "low", "info"]
FindingStatus = Literal[
    "new",
    "triaging",
    "confirmed",
    "false_positive",
    "duplicate",
    "risk_accepted",
]


class FindingCreate(BaseModel):
    engagement_id: uuid.UUID
    source_tool: str = Field(max_length=64)
    title: str
    description: str | None = None
    severity: FindingSeverity
    asset: str | None = None
    endpoint: str | None = None
    evidence: dict[str, Any] | None = None
    cvss_score: Decimal | None = None
    cve_ids: list[str] | None = None
    tags: list[str] | None = None


class FindingUpdate(BaseModel):
    status: FindingStatus | None = None
    description: str | None = None
    severity: FindingSeverity | None = None
    tags: list[str] | None = None


class TriageRequest(BaseModel):
    new_status: FindingStatus
    fp_reason: str | None = None
    triaged_by: str


class FindingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    engagement_id: uuid.UUID
    source_tool: str
    title: str
    description: str | None
    severity: str
    asset: str | None
    endpoint: str | None
    evidence: dict[str, Any] | None
    status: str
    dedupe_hash: str | None
    fp_reason: str | None
    triaged_by: str | None
    triaged_at: datetime | None
    defectdojo_id: int | None
    defectdojo_synced_at: datetime | None
    cvss_score: Decimal | None
    cve_ids: list[str] | None
    tags: list[str] | None
    created_at: datetime
    updated_at: datetime


class FindingListResponse(BaseModel):
    items: list[FindingOut]
    total: int
    page: int
    page_size: int
    next_page: int | None


class DuplicateConflict(BaseModel):
    detail: str = "Duplicate finding"
    existing_finding_id: uuid.UUID


class FindingStats(BaseModel):
    by_severity: dict[str, int]
    by_status: dict[str, int]
    total: int
