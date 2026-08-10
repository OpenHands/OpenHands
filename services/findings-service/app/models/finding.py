from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON, Uuid

from app.db import Base

FINDING_STATUSES = (
    "new",
    "triaging",
    "confirmed",
    "false_positive",
    "duplicate",
    "risk_accepted",
)
FINDING_SEVERITIES = ("critical", "high", "medium", "low", "info")


class Finding(Base):
    __tablename__ = "findings"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    engagement_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    source_tool: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    asset: Mapped[str | None] = mapped_column(Text, nullable=True)
    endpoint: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="new")
    dedupe_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    fp_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    triaged_by: Mapped[str | None] = mapped_column(String(256), nullable=True)
    triaged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    defectdojo_id: Mapped[int | None] = mapped_column(nullable=True)
    defectdojo_synced_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    cvss_score: Mapped[Decimal | None] = mapped_column(Numeric(4, 1), nullable=True)
    cve_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)
    tags: Mapped[list | None] = mapped_column(JSON, nullable=True)
    # Ownership claim until EngMgr membership roundtrip exists (fail-closed IDOR).
    created_by: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
