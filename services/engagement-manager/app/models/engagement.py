from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid

from app.db import Base


class Engagement(Base):
    __tablename__ = "engagements"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    client_name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="draft")
    scope_authorized_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    scope_document_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    autonomy_mode: Mapped[str] = mapped_column(
        String(32), nullable=False, default="semi_autonomous"
    )
    runtime_profile: Mapped[str] = mapped_column(
        String(32), nullable=False, default="web"
    )
    sandbox_status: Mapped[str | None] = mapped_column(
        String(32), nullable=True, default="stopped"
    )
    sandbox_compose_project: Mapped[str | None] = mapped_column(Text, nullable=True)
    defectdojo_engagement_id: Mapped[int | None] = mapped_column(nullable=True)
    created_by: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    scope_rules: Mapped[list[ScopeRule]] = relationship(
        "ScopeRule", back_populates="engagement", cascade="all, delete-orphan"
    )


class ScopeRule(Base):
    __tablename__ = "scope_rules"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    engagement_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("engagements.id", ondelete="CASCADE"),
        nullable=False,
    )
    rule_type: Mapped[str] = mapped_column(String(16), nullable=False)
    target_type: Mapped[str] = mapped_column(String(16), nullable=False)
    target_value: Mapped[str] = mapped_column(Text, nullable=False)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    engagement: Mapped[Engagement] = relationship(
        "Engagement", back_populates="scope_rules"
    )
