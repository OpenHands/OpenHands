from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import Boolean, DateTime, ForeignKey, Identity, Integer
from sqlalchemy.orm import Mapped, mapped_column
from storage.base import Base


class OrgBudgetThreshold(Base):
    __tablename__ = 'org_budget_threshold'

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    org_id: Mapped[UUID] = mapped_column(
        ForeignKey('org.id'), nullable=False, index=True
    )
    percentage: Mapped[int] = mapped_column(Integer, nullable=False)
    email_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    slack_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    last_triggered_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_triggered_cycle_start: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )
