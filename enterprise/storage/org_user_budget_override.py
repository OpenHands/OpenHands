from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Identity
from sqlalchemy.orm import Mapped, mapped_column
from storage.base import Base


class OrgUserBudgetOverride(Base):
    __tablename__ = 'org_user_budget_override'

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    org_id: Mapped[UUID] = mapped_column(
        ForeignKey('org.id'), nullable=False, index=True
    )
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey('user.id'), nullable=False, index=True
    )
    monthly_limit: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_disabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
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
