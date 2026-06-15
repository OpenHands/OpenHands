from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import DECIMAL, DateTime, Enum, String
from sqlalchemy.orm import Mapped, mapped_column
from storage.base import Base


class SubscriptionAccess(Base):
    """
    Represents a user's subscription access record.
    Tracks subscription status, duration, payment information, and cancellation status.
    """

    __tablename__ = 'subscription_access'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    status: Mapped[str] = mapped_column(
        Enum(
            'ACTIVE',
            'DISABLED',
            name='subscription_access_status_enum',
        ),
        nullable=False,
        index=True,
    )
    user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    start_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    end_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    amount_paid: Mapped[Decimal | None] = mapped_column(DECIMAL(19, 4), nullable=True)
    stripe_invoice_payment_id: Mapped[str] = mapped_column(String, nullable=False)
    cancelled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    stripe_subscription_id: Mapped[str | None] = mapped_column(
        String, nullable=True, index=True
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

    class Config:
        from_attributes = True
