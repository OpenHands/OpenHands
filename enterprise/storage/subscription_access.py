from datetime import UTC, datetime

from sqlalchemy import DECIMAL, Column, DateTime, Enum, Integer, String
from storage.base import Base


class SubscriptionAccess(Base):  # type: ignore
    """Represents a user's subscription access record.
    Tracks subscription status, duration, payment information, and cancellation status.
    """

    __tablename__ = 'subscription_access'

    id = Column(Integer, primary_key=True, autoincrement=True)
    status = Column(
        Enum(
            'ACTIVE',
            'DISABLED',
            name='subscription_access_status_enum',
        ),
        nullable=False,
        index=True,
    )
    user_id = Column(String, nullable=False, index=True)
    start_at = Column(DateTime(timezone=True), nullable=True)
    end_at = Column(DateTime(timezone=True), nullable=True)
    amount_paid = Column(DECIMAL(19, 4), nullable=True)
    stripe_invoice_payment_id = Column(String, nullable=False)
    cancelled_at = Column(DateTime(timezone=True), nullable=True)
    stripe_subscription_id = Column(String, nullable=True, index=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),  # type: ignore[attr-defined]
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),  # type: ignore[attr-defined]
        onupdate=lambda: datetime.now(UTC),  # type: ignore[attr-defined]
        nullable=False,
    )

    class Config:
        from_attributes = True
