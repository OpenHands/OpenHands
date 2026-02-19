from datetime import UTC, datetime

from sqlalchemy import DECIMAL, Column, DateTime, Enum, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from storage.base import Base


class BillingSession(Base):  # type: ignore
    """
    Represents a Stripe billing session for credit purchases.
    Tracks the status of payment transactions and associated user information.
    """

    __tablename__ = 'billing_sessions'
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    org_id = Column(UUID(as_uuid=True), ForeignKey('org.id'), nullable=True)
    status = Column(
        Enum(
            'in_progress',
            'completed',
            'cancelled',
            'error',
            name='billing_session_status_enum',
        ),
        default='in_progress',
    )
    price = Column(DECIMAL(19, 4), nullable=False)
    price_code = Column(String, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),  # type: ignore[attr-defined]
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),  # type: ignore[attr-defined]
    )

    # Relationships
    org = relationship('Org', back_populates='billing_sessions')
