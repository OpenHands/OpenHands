from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import DECIMAL, DateTime, Enum, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from storage.base import Base

if TYPE_CHECKING:
    from storage.org import Org


class BillingSession(Base):
    """
    Represents a Stripe billing session for credit purchases.
    Tracks the status of payment transactions and associated user information.
    """

    __tablename__ = 'billing_sessions'

    __table_args__ = (
        # The completed-credit lookup filters on (user_id, status) (user_store.py),
        # but neither column was indexed, so it full-scanned billing_sessions
        # (~4.6K seq scans / 42M rows read, INC-95). The other lookup filters by
        # id and is served by the primary key.
        Index('ix_billing_sessions_user_id_status', 'user_id', 'status'),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    org_id: Mapped[UUID | None] = mapped_column(ForeignKey('org.id'), nullable=True)
    status: Mapped[str] = mapped_column(
        Enum(
            'in_progress',
            'completed',
            'cancelled',
            'error',
            name='billing_session_status_enum',
        ),
        default='in_progress',
    )
    price: Mapped[Decimal] = mapped_column(DECIMAL(19, 4), nullable=False)
    price_code: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )

    # Relationships
    org: Mapped['Org | None'] = relationship('Org', back_populates='billing_sessions')
