from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, String, text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from storage.base import Base

if TYPE_CHECKING:
    from storage.org import Org


class StripeCustomer(Base):
    """
    Represents a stripe customer. We cannot simply use the stripe API for this because:
    "Do not use search in read-after-write flows where strict consistency is necessary.
    Under normal operating conditions, data is searchable in less than a minute.
    Occasionally, propagation of new or updated data can be up to an hour behind during outages"
    """

    __tablename__ = 'stripe_customers'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    keycloak_user_id: Mapped[str] = mapped_column(String, nullable=False)
    org_id: Mapped[UUID | None] = mapped_column(ForeignKey('org.id'), nullable=True)
    stripe_customer_id: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=text('CURRENT_TIMESTAMP'), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=text('CURRENT_TIMESTAMP'),
        onupdate=text('CURRENT_TIMESTAMP'),
        nullable=False,
    )

    # Relationships
    org: Mapped['Org | None'] = relationship('Org', back_populates='stripe_customers')
