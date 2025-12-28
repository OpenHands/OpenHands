"""SQLAlchemy model for telemetry identity.

This model stores persistent identity information that must survive container restarts
for the OpenHands Enterprise Telemetry Service.
"""

from datetime import UTC, datetime
from typing import Optional

from sqlalchemy import CheckConstraint, Column, DateTime, Integer, String
from storage.base import Base


class TelemetryIdentity(Base):  # type: ignore
    """Stores persistent identity information for telemetry.

    This table is designed to contain exactly one row (enforced by database constraint)
    that maintains only the identity data that cannot be reliably recomputed:
    - customer_id: Established relationship with Replicated
    - instance_id: Generated once, must remain stable

    Operational data like timestamps are derived from the telemetry_metrics table.
    """

    __tablename__ = 'telemetry_replicated_identity'
    __table_args__ = (CheckConstraint('id = 1', name='single_identity_row'),)

    id = Column(Integer, primary_key=True, default=1)
    customer_id = Column(String(255), nullable=True)
    instance_id = Column(String(255), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )

    def __init__(
        self,
        customer_id: Optional[str] = None,
        instance_id: Optional[str] = None,
        **kwargs,
    ):
        """Initialize telemetry identity.

        Args:
            customer_id: Unique identifier for the customer
            instance_id: Unique identifier for this OpenHands instance
            **kwargs: Additional keyword arguments for SQLAlchemy
        """
        super().__init__(**kwargs)

        # Set defaults for fields that would normally be set by SQLAlchemy
        now = datetime.now(UTC)
        if not hasattr(self, 'created_at') or self.created_at is None:
            self.created_at = now
        if not hasattr(self, 'updated_at') or self.updated_at is None:
            self.updated_at = now

        # Force id to be 1 to maintain single-row constraint
        self.id = 1
        self.customer_id = customer_id
        self.instance_id = instance_id

    def set_customer_info(
        self,
        customer_id: Optional[str] = None,
        instance_id: Optional[str] = None,
    ) -> None:
        """Update customer and instance identification information.

        Args:
            customer_id: Unique identifier for the customer
            instance_id: Unique identifier for this OpenHands instance
        """
        if customer_id is not None:
            self.customer_id = customer_id
        if instance_id is not None:
            self.instance_id = instance_id

    @property
    def has_customer_info(self) -> bool:
        """Check if customer identification information is configured."""
        return bool(self.customer_id and self.instance_id)

    def __repr__(self) -> str:
        return (
            f"<TelemetryIdentity(customer_id='{self.customer_id}', "
            f"instance_id='{self.instance_id}')>"
        )

    class Config:
        from_attributes = True
