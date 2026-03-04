"""Database model for experiment assignments.

This model tracks which experiments a conversation is assigned to and what variant
they received from PostHog feature flags.
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import Column, DateTime, String, UniqueConstraint
from storage.base import Base


class ExperimentAssignment(Base):  # type: ignore
    __tablename__ = 'experiment_assignments'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String, nullable=True, index=True)
    experiment_name = Column(String, nullable=False)
    variant = Column(String, nullable=False)

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

    __table_args__ = (
        UniqueConstraint(
            'conversation_id',
            'experiment_name',
            name='uq_experiment_assignments_conversation_experiment',
        ),
    )
