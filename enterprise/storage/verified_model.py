"""SQLAlchemy model for verified LLM models."""

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Identity,
    Integer,
    String,
    UniqueConstraint,
    func,
    text,
)
from storage.base import Base


class VerifiedModel(Base):  # type: ignore
    """A verified LLM model available in the model selector.

    The composite unique constraint on (model_name, provider) allows the same
    model name to exist under different providers (e.g. 'claude-sonnet' under
    both 'openhands' and 'anthropic').
    """

    __tablename__ = 'verified_models'
    __table_args__ = (
        UniqueConstraint('model_name', 'provider', name='uq_verified_model_provider'),
    )

    id = Column(Integer, Identity(), primary_key=True)
    model_name = Column(String(255), nullable=False)
    provider = Column(String(100), nullable=False, index=True)
    is_enabled = Column(
        Boolean, nullable=False, default=True, server_default=text('true')
    )
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )
