from datetime import UTC, datetime

from sqlalchemy import Index, String
from sqlalchemy.orm import Mapped, mapped_column
from storage.base import Base


class ConversationWork(Base):
    __tablename__ = 'conversation_work'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(
        String, nullable=False, unique=True, index=True
    )
    user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    seconds: Mapped[float] = mapped_column(nullable=False, default=0.0)
    created_at: Mapped[str] = mapped_column(
        String, default=lambda: datetime.now(UTC).isoformat(), nullable=False
    )
    updated_at: Mapped[str] = mapped_column(
        String,
        default=lambda: datetime.now(UTC).isoformat(),
        onupdate=lambda: datetime.now(UTC).isoformat(),
        nullable=False,
    )

    # Create composite index for efficient queries
    __table_args__ = (
        Index('ix_conversation_work_user_conversation', 'user_id', 'conversation_id'),
    )
