from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Enum, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column
from storage.base import Base


class Feedback(Base):
    __tablename__ = 'feedback'

    id: Mapped[str] = mapped_column(String, primary_key=True)
    version: Mapped[str] = mapped_column(String, nullable=False)
    email: Mapped[str] = mapped_column(String, nullable=False)
    polarity: Mapped[str] = mapped_column(
        Enum('positive', 'negative', name='polarity_enum'), nullable=False
    )
    permissions: Mapped[str] = mapped_column(
        Enum('public', 'private', name='permissions_enum'), nullable=False
    )
    trajectory: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class ConversationFeedback(Base):
    __tablename__ = 'conversation_feedback'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    event_id: Mapped[int | None] = mapped_column(nullable=True)
    rating: Mapped[int] = mapped_column(nullable=False)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now()
    )
