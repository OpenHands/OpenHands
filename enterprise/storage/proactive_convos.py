from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column
from storage.base import Base


class ProactiveConversation(Base):
    __tablename__ = 'proactive_conversation_table'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    repo_id: Mapped[str] = mapped_column(String, nullable=False)
    pr_number: Mapped[int] = mapped_column(nullable=False)
    workflow_runs: Mapped[list[Any]] = mapped_column(JSON, nullable=False)
    commit: Mapped[str] = mapped_column(String, nullable=False)
    conversation_starter_sent: Mapped[bool] = mapped_column(
        nullable=False, default=False
    )
    last_updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )
