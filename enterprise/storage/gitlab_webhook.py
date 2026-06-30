import sys
from datetime import datetime
from enum import IntEnum
from typing import Any

from sqlalchemy import (
    ARRAY,
    DateTime,
    Index,
    String,
    Text,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column
from storage.base import Base


class WebhookStatus(IntEnum):
    PENDING = 0  # Conditions for installation webhook need checking
    VERIFIED = 1  # Conditions are met for installing webhook
    RATE_LIMITED = 2  # API was rate limited, failed to check
    INVALID = 3  # Unexpected error occur when checking (keycloak connection, etc)


class GitlabWebhook(Base):
    """
    Represents a Gitlab webhook configuration for a repository or group.
    """

    __tablename__ = 'gitlab_webhook'

    __table_args__ = (
        # Webhook lookups filter by project_id or group_id (gitlab_webhook_store
        # get/update/delete/reset by resource), but the table had no non-PK index,
        # so every lookup full-scanned it (~77K seq scans / 395M rows read, INC-95).
        Index('ix_gitlab_webhook_project_id', 'project_id'),
        Index('ix_gitlab_webhook_group_id', 'group_id'),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    group_id: Mapped[str | None] = mapped_column(String, nullable=True)
    project_id: Mapped[str | None] = mapped_column(String, nullable=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    webhook_exists: Mapped[bool] = mapped_column(nullable=False)
    webhook_url: Mapped[str | None] = mapped_column(String, nullable=True)
    webhook_secret: Mapped[str | None] = mapped_column(String, nullable=True)
    webhook_uuid: Mapped[str | None] = mapped_column(String, nullable=True)
    # Use Text for tests (SQLite compatibility) and ARRAY for production (PostgreSQL)
    scopes: Mapped[Any] = mapped_column(
        Text if 'pytest' in sys.modules else ARRAY(Text), nullable=True
    )
    last_synced: Mapped[datetime | None] = mapped_column(
        DateTime,
        server_default=text('CURRENT_TIMESTAMP'),
        onupdate=text('CURRENT_TIMESTAMP'),
        nullable=True,
    )

    def __repr__(self) -> str:
        return (
            f'<GitlabWebhook(id={self.id}, group_id={self.group_id}, '
            f'project_id={self.project_id}, last_synced={self.last_synced})>'
        )
