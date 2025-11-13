import sys
from enum import IntEnum

from sqlalchemy import (
    ARRAY,
    Boolean,
    Column,
    DateTime,
    Integer,
    String,
    Text,
    text,
)
from storage.base import Base


class WebhookStatus(IntEnum):
    PENDING = 0  # Conditions for installation webhook need checking
    VERIFIED = 1  # Conditions are met for installing webhook
    RATE_LIMITED = 2  # API was rate limited, failed to check
    INVALID = 3  # Unexpected error occur when checking (keycloak connection, etc)


class GitlabWebhook(Base):  # type: ignore
    """
    Represents a Gitlab webhook configuration for a repository or group.
    """

    __tablename__ = 'gitlab_webhook'
    id = Column(Integer, primary_key=True, autoincrement=True)
    group_id = Column(String, nullable=True)
    project_id = Column(String, nullable=True)
    user_id = Column(String, nullable=False)
    webhook_exists = Column(Boolean, nullable=False)
    webhook_url = Column(String, nullable=True)
    webhook_secret = Column(String, nullable=True)
    webhook_uuid = Column(String, nullable=True)
    # Use Text for tests (SQLite compatibility) and ARRAY for production (PostgreSQL)
    scopes = Column(Text if 'pytest' in sys.modules else ARRAY(Text), nullable=True)
    last_synced = Column(
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
