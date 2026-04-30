from datetime import datetime

from sqlalchemy import (
    DateTime,
    String,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column
from storage.base import Base


class BitbucketDCWebhook(Base):
    """A Bitbucket Data Center webhook installed against a single repository.

    Bitbucket DC does not expose a per-installation UUID in webhook
    request headers, so each row is identified by ``(project_key,
    repo_slug)`` and the secret + installer are looked up at delivery
    time using the repository identity carried in the payload.
    """

    __tablename__ = 'bitbucket_dc_webhook'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    project_key: Mapped[str] = mapped_column(String, nullable=False)
    repo_slug: Mapped[str] = mapped_column(String, nullable=False)
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    webhook_id: Mapped[str | None] = mapped_column(String, nullable=True)
    webhook_secret: Mapped[str | None] = mapped_column(String, nullable=True)
    last_synced: Mapped[datetime | None] = mapped_column(
        DateTime,
        server_default=text('CURRENT_TIMESTAMP'),
        onupdate=text('CURRENT_TIMESTAMP'),
        nullable=True,
    )

    def __repr__(self) -> str:
        return (
            f'<BitbucketDCWebhook(id={self.id}, project_key={self.project_key}, '
            f'repo_slug={self.repo_slug}, webhook_id={self.webhook_id})>'
        )
