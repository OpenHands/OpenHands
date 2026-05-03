from datetime import datetime

from sqlalchemy import (
    DateTime,
    String,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column
from storage.base import Base


class BitbucketWebhook(Base):
    """A Bitbucket Cloud webhook installed against a workspace or repository.

    Each row stores the per-installation ``webhook_uuid`` and the shared
    ``webhook_secret`` used to verify the ``X-Hub-Signature`` header on
    incoming events. The ``user_id`` is the keycloak id of the OpenHands
    user that registered the hook.
    """

    __tablename__ = 'bitbucket_webhook'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    workspace: Mapped[str] = mapped_column(String, nullable=False)
    repo_slug: Mapped[str | None] = mapped_column(String, nullable=True)
    user_id: Mapped[str] = mapped_column(String, nullable=False)
    webhook_uuid: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    webhook_secret: Mapped[str | None] = mapped_column(String, nullable=True)
    last_synced: Mapped[datetime | None] = mapped_column(
        DateTime,
        server_default=text('CURRENT_TIMESTAMP'),
        onupdate=text('CURRENT_TIMESTAMP'),
        nullable=True,
    )

    def __repr__(self) -> str:
        return (
            f'<BitbucketWebhook(id={self.id}, workspace={self.workspace}, '
            f'repo_slug={self.repo_slug}, webhook_uuid={self.webhook_uuid})>'
        )
