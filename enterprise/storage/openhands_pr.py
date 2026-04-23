from datetime import datetime

from integrations.types import PRStatus
from sqlalchemy import DateTime, Enum, Identity, String, text
from sqlalchemy.orm import Mapped, mapped_column
from storage.base import Base


class OpenhandsPR(Base):
    """
    Represents a pull request created by OpenHands.
    """

    __tablename__ = 'openhands_prs'

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    repo_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    repo_name: Mapped[str] = mapped_column(String, nullable=False)
    pr_number: Mapped[int] = mapped_column(nullable=False, index=True)
    status: Mapped[PRStatus] = mapped_column(Enum(PRStatus), nullable=False, index=True)
    provider: Mapped[str] = mapped_column(String, nullable=False)
    installation_id: Mapped[str | None] = mapped_column(String, nullable=True)
    private: Mapped[bool | None] = mapped_column(nullable=True)

    # PR metrics columns (optional fields as all providers may not include this
    # information, and will require post processing to enrich)
    num_reviewers: Mapped[int | None] = mapped_column(nullable=True)
    num_commits: Mapped[int | None] = mapped_column(nullable=True)
    num_review_comments: Mapped[int | None] = mapped_column(nullable=True)
    num_general_comments: Mapped[int | None] = mapped_column(nullable=True)
    num_changed_files: Mapped[int | None] = mapped_column(nullable=True)
    num_additions: Mapped[int | None] = mapped_column(nullable=True)
    num_deletions: Mapped[int | None] = mapped_column(nullable=True)
    merged: Mapped[bool | None] = mapped_column(nullable=True)

    # Fields that will definitely require post processing to enrich
    openhands_helped_author: Mapped[bool | None] = mapped_column(nullable=True)
    num_openhands_commits: Mapped[int | None] = mapped_column(nullable=True)
    num_openhands_review_comments: Mapped[int | None] = mapped_column(nullable=True)
    num_openhands_general_comments: Mapped[int | None] = mapped_column(nullable=True)

    # Attributes to track progress on enrichment
    processed: Mapped[bool] = mapped_column(
        nullable=False, server_default=text('FALSE')
    )
    process_attempts: Mapped[int] = mapped_column(
        nullable=False, server_default=text('0')
    )  # Max attempts in case we hit rate limits or information is no longer accessible
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=text('CURRENT_TIMESTAMP'), nullable=False
    )  # To buffer between attempts
    closed_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False
    )  # Timestamp when the PR was closed
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False
    )  # Timestamp when the PR was created
