from integrations.types import PRStatus
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Identity,
    Integer,
    String,
    text,
)
from storage.base import Base


class OpenhandsPR(Base):  # type: ignore
    """Represents a pull request created by OpenHands.
    """

    __tablename__ = 'openhands_prs'
    id = Column(Integer, Identity(), primary_key=True)
    repo_id = Column(String, nullable=False, index=True)
    repo_name = Column(String, nullable=False)
    pr_number = Column(Integer, nullable=False, index=True)
    status = Column(
        Enum(PRStatus),
        nullable=False,
        index=True,
    )
    provider = Column(String, nullable=False)
    installation_id = Column(String, nullable=True)
    private = Column(Boolean, nullable=True)

    # PR metrics columns (optional fields as all providers may not include this information, and will require post processing to enrich)
    num_reviewers = Column(Integer, nullable=True)
    num_commits = Column(Integer, nullable=True)
    num_review_comments = Column(Integer, nullable=True)
    num_general_comments = Column(Integer, nullable=True)
    num_changed_files = Column(Integer, nullable=True)
    num_additions = Column(Integer, nullable=True)
    num_deletions = Column(Integer, nullable=True)
    merged = Column(Boolean, nullable=True)

    # Fields that will definitely require post processing to enrich
    openhands_helped_author = Column(Boolean, nullable=True)
    num_openhands_commits = Column(Integer, nullable=True)
    num_openhands_review_comments = Column(Integer, nullable=True)
    num_openhands_general_comments = Column(Integer, nullable=True)

    # Attributes to track progress on enrichment
    processed = Column(Boolean, nullable=False, server_default=text('FALSE'))
    process_attempts = Column(
        Integer, nullable=False, server_default=text('0')
    )  # Max attempts in case we hit rate limits or information is no longer accessible
    updated_at = Column(
        DateTime,
        server_default=text('CURRENT_TIMESTAMP'),
        nullable=False,
    )  # To buffer between attempts
    closed_at = Column(
        DateTime,
        nullable=False,
    )  # Timestamp when the PR was closed
    created_at = Column(
        DateTime,
        nullable=False,
    )  # Timestamp when the PR was created
