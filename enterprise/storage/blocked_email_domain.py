from datetime import UTC, datetime

from sqlalchemy import Column, DateTime, Identity, Integer, String
from storage.base import Base


class BlockedEmailDomain(Base):  # type: ignore
    """Stores blocked email domain patterns.

    Supports blocking:
    - Exact domains: 'example.com' blocks 'user@example.com'
    - Subdomains: 'example.com' blocks 'user@subdomain.example.com'
    - TLDs: '.us' blocks 'user@company.us' and 'user@subdomain.company.us'
    """

    __tablename__ = 'blocked_email_domains'

    id = Column(Integer, Identity(), primary_key=True)
    domain = Column(String, nullable=False, unique=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )
