from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, Identity, String, text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from storage.base import Base

if TYPE_CHECKING:
    from storage.org import Org


class SlackUser(Base):
    __tablename__ = 'slack_users'

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    keycloak_user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    org_id: Mapped[UUID | None] = mapped_column(ForeignKey('org.id'), nullable=True)
    slack_user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    slack_display_name: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=text('CURRENT_TIMESTAMP'),
        nullable=False,
    )

    # Relationships
    org: Mapped['Org | None'] = relationship('Org', back_populates='slack_users')
