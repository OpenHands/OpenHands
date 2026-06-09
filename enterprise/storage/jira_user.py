from datetime import datetime

from sqlalchemy import DateTime, String, text
from sqlalchemy.orm import Mapped, mapped_column
from storage.base import Base


class JiraUser(Base):
    __tablename__ = 'jira_users'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    keycloak_user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    jira_user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    jira_workspace_id: Mapped[int] = mapped_column(nullable=False, index=True)
    status: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=text('CURRENT_TIMESTAMP'),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=text('CURRENT_TIMESTAMP'),
        onupdate=text('CURRENT_TIMESTAMP'),
        nullable=False,
    )
