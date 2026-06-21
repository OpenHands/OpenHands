from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Index, String, text
from sqlalchemy.orm import Mapped, mapped_column
from storage.base import Base


class JiraDcUser(Base):
    __tablename__ = 'jira_dc_users'
    __table_args__ = (
        Index(
            'uq_jira_dc_users_one_active_per_user',
            'keycloak_user_id',
            unique=True,
            postgresql_where=text("status = 'active'"),
            sqlite_where=text("status = 'active'"),
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    keycloak_user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    jira_dc_user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    jira_dc_workspace_id: Mapped[int] = mapped_column(nullable=False, index=True)
    status: Mapped[str] = mapped_column(String, nullable=False)
    oauth_access_token_encrypted: Mapped[str | None] = mapped_column(
        String, nullable=True
    )
    oauth_refresh_token_encrypted: Mapped[str | None] = mapped_column(
        String, nullable=True
    )
    # Epoch seconds; 0 = unknown / no expiry info supplied by the IdP.
    oauth_access_token_expires_at: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True
    )
    oauth_refresh_token_expires_at: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True
    )
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
