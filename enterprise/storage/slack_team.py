from datetime import datetime

from sqlalchemy import DateTime, Identity, String, text
from sqlalchemy.orm import Mapped, mapped_column
from storage.base import Base


class SlackTeam(Base):
    __tablename__ = 'slack_teams'

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    team_id: Mapped[str] = mapped_column(
        String, nullable=False, index=True, unique=True
    )
    bot_access_token: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=text('CURRENT_TIMESTAMP'),
        nullable=False,
    )
