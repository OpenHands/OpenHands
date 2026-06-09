from datetime import datetime

from sqlalchemy import DateTime, String, text
from sqlalchemy.orm import Mapped, mapped_column
from storage.base import Base


class StoredOfflineToken(Base):
    __tablename__ = 'offline_tokens'

    user_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    offline_token: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=text('CURRENT_TIMESTAMP'), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=text('CURRENT_TIMESTAMP'),
        onupdate=text('CURRENT_TIMESTAMP'),
        nullable=False,
    )
