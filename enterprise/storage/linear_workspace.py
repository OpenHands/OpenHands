from datetime import datetime

from sqlalchemy import DateTime, String, text
from sqlalchemy.orm import Mapped, mapped_column
from storage.base import Base


class LinearWorkspace(Base):
    __tablename__ = 'linear_workspaces'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    linear_org_id: Mapped[str] = mapped_column(String, nullable=False)
    admin_user_id: Mapped[str] = mapped_column(String, nullable=False)
    webhook_secret: Mapped[str] = mapped_column(String, nullable=False)
    svc_acc_email: Mapped[str] = mapped_column(String, nullable=False)
    svc_acc_api_key: Mapped[str] = mapped_column(String, nullable=False)
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
