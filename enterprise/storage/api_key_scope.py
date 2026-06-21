from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from storage.base import Base

if TYPE_CHECKING:
    from storage.api_key import ApiKey


class ApiKeyScope(Base):
    """Represents an explicit scope for an API key."""

    __tablename__ = "api_key_scopes"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    api_key_id: Mapped[int] = mapped_column(
        ForeignKey("api_keys.id", ondelete="CASCADE"), nullable=False, index=True
    )
    scope: Mapped[str] = mapped_column(String(255), nullable=False)

    # Relationships
    api_key: Mapped["ApiKey"] = relationship("ApiKey", back_populates="scopes")
