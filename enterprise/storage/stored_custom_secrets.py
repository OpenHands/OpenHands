from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import ForeignKey, Identity, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from storage.base import Base

if TYPE_CHECKING:
    from storage.org import Org


class StoredCustomSecrets(Base):
    __tablename__ = 'custom_secrets'

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    keycloak_user_id: Mapped[str | None] = mapped_column(
        String, nullable=True, index=True
    )
    org_id: Mapped[UUID | None] = mapped_column(ForeignKey('org.id'), nullable=True)
    secret_name: Mapped[str] = mapped_column(String, nullable=False)
    secret_value: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(String, nullable=True)

    # Relationships
    org: Mapped['Org | None'] = relationship('Org', back_populates='user_secrets')
