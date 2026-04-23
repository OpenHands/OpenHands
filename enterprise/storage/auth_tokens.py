from sqlalchemy import BigInteger, Identity, Index, String
from sqlalchemy.orm import Mapped, mapped_column
from storage.base import Base


class AuthTokens(Base):
    __tablename__ = 'auth_tokens'

    id: Mapped[int] = mapped_column(Identity(), primary_key=True)
    keycloak_user_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    identity_provider: Mapped[str] = mapped_column(String, nullable=False)
    access_token: Mapped[str] = mapped_column(String, nullable=False)
    refresh_token: Mapped[str] = mapped_column(String, nullable=False)
    access_token_expires_at: Mapped[int] = mapped_column(
        BigInteger, nullable=False
    )  # Time since epoch in seconds
    refresh_token_expires_at: Mapped[int] = mapped_column(
        BigInteger, nullable=False
    )  # Time since epoch in seconds

    __table_args__ = (
        Index(
            'idx_auth_tokens_keycloak_user_identity_provider',
            'keycloak_user_id',
            'identity_provider',
            unique=True,
        ),
    )
