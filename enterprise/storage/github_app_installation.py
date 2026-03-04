from sqlalchemy import Column, DateTime, Integer, String, text
from storage.base import Base


class GithubAppInstallation(Base):  # type: ignore
    """Represents a Github App Installation with associated token.
    """

    __tablename__ = 'github_app_installations'
    id = Column(Integer, primary_key=True, autoincrement=True)
    installation_id = Column(String, nullable=False)
    encrypted_token = Column(String, nullable=False)
    created_at = Column(
        DateTime, server_default=text('CURRENT_TIMESTAMP'), nullable=False
    )
    updated_at = Column(
        DateTime,
        server_default=text('CURRENT_TIMESTAMP'),
        onupdate=text('CURRENT_TIMESTAMP'),
        nullable=False,
    )
