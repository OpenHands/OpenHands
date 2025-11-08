import sys

from sqlalchemy import ARRAY, Boolean, Column, DateTime, Integer, String, Text, text
from storage.base import Base


class AzureDevOpsWebhook(Base):  # type: ignore
    """Represents an Azure DevOps webhook configuration for a project or repository."""

    __tablename__ = 'azure_devops_webhook'
    id = Column(Integer, primary_key=True, autoincrement=True)
    organization = Column(String, nullable=False)
    project_id = Column(String, nullable=False)  # Azure DevOps project ID
    repository_id = Column(String, nullable=True)  # NULL for project-level webhooks
    subscription_id = Column(
        String, nullable=True
    )  # Azure DevOps Service Hook subscription ID (native identifier)
    user_id = Column(String, nullable=False)
    webhook_exists = Column(Boolean, nullable=False)
    webhook_url = Column(String, nullable=True)
    webhook_secret = Column(String, nullable=True)  # Secret for webhook authentication
    # Use Text for tests (SQLite compatibility) and ARRAY for production (PostgreSQL)
    scopes = Column(Text if 'pytest' in sys.modules else ARRAY(Text), nullable=True)
    last_synced = Column(
        DateTime,
        server_default=text('CURRENT_TIMESTAMP'),
        onupdate=text('CURRENT_TIMESTAMP'),
        nullable=True,
    )

    def __repr__(self) -> str:
        return (
            f'<AzureDevOpsWebhook(id={self.id}, organization={self.organization}, '
            f'project_id={self.project_id}, repository_id={self.repository_id}, '
            f'last_synced={self.last_synced})>'
        )
