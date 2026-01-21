from sqlalchemy import Boolean, Column, DateTime, Integer, String, text
from storage.base import Base


class AzureDevOpsWebhook(Base):  # type: ignore
    """Represents an Azure DevOps webhook configuration for a project or repository."""

    __tablename__ = 'azure_devops_webhook'
    id = Column(Integer, primary_key=True, autoincrement=True)
    organization = Column(String, nullable=False)
    project_id = Column(String, nullable=False)  # Azure DevOps project ID
    repository_id = Column(String, nullable=True)  # NULL for project-level webhooks
    user_id = Column(String, nullable=False)
    webhook_exists = Column(Boolean, nullable=False)
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
