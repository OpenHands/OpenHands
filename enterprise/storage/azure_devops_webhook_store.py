from dataclasses import dataclass

from sqlalchemy import and_, select
from sqlalchemy.orm import sessionmaker
from storage.azure_devops_webhook import AzureDevOpsWebhook
from storage.database import a_session_maker


@dataclass
class AzureDevOpsWebhookStore:
    a_session_maker: sessionmaker = a_session_maker

    async def store_webhooks(self, webhook_details: list[AzureDevOpsWebhook]) -> None:
        """Store list of Azure DevOps webhook details in db using INSERT pattern.

        Args:
            webhook_details: List of AzureDevOpsWebhook objects to store

        Notes:
            1. Checks for existing records before inserting to avoid duplicates
            2. Leverages database-level partial unique indexes for uniqueness
            3. Performs the operation in a single database transaction
        """
        if not webhook_details:
            return

        async with self.a_session_maker() as session:
            async with session.begin():
                for webhook in webhook_details:
                    # Check if webhook already exists
                    if webhook.repository_id:
                        # Repository-level webhook
                        query = select(AzureDevOpsWebhook).where(
                            and_(
                                AzureDevOpsWebhook.organization == webhook.organization,
                                AzureDevOpsWebhook.project_id == webhook.project_id,
                                AzureDevOpsWebhook.repository_id
                                == webhook.repository_id,
                            )
                        )
                    else:
                        # Project-level webhook
                        query = select(AzureDevOpsWebhook).where(
                            and_(
                                AzureDevOpsWebhook.organization == webhook.organization,
                                AzureDevOpsWebhook.project_id == webhook.project_id,
                                AzureDevOpsWebhook.repository_id.is_(None),
                            )
                        )

                    result = await session.execute(query)
                    existing = result.scalar_one_or_none()

                    if existing is None:
                        # Insert new webhook
                        session.add(webhook)
                    # If exists, do nothing (you can add update logic here if needed)
