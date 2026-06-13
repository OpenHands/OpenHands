from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import delete, select
from storage.database import a_session_maker
from storage.maintenance_task import MaintenanceTask, MaintenanceTaskProcessor
from storage.org import Org
from storage.stored_conversation_metadata import StoredConversationMetadata
from storage.stored_conversation_metadata_saas import StoredConversationMetadataSaas

from openhands.app_server.app_conversation.sql_app_conversation_start_task_service import (
    StoredAppConversationStartTask,
)


class ConversationExpirationProcessor(MaintenanceTaskProcessor):
    """Delete conversations older than each org's conversation_expiration setting."""

    async def __call__(self, task: MaintenanceTask) -> dict:
        orgs_processed = 0
        deleted_conversations = 0
        deleted_start_tasks = 0

        async with a_session_maker() as session:
            orgs = (
                await session.scalars(
                    select(Org).where(
                        Org.conversation_expiration.is_not(None),
                        Org.conversation_expiration > 0,
                    )
                )
            ).all()

            for org in orgs:
                orgs_processed += 1
                threshold = datetime.now(UTC) - timedelta(
                    days=org.conversation_expiration
                )
                expired_conversation_ids = list(
                    await session.scalars(
                        select(StoredConversationMetadata.conversation_id)
                        .join(
                            StoredConversationMetadataSaas,
                            StoredConversationMetadata.conversation_id
                            == StoredConversationMetadataSaas.conversation_id,
                        )
                        .where(
                            StoredConversationMetadataSaas.org_id == org.id,
                            StoredConversationMetadata.conversation_version == 'V1',
                            StoredConversationMetadata.last_updated_at < threshold,
                        )
                    )
                )

                if not expired_conversation_ids:
                    continue

                expired_conversation_uuids = [
                    UUID(conversation_id)
                    for conversation_id in expired_conversation_ids
                ]
                start_task_result = await session.execute(
                    delete(StoredAppConversationStartTask).where(
                        StoredAppConversationStartTask.app_conversation_id.in_(
                            expired_conversation_uuids
                        )
                    )
                )
                deleted_start_tasks += start_task_result.rowcount or 0

                await session.execute(
                    delete(StoredConversationMetadataSaas).where(
                        StoredConversationMetadataSaas.conversation_id.in_(
                            expired_conversation_ids
                        )
                    )
                )
                metadata_result = await session.execute(
                    delete(StoredConversationMetadata).where(
                        StoredConversationMetadata.conversation_id.in_(
                            expired_conversation_ids
                        )
                    )
                )
                deleted_conversations += metadata_result.rowcount or 0

            await session.commit()

        return {
            'orgs_processed': orgs_processed,
            'deleted_conversations': deleted_conversations,
            'deleted_start_tasks': deleted_start_tasks,
        }
