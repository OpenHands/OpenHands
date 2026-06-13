from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from server.maintenance_task_processor.conversation_expiration_processor import (
    ConversationExpirationProcessor,
)
from sqlalchemy import select
from storage.maintenance_task import MaintenanceTask
from storage.org import Org
from storage.stored_conversation_metadata import StoredConversationMetadata
from storage.stored_conversation_metadata_saas import StoredConversationMetadataSaas

from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationStartRequest,
    AppConversationStartTaskStatus,
)
from openhands.app_server.app_conversation.sql_app_conversation_start_task_service import (
    StoredAppConversationStartTask,
)


@asynccontextmanager
async def _session_context(async_session_maker):
    async with async_session_maker() as session:
        yield session


@pytest.mark.asyncio
async def test_processor_deletes_only_expired_conversations_for_org_policy(
    async_session_maker, monkeypatch
):
    processor = ConversationExpirationProcessor()
    now = datetime.now(timezone.utc)
    expired_conversation_id = uuid4()
    active_conversation_id = uuid4()
    disabled_policy_conversation_id = uuid4()
    org_with_policy = uuid4()
    org_without_policy = uuid4()
    user_id = uuid4()

    async with async_session_maker() as session:
        session.add_all(
            [
                Org(
                    id=org_with_policy,
                    name='retention-enabled',
                    conversation_expiration=30,
                ),
                Org(
                    id=org_without_policy,
                    name='retention-disabled',
                    conversation_expiration=None,
                ),
                StoredConversationMetadata(
                    conversation_id=str(expired_conversation_id),
                    conversation_version='V1',
                    last_updated_at=now - timedelta(days=31),
                    created_at=now - timedelta(days=40),
                    title='expired',
                ),
                StoredConversationMetadataSaas(
                    conversation_id=str(expired_conversation_id),
                    user_id=user_id,
                    org_id=org_with_policy,
                ),
                StoredAppConversationStartTask(
                    id=uuid4(),
                    created_by_user_id=str(user_id),
                    status=AppConversationStartTaskStatus.WORKING,
                    app_conversation_id=expired_conversation_id,
                    request=AppConversationStartRequest(),
                ),
                StoredConversationMetadata(
                    conversation_id=str(active_conversation_id),
                    conversation_version='V1',
                    last_updated_at=now - timedelta(days=29),
                    created_at=now - timedelta(days=40),
                    title='active',
                ),
                StoredConversationMetadataSaas(
                    conversation_id=str(active_conversation_id),
                    user_id=user_id,
                    org_id=org_with_policy,
                ),
                StoredAppConversationStartTask(
                    id=uuid4(),
                    created_by_user_id=str(user_id),
                    status=AppConversationStartTaskStatus.WORKING,
                    app_conversation_id=active_conversation_id,
                    request=AppConversationStartRequest(),
                ),
                StoredConversationMetadata(
                    conversation_id=str(disabled_policy_conversation_id),
                    conversation_version='V1',
                    last_updated_at=now - timedelta(days=365),
                    created_at=now - timedelta(days=400),
                    title='disabled-policy',
                ),
                StoredConversationMetadataSaas(
                    conversation_id=str(disabled_policy_conversation_id),
                    user_id=user_id,
                    org_id=org_without_policy,
                ),
            ]
        )
        await session.commit()

    monkeypatch.setattr(
        'server.maintenance_task_processor.conversation_expiration_processor.a_session_maker',
        lambda: _session_context(async_session_maker),
    )

    result = await processor(MaintenanceTask(id=1))

    assert result == {
        'orgs_processed': 1,
        'deleted_conversations': 1,
        'deleted_start_tasks': 1,
    }

    async with async_session_maker() as session:
        remaining_conversation_ids = set(
            await session.scalars(select(StoredConversationMetadata.conversation_id))
        )
        remaining_saas_ids = set(
            await session.scalars(
                select(StoredConversationMetadataSaas.conversation_id)
            )
        )
        remaining_start_task_ids = set(
            await session.scalars(
                select(StoredAppConversationStartTask.app_conversation_id)
            )
        )

    assert str(expired_conversation_id) not in remaining_conversation_ids
    assert str(expired_conversation_id) not in remaining_saas_ids
    assert expired_conversation_id not in remaining_start_task_ids
    assert str(active_conversation_id) in remaining_conversation_ids
    assert str(disabled_policy_conversation_id) in remaining_conversation_ids


@pytest.mark.asyncio
async def test_processor_ignores_orgs_with_non_positive_expiration(
    async_session_maker, monkeypatch
):
    processor = ConversationExpirationProcessor()

    async with async_session_maker() as session:
        session.add_all(
            [
                Org(id=uuid4(), name='zero-retention', conversation_expiration=0),
                Org(id=uuid4(), name='negative-retention', conversation_expiration=-1),
            ]
        )
        await session.commit()

    monkeypatch.setattr(
        'server.maintenance_task_processor.conversation_expiration_processor.a_session_maker',
        lambda: _session_context(async_session_maker),
    )

    result = await processor(MaintenanceTask(id=1))

    assert result == {
        'orgs_processed': 0,
        'deleted_conversations': 0,
        'deleted_start_tasks': 0,
    }


@pytest.mark.asyncio
async def test_processor_ignores_legacy_non_v1_conversations(
    async_session_maker, monkeypatch
):
    processor = ConversationExpirationProcessor()
    now = datetime.now(timezone.utc)
    org_id = uuid4()
    user_id = uuid4()

    async with async_session_maker() as session:
        session.add_all(
            [
                Org(
                    id=org_id,
                    name='retention-enabled',
                    conversation_expiration=30,
                ),
                StoredConversationMetadata(
                    conversation_id='legacy-conversation-id',
                    conversation_version='V0',
                    last_updated_at=now - timedelta(days=365),
                    created_at=now - timedelta(days=400),
                    title='legacy',
                ),
                StoredConversationMetadataSaas(
                    conversation_id='legacy-conversation-id',
                    user_id=user_id,
                    org_id=org_id,
                ),
            ]
        )
        await session.commit()

    monkeypatch.setattr(
        'server.maintenance_task_processor.conversation_expiration_processor.a_session_maker',
        lambda: _session_context(async_session_maker),
    )

    result = await processor(MaintenanceTask(id=1))

    assert result == {
        'orgs_processed': 1,
        'deleted_conversations': 0,
        'deleted_start_tasks': 0,
    }
    async with async_session_maker() as session:
        remaining_conversation_ids = set(
            await session.scalars(select(StoredConversationMetadata.conversation_id))
        )

    assert 'legacy-conversation-id' in remaining_conversation_ids
