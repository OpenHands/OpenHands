"""Tests for SaasConversationStore resolver_org_id routing.

Tests that when resolver_org_id is provided, it overrides the default org_id
when saving conversation metadata to the SaaS metadata table.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest
from storage.saas_conversation_store import SaasConversationStore
from storage.stored_conversation_metadata_saas import StoredConversationMetadataSaas
from storage.user import User

from openhands.storage.data_models.conversation_metadata import ConversationMetadata

USER_ID = '5594c7b6-f959-4b81-92e9-b09c206f5081'
DEFAULT_ORG_ID = UUID('5594c7b6-f959-4b81-92e9-b09c206f5081')
RESOLVER_ORG_ID = UUID('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa')


@pytest.fixture(autouse=True)
def mock_call_sync_from_async():
    """Replace call_sync_from_async with a direct call"""

    def _direct_call(func):
        return func()

    with patch(
        'storage.saas_conversation_store.call_sync_from_async', side_effect=_direct_call
    ):
        yield


@pytest.fixture(autouse=True)
def mock_user_store():
    """Mock UserStore.get_user_by_id to return a mock user"""
    mock_user = MagicMock(spec=User)
    mock_user.current_org_id = DEFAULT_ORG_ID

    with patch('storage.user_store.UserStore.get_user_by_id', return_value=mock_user):
        yield


@pytest.mark.asyncio
async def test_save_metadata_uses_resolver_org_id(session_maker):
    """When resolver_org_id is set, it should override org_id in SaaS metadata."""
    store = SaasConversationStore(
        USER_ID,
        DEFAULT_ORG_ID,
        session_maker,
        resolver_org_id=RESOLVER_ORG_ID,
    )
    metadata = ConversationMetadata(
        conversation_id='resolver-conv-id',
        user_id=USER_ID,
        selected_repository='MyOrg/my-repo',
        selected_branch=None,
        created_at=datetime.now(UTC),
        last_updated_at=datetime.now(UTC),
    )
    await store.save_metadata(metadata)

    # Verify the SaaS metadata was saved with the resolver org_id
    with session_maker() as session:
        saas_metadata = (
            session.query(StoredConversationMetadataSaas)
            .filter(
                StoredConversationMetadataSaas.conversation_id == 'resolver-conv-id'
            )
            .first()
        )
        assert saas_metadata is not None
        assert saas_metadata.org_id == RESOLVER_ORG_ID
        assert saas_metadata.user_id == UUID(USER_ID)


@pytest.mark.asyncio
async def test_save_metadata_uses_default_org_when_no_resolver(session_maker):
    """When resolver_org_id is None, the default org_id should be used."""
    store = SaasConversationStore(
        USER_ID,
        DEFAULT_ORG_ID,
        session_maker,
        resolver_org_id=None,
    )
    metadata = ConversationMetadata(
        conversation_id='default-conv-id',
        user_id=USER_ID,
        selected_repository='MyOrg/my-repo',
        selected_branch=None,
        created_at=datetime.now(UTC),
        last_updated_at=datetime.now(UTC),
    )
    await store.save_metadata(metadata)

    # Verify the SaaS metadata was saved with the default org_id
    with session_maker() as session:
        saas_metadata = (
            session.query(StoredConversationMetadataSaas)
            .filter(StoredConversationMetadataSaas.conversation_id == 'default-conv-id')
            .first()
        )
        assert saas_metadata is not None
        assert saas_metadata.org_id == DEFAULT_ORG_ID


@pytest.mark.asyncio
async def test_get_instance_passes_resolver_org_id(session_maker):
    """get_instance should accept and pass resolver_org_id to constructor."""
    with patch('storage.saas_conversation_store.session_maker', session_maker):
        store = await SaasConversationStore.get_instance(
            MagicMock(),
            USER_ID,
            resolver_org_id=RESOLVER_ORG_ID,
        )
    assert store.resolver_org_id == RESOLVER_ORG_ID


@pytest.mark.asyncio
async def test_get_instance_default_no_resolver_org_id(session_maker):
    """get_instance without resolver_org_id should set it to None."""
    with patch('storage.saas_conversation_store.session_maker', session_maker):
        store = await SaasConversationStore.get_instance(
            MagicMock(),
            USER_ID,
        )
    assert store.resolver_org_id is None
