from datetime import UTC, datetime
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest

from openhands.storage.data_models.conversation_metadata import ConversationMetadata

# Mock the database module before importing
with patch('storage.database.engine'), patch('storage.database.a_engine'):
    from storage.saas_conversation_store import SaasConversationStore
    from storage.user import User


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
    mock_user.current_org_id = UUID('5594c7b6-f959-4b81-92e9-b09c206f5081')

    with patch('storage.user_store.UserStore.get_user_by_id', return_value=mock_user):
        yield


@pytest.mark.asyncio
async def test_save_and_get(session_maker):
    store = SaasConversationStore('5594c7b6-f959-4b81-92e9-b09c206f5081', session_maker)
    metadata = ConversationMetadata(
        conversation_id='my-conversation-id',
        user_id='5594c7b6-f959-4b81-92e9-b09c206f5081',
        selected_repository='my-repo',
        selected_branch=None,
        created_at=datetime.now(UTC),
        last_updated_at=datetime.now(UTC),
        accumulated_cost=10.5,
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
    )
    await store.save_metadata(metadata)
    loaded = await store.get_metadata('my-conversation-id')
    assert loaded.conversation_id == metadata.conversation_id
    assert loaded.selected_repository == metadata.selected_repository
    assert loaded.accumulated_cost == metadata.accumulated_cost
    assert loaded.prompt_tokens == metadata.prompt_tokens
    assert loaded.completion_tokens == metadata.completion_tokens
    assert loaded.total_tokens == metadata.total_tokens


@pytest.mark.asyncio
async def test_search(session_maker):
    store = SaasConversationStore('5594c7b6-f959-4b81-92e9-b09c206f5081', session_maker)

    # Create test conversations with different timestamps
    conversations = [
        ConversationMetadata(
            conversation_id=f'conv-{i}',
            user_id='5594c7b6-f959-4b81-92e9-b09c206f5081',
            selected_repository='repo',
            selected_branch=None,
            created_at=datetime(2024, 1, i + 1, tzinfo=UTC),
            last_updated_at=datetime(2024, 1, i + 1, tzinfo=UTC),
        )
        for i in range(5)
    ]

    # Save conversations
    for conv in conversations:
        await store.save_metadata(conv)

    # Test basic search - should return all valid conversations sorted by created_at
    result = await store.search(limit=10)
    assert len(result.results) == 5
    assert [c.conversation_id for c in result.results] == [
        'conv-4',
        'conv-3',
        'conv-2',
        'conv-1',
        'conv-0',
    ]
    assert result.next_page_id is None

    # Test pagination
    result = await store.search(limit=2)
    assert len(result.results) == 2
    assert [c.conversation_id for c in result.results] == ['conv-4', 'conv-3']
    assert result.next_page_id is not None

    # Test next page
    result = await store.search(page_id=result.next_page_id, limit=2)
    assert len(result.results) == 2
    assert [c.conversation_id for c in result.results] == ['conv-2', 'conv-1']


@pytest.mark.asyncio
async def test_delete_metadata(session_maker):
    store = SaasConversationStore('5594c7b6-f959-4b81-92e9-b09c206f5081', session_maker)
    metadata = ConversationMetadata(
        conversation_id='to-delete',
        user_id='5594c7b6-f959-4b81-92e9-b09c206f5081',
        selected_repository='repo',
        selected_branch=None,
        created_at=datetime.now(UTC),
        last_updated_at=datetime.now(UTC),
    )
    await store.save_metadata(metadata)
    assert await store.exists('to-delete')

    await store.delete_metadata('to-delete')
    with pytest.raises(FileNotFoundError):
        await store.get_metadata('to-delete')
    assert not await store.exists('to-delete')


@pytest.mark.asyncio
async def test_get_nonexistent_metadata(session_maker):
    store = SaasConversationStore('5594c7b6-f959-4b81-92e9-b09c206f5081', session_maker)
    with pytest.raises(FileNotFoundError):
        await store.get_metadata('nonexistent-id')


@pytest.mark.asyncio
async def test_exists(session_maker):
    store = SaasConversationStore('5594c7b6-f959-4b81-92e9-b09c206f5081', session_maker)
    metadata = ConversationMetadata(
        conversation_id='exists-test',
        user_id='5594c7b6-f959-4b81-92e9-b09c206f5081',
        selected_repository='repo',
        selected_branch='test-branch',
        created_at=datetime.now(UTC),
        last_updated_at=datetime.now(UTC),
    )
    assert not await store.exists('exists-test')
    await store.save_metadata(metadata)
    assert await store.exists('exists-test')
