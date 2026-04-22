"""Tests for conversation tags processing in webhook_router.

This module tests the tag persistence, merging, and automation trigger detection
functionality for conversations created via SDK/automations.
"""

from datetime import datetime, timezone
from typing import AsyncGenerator
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from openhands.app_server.app_conversation.sql_app_conversation_info_service import (
    SQLAppConversationInfoService,
    StoredConversationMetadata,
)
from openhands.app_server.event_callback.webhook_router import (
    detect_automation_trigger,
    merge_conversation_tags,
)
from openhands.app_server.user.specifiy_user_context import SpecifyUserContext
from openhands.app_server.utils.sql_utils import Base
from openhands.storage.data_models.conversation_metadata import ConversationTrigger

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def async_engine():
    """Create an async SQLite engine for testing."""
    engine = create_async_engine(
        'sqlite+aiosqlite:///:memory:',
        poolclass=StaticPool,
        connect_args={'check_same_thread': False},
        echo=False,
    )

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()


@pytest.fixture
async def async_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create an async session for testing."""
    async_session_maker = async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session_maker() as db_session:
        yield db_session


@pytest.fixture
def service(async_session) -> SQLAppConversationInfoService:
    """Create a SQLAppConversationInfoService instance for testing."""
    return SQLAppConversationInfoService(
        db_session=async_session, user_context=SpecifyUserContext(user_id=None)
    )


# ---------------------------------------------------------------------------
# Tag Persistence Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_conversation_with_tags(async_session, service):
    """Save a conversation with tags, load it back, verify tags match."""
    conversation_id = uuid4()
    tags = {'trigger': 'automation', 'automation_id': 'auto-123', 'run_id': 'run-456'}

    # Create and save conversation with tags
    stored = StoredConversationMetadata(
        conversation_id=str(conversation_id),
        sandbox_id='sandbox_123',
        title='Test Conversation with Tags',
        tags=tags,
        conversation_version='V1',
        pr_number=[],
        created_at=datetime.now(timezone.utc),
        last_updated_at=datetime.now(timezone.utc),
    )
    async_session.add(stored)
    await async_session.commit()

    # Load it back
    result = await service.get_app_conversation_info(conversation_id)

    assert result is not None
    assert result.tags == tags
    assert result.tags['trigger'] == 'automation'
    assert result.tags['automation_id'] == 'auto-123'
    assert result.tags['run_id'] == 'run-456'


@pytest.mark.asyncio
async def test_save_conversation_with_empty_tags(async_session, service):
    """Save a conversation with empty tags dict."""
    conversation_id = uuid4()

    stored = StoredConversationMetadata(
        conversation_id=str(conversation_id),
        sandbox_id='sandbox_123',
        title='Test Conversation',
        tags={},
        conversation_version='V1',
        pr_number=[],
        created_at=datetime.now(timezone.utc),
        last_updated_at=datetime.now(timezone.utc),
    )
    async_session.add(stored)
    await async_session.commit()

    result = await service.get_app_conversation_info(conversation_id)

    assert result is not None
    assert result.tags == {}


@pytest.mark.asyncio
async def test_save_conversation_with_none_tags(async_session, service):
    """Save a conversation with None tags (should default to empty dict)."""
    conversation_id = uuid4()

    stored = StoredConversationMetadata(
        conversation_id=str(conversation_id),
        sandbox_id='sandbox_123',
        title='Test Conversation',
        tags=None,
        conversation_version='V1',
        pr_number=[],
        created_at=datetime.now(timezone.utc),
        last_updated_at=datetime.now(timezone.utc),
    )
    async_session.add(stored)
    await async_session.commit()

    result = await service.get_app_conversation_info(conversation_id)

    assert result is not None
    # None tags should be converted to empty dict on read
    assert result.tags == {}


# ---------------------------------------------------------------------------
# Tag Merging Tests
# ---------------------------------------------------------------------------


def test_merge_conversation_tags_incoming_overrides_existing():
    """Test that incoming tags override existing tags with same key."""
    existing_tags = {'key1': 'original', 'key2': 'keep_this'}
    incoming_tags = {'key1': 'updated', 'key3': 'new_value'}

    merged_tags = merge_conversation_tags(existing_tags, incoming_tags)

    assert merged_tags == {
        'key1': 'updated',  # Overridden
        'key2': 'keep_this',  # Preserved
        'key3': 'new_value',  # Added
    }


def test_merge_conversation_tags_with_empty_incoming():
    """Test that empty incoming tags preserves existing tags."""
    existing_tags = {'existing': 'value'}
    incoming_tags = {}

    merged_tags = merge_conversation_tags(existing_tags, incoming_tags)

    assert merged_tags == {'existing': 'value'}


def test_merge_conversation_tags_with_none_existing():
    """Test merging when existing tags is None."""
    existing_tags = None
    incoming_tags = {'new': 'value'}

    merged_tags = merge_conversation_tags(existing_tags, incoming_tags)

    assert merged_tags == {'new': 'value'}


def test_merge_conversation_tags_with_both_none():
    """Test merging when both are None."""
    merged_tags = merge_conversation_tags(None, None)

    assert merged_tags == {}


# ---------------------------------------------------------------------------
# Automation Trigger Detection Tests
# ---------------------------------------------------------------------------


def test_detect_automation_trigger_by_trigger_tag():
    """Test that 'automationtrigger' tag triggers AUTOMATION detection."""
    merged_tags = {'automationtrigger': 'cron', 'automationid': 'auto-123'}

    trigger = detect_automation_trigger(None, merged_tags)

    assert trigger == ConversationTrigger.AUTOMATION


def test_detect_automation_trigger_by_automation_id():
    """Test that 'automationid' tag alone triggers AUTOMATION detection."""
    merged_tags = {'automationid': 'auto-123'}

    trigger = detect_automation_trigger(None, merged_tags)

    assert trigger == ConversationTrigger.AUTOMATION


def test_detect_automation_trigger_by_automation_run_id():
    """Test that 'automationrunid' tag alone triggers AUTOMATION detection."""
    merged_tags = {'automationrunid': 'run-123'}

    trigger = detect_automation_trigger(None, merged_tags)

    assert trigger == ConversationTrigger.AUTOMATION


def test_detect_automation_trigger_not_set_without_relevant_tags():
    """Test that trigger is not set without automation-related tags."""
    merged_tags = {'some_other_key': 'value'}

    trigger = detect_automation_trigger(None, merged_tags)

    assert trigger is None


def test_detect_automation_trigger_not_overridden_if_already_set():
    """Test that existing trigger is not overridden."""
    merged_tags = {'automationtrigger': 'cron', 'automationid': 'auto-123'}

    # Trigger already set (e.g., from previous update)
    trigger = detect_automation_trigger(ConversationTrigger.GUI, merged_tags)

    # Should remain GUI, not AUTOMATION
    assert trigger == ConversationTrigger.GUI


def test_detect_automation_trigger_with_empty_tags():
    """Test that empty tags don't set trigger."""
    trigger = detect_automation_trigger(None, {})

    assert trigger is None


# ---------------------------------------------------------------------------
# Integration-style Tests (Save + Load + Verify)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_tag_roundtrip_with_automation_context(async_session, service):
    """Full integration test: save tags, load, verify, update, verify again."""
    conversation_id = uuid4()

    # Initial save with automation tags
    initial_tags = {
        'trigger': 'webhook',
        'automation_id': 'auto-abc',
        'automation_name': 'Daily Report',
        'run_id': 'run-xyz',
        'plugins': 'https://github.com/OpenHands/skill1,https://github.com/OpenHands/skill2',
    }

    stored = StoredConversationMetadata(
        conversation_id=str(conversation_id),
        sandbox_id='sandbox_123',
        title='Automation Conversation',
        tags=initial_tags,
        trigger='automation',
        conversation_version='V1',
        pr_number=[],
        created_at=datetime.now(timezone.utc),
        last_updated_at=datetime.now(timezone.utc),
    )
    async_session.add(stored)
    await async_session.commit()

    # Load and verify
    result = await service.get_app_conversation_info(conversation_id)
    assert result is not None
    assert result.tags == initial_tags
    assert result.tags['trigger'] == 'webhook'
    assert result.tags['automation_id'] == 'auto-abc'
    assert 'skill1' in result.tags['plugins']
    assert 'skill2' in result.tags['plugins']

    # Update with additional tag
    result.tags['custom_key'] = 'custom_value'
    stored.tags = result.tags
    await async_session.commit()

    # Reload and verify update
    result2 = await service.get_app_conversation_info(conversation_id)
    assert result2.tags['custom_key'] == 'custom_value'
    # Original tags should still be present
    assert result2.tags['automation_id'] == 'auto-abc'
