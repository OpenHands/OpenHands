"""Tests for public conversation models."""

from datetime import datetime
from uuid import uuid4

from openhands.app_server.sharing.public_conversation_models import (
    PublicConversation,
    PublicConversationPage,
    PublicConversationSortOrder,
)


def test_public_conversation_creation():
    """Test that PublicConversation can be created with all required fields."""
    conversation_id = uuid4()
    now = datetime.utcnow()

    conversation = PublicConversation(
        id=conversation_id,
        created_by_user_id='test_user',
        sandbox_id='test_sandbox',
        title='Test Conversation',
        created_at=now,
        updated_at=now,
        selected_repository=None,
        parent_conversation_id=None,
    )

    assert conversation.id == conversation_id
    assert conversation.title == 'Test Conversation'
    assert conversation.created_by_user_id == 'test_user'
    assert conversation.sandbox_id == 'test_sandbox'


def test_public_conversation_page_creation():
    """Test that PublicConversationPage can be created."""
    conversation_id = uuid4()
    now = datetime.utcnow()

    conversation = PublicConversation(
        id=conversation_id,
        created_by_user_id='test_user',
        sandbox_id='test_sandbox',
        title='Test Conversation',
        created_at=now,
        updated_at=now,
        selected_repository=None,
        parent_conversation_id=None,
    )

    page = PublicConversationPage(
        items=[conversation],
        next_page_id='next_page',
    )

    assert len(page.items) == 1
    assert page.items[0].id == conversation_id
    assert page.next_page_id == 'next_page'


def test_public_conversation_sort_order_enum():
    """Test that PublicConversationSortOrder enum has expected values."""
    assert hasattr(PublicConversationSortOrder, 'CREATED_AT')
    assert hasattr(PublicConversationSortOrder, 'CREATED_AT_DESC')
    assert hasattr(PublicConversationSortOrder, 'UPDATED_AT')
    assert hasattr(PublicConversationSortOrder, 'UPDATED_AT_DESC')
    assert hasattr(PublicConversationSortOrder, 'TITLE')
    assert hasattr(PublicConversationSortOrder, 'TITLE_DESC')


def test_public_conversation_optional_fields():
    """Test that PublicConversation works with optional fields."""
    conversation_id = uuid4()
    parent_id = uuid4()
    now = datetime.utcnow()

    conversation = PublicConversation(
        id=conversation_id,
        created_by_user_id='test_user',
        sandbox_id='test_sandbox',
        title='Test Conversation',
        created_at=now,
        updated_at=now,
        selected_repository='owner/repo',
        parent_conversation_id=parent_id,
        llm_model='gpt-4',
    )

    assert conversation.selected_repository == 'owner/repo'
    assert conversation.parent_conversation_id == parent_id
    assert conversation.llm_model == 'gpt-4'
