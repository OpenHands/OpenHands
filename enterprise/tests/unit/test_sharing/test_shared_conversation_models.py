"""Tests for public conversation models."""

from datetime import datetime
from uuid import uuid4

from server.sharing.shared_conversation_models import (
    SharedConversation,
)


def test_public_conversation_creation():
    """Test that SharedConversation can be created with all required fields."""
    conversation_id = uuid4()
    now = datetime.utcnow()

    conversation = SharedConversation(
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


def test_public_conversation_optional_fields():
    """Test that SharedConversation works with optional fields."""
    conversation_id = uuid4()
    parent_id = uuid4()
    now = datetime.utcnow()

    conversation = SharedConversation(
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
