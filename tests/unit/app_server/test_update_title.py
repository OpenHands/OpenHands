"""Tests for the column-specific title update."""

from typing import AsyncGenerator
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from openhands.app_server.app_conversation.sql_app_conversation_info_service import (
    SQLAppConversationInfoService,
    StoredConversationMetadata,
)
from openhands.app_server.user.specifiy_user_context import SpecifyUserContext
from openhands.app_server.utils.sql_utils import Base


@pytest.fixture
async def async_session() -> AsyncGenerator[AsyncSession, None]:
    engine = create_async_engine(
        'sqlite+aiosqlite:///:memory:',
        poolclass=StaticPool,
        connect_args={'check_same_thread': False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        yield session
    await engine.dispose()


@pytest.mark.asyncio
async def test_update_title_leaves_other_columns_alone(async_session):
    conversation_id = uuid4()
    stored = StoredConversationMetadata(
        conversation_id=str(conversation_id),
        conversation_version='V1',
        title='stub',
        llm_model='litellm_proxy/gpt-5.5',
        accumulated_cost=0.25,
        prompt_tokens=300,
        completion_tokens=30,
    )
    async_session.add(stored)
    await async_session.commit()

    service = SQLAppConversationInfoService(
        db_session=async_session, user_context=SpecifyUserContext(user_id=None)
    )
    await service.update_title(conversation_id, 'Generated Title')

    await async_session.refresh(stored)
    assert stored.title == 'Generated Title'
    assert stored.llm_model == 'litellm_proxy/gpt-5.5'
    assert stored.accumulated_cost == pytest.approx(0.25)
    assert stored.prompt_tokens == 300
    assert stored.completion_tokens == 30


@pytest.mark.asyncio
async def test_update_title_missing_conversation_is_noop(async_session):
    service = SQLAppConversationInfoService(
        db_session=async_session, user_context=SpecifyUserContext(user_id=None)
    )
    await service.update_title(uuid4(), 'whatever')
