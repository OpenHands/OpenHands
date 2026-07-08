"""Race-safety tests for SaaS conversation metadata saves."""

import logging
from typing import AsyncGenerator
from uuid import UUID, uuid4

import pytest
from server.utils.saas_app_conversation_info_injector import (
    SaasSQLAppConversationInfoService,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool
from storage.base import Base
from storage.org import Org
from storage.stored_conversation_metadata_saas import StoredConversationMetadataSaas
from storage.user import User

from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationInfo,
)
from openhands.app_server.user.specifiy_user_context import SpecifyUserContext

USER1_ID = UUID("a1111111-1111-1111-1111-111111111111")
USER2_ID = UUID("b2222222-2222-2222-2222-222222222222")
ORG1_ID = UUID("c1111111-1111-1111-1111-111111111111")
ORG2_ID = UUID("d2222222-2222-2222-2222-222222222222")


@pytest.fixture
async def async_engine():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=False,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def async_session_with_users(async_engine) -> AsyncGenerator[AsyncSession, None]:
    async_session_maker = async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session_maker() as db_session:
        db_session.add_all(
            [
                Org(
                    id=ORG1_ID,
                    name="test-org-1",
                    enable_proactive_conversation_starters=True,
                ),
                Org(
                    id=ORG2_ID,
                    name="test-org-2",
                    enable_proactive_conversation_starters=True,
                ),
            ]
        )
        await db_session.flush()
        db_session.add_all(
            [
                User(id=USER1_ID, current_org_id=ORG1_ID),
                User(id=USER2_ID, current_org_id=ORG2_ID),
            ]
        )
        await db_session.commit()
        yield db_session


@pytest.mark.asyncio
async def test_existing_saas_metadata_with_different_user_is_preserved(
    async_session_with_users: AsyncSession,
    caplog: pytest.LogCaptureFixture,
):
    user1_service = SaasSQLAppConversationInfoService(
        db_session=async_session_with_users,
        user_context=SpecifyUserContext(user_id=str(USER1_ID)),
    )

    conv_id = uuid4()
    await user1_service.save_app_conversation_info(
        AppConversationInfo(
            id=conv_id,
            created_by_user_id=str(USER1_ID),
            sandbox_id="sandbox_owner_race",
            title="Original owner conversation",
        )
    )

    user2_service = SaasSQLAppConversationInfoService(
        db_session=async_session_with_users,
        user_context=SpecifyUserContext(user_id=str(USER2_ID)),
    )
    with caplog.at_level(logging.WARNING):
        await user2_service.save_app_conversation_info(
            AppConversationInfo(
                id=conv_id,
                created_by_user_id=str(USER2_ID),
                sandbox_id="sandbox_owner_race",
                title="Conflicting owner conversation",
            )
        )

    saas_query = select(StoredConversationMetadataSaas).where(
        StoredConversationMetadataSaas.conversation_id == str(conv_id)
    )
    result = await async_session_with_users.execute(saas_query)
    saas_metadata = result.scalar_one_or_none()

    assert saas_metadata is not None
    assert saas_metadata.user_id == USER1_ID
    assert saas_metadata.org_id == ORG1_ID
    assert "Ignoring conflicting SaaS conversation metadata owner" in caplog.text
