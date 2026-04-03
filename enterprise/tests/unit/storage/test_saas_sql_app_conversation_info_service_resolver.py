"""Tests for SaasSQLAppConversationInfoService resolver_org_id routing.

Tests that when user_context has resolver_org_id set, it overrides
the default org_id when saving conversation SAAS metadata.
"""

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
from openhands.storage.data_models.conversation_metadata import ConversationTrigger

# Test UUIDs
USER_ID = UUID('a1111111-1111-1111-1111-111111111111')
DEFAULT_ORG_ID = UUID('c1111111-1111-1111-1111-111111111111')
RESOLVER_ORG_ID = UUID('d2222222-2222-2222-2222-222222222222')


@pytest.fixture
async def async_engine():
    """Create an async SQLite engine for testing."""
    engine = create_async_engine(
        'sqlite+aiosqlite:///:memory:',
        poolclass=StaticPool,
        connect_args={'check_same_thread': False},
        echo=False,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def async_session_with_users(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create an async session with pre-populated Org and User rows."""
    async_session_maker = async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session_maker() as db_session:
        # Create both orgs
        db_session.add(
            Org(
                id=DEFAULT_ORG_ID,
                name='default-org',
                enable_default_condenser=True,
                enable_proactive_conversation_starters=True,
            )
        )
        db_session.add(
            Org(
                id=RESOLVER_ORG_ID,
                name='resolver-org',
                enable_default_condenser=True,
                enable_proactive_conversation_starters=True,
            )
        )
        await db_session.flush()

        db_session.add(User(id=USER_ID, current_org_id=DEFAULT_ORG_ID))
        await db_session.commit()
        yield db_session


class _ResolverUserContext:
    """Minimal user context that mimics ResolverUserContext for testing."""

    def __init__(self, user_id: str, resolver_org_id: UUID | None = None):
        self._user_id = user_id
        self.resolver_org_id = resolver_org_id

    async def get_user_id(self) -> str | None:
        return self._user_id


class TestResolverOrgIdOverride:
    """Test that resolver_org_id overrides default org in save_app_conversation_info."""

    @pytest.mark.asyncio
    async def test_save_uses_resolver_org_id(self, async_session_with_users):
        """When resolver_org_id is set, conversation should be saved under that org."""
        context = _ResolverUserContext(
            user_id=str(USER_ID),
            resolver_org_id=RESOLVER_ORG_ID,
        )
        service = SaasSQLAppConversationInfoService(
            db_session=async_session_with_users,
            user_context=context,
        )

        conv_info = AppConversationInfo(
            id=uuid4(),
            created_by_user_id=str(USER_ID),
            sandbox_id='sandbox_resolver',
            title='Resolver Conversation',
            trigger=ConversationTrigger.SLACK,
        )
        await service.save_app_conversation_info(conv_info)

        # Verify SAAS metadata uses resolver org
        result = await async_session_with_users.execute(
            select(StoredConversationMetadataSaas).where(
                StoredConversationMetadataSaas.conversation_id == str(conv_info.id)
            )
        )
        saas_metadata = result.scalar_one_or_none()
        assert saas_metadata is not None
        assert saas_metadata.org_id == RESOLVER_ORG_ID
        assert saas_metadata.user_id == USER_ID

    @pytest.mark.asyncio
    async def test_save_uses_default_org_when_no_resolver(
        self, async_session_with_users
    ):
        """When resolver_org_id is None, the user's current_org_id should be used."""
        context = _ResolverUserContext(
            user_id=str(USER_ID),
            resolver_org_id=None,
        )
        service = SaasSQLAppConversationInfoService(
            db_session=async_session_with_users,
            user_context=context,
        )

        conv_info = AppConversationInfo(
            id=uuid4(),
            created_by_user_id=str(USER_ID),
            sandbox_id='sandbox_default',
            title='Default Conversation',
            trigger=ConversationTrigger.SLACK,
        )
        await service.save_app_conversation_info(conv_info)

        # Verify SAAS metadata uses default org
        result = await async_session_with_users.execute(
            select(StoredConversationMetadataSaas).where(
                StoredConversationMetadataSaas.conversation_id == str(conv_info.id)
            )
        )
        saas_metadata = result.scalar_one_or_none()
        assert saas_metadata is not None
        assert saas_metadata.org_id == DEFAULT_ORG_ID

    @pytest.mark.asyncio
    async def test_save_without_resolver_attr_uses_default(
        self, async_session_with_users
    ):
        """When user_context lacks resolver_org_id attribute, default org is used."""
        from openhands.app_server.user.specifiy_user_context import SpecifyUserContext

        context = SpecifyUserContext(user_id=str(USER_ID))
        service = SaasSQLAppConversationInfoService(
            db_session=async_session_with_users,
            user_context=context,
        )

        conv_info = AppConversationInfo(
            id=uuid4(),
            created_by_user_id=str(USER_ID),
            sandbox_id='sandbox_no_attr',
            title='No Attr Conversation',
            trigger=ConversationTrigger.GUI,
        )
        await service.save_app_conversation_info(conv_info)

        # Verify SAAS metadata uses default org
        result = await async_session_with_users.execute(
            select(StoredConversationMetadataSaas).where(
                StoredConversationMetadataSaas.conversation_id == str(conv_info.id)
            )
        )
        saas_metadata = result.scalar_one_or_none()
        assert saas_metadata is not None
        assert saas_metadata.org_id == DEFAULT_ORG_ID
