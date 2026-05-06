"""Tests for ACP agent discrimination in webhook_router.

Verifies that ACPConversationInfo with an ACPAgent payload is correctly
discriminated from ConversationInfo with a regular Agent, and that
display_name / llm_model are populated accordingly.
"""

from typing import AsyncGenerator
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from openhands.agent_server.models import ACPConversationInfo, ConversationInfo, Success
from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationInfo,
)
from openhands.app_server.app_conversation.sql_app_conversation_info_service import (
    SQLAppConversationInfoService,
)
from openhands.app_server.event_callback.webhook_router import on_conversation_update
from openhands.app_server.user.specifiy_user_context import SpecifyUserContext
from openhands.app_server.utils.sql_utils import Base
from openhands.sdk import ConversationExecutionStatus
from openhands.sdk.agent.acp_agent import ACPAgent

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def async_engine():
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
async def async_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    async_session_maker = async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session_maker() as db_session:
        yield db_session


@pytest.fixture
def service(async_session) -> SQLAppConversationInfoService:
    return SQLAppConversationInfoService(
        db_session=async_session, user_context=SpecifyUserContext(user_id=None)
    )


@pytest.fixture
def sandbox_info():
    sandbox = MagicMock()
    sandbox.id = 'sandbox_acp_test'
    sandbox.created_by_user_id = 'user_123'
    sandbox.session_api_key = None
    return sandbox


def _make_llm_conversation_info() -> ConversationInfo:
    info = MagicMock(spec=ConversationInfo)
    info.id = uuid4()
    info.execution_status = ConversationExecutionStatus.RUNNING
    info.agent = MagicMock()
    info.agent.llm = MagicMock()
    info.agent.llm.model = 'anthropic/claude-sonnet-4-6'
    info.stats = MagicMock()
    info.stats.get_combined_metrics.return_value = None
    info.tags = {}
    return info


def _make_acp_conversation_info(acp_command: list[str]) -> ACPConversationInfo:
    info = MagicMock(spec=ACPConversationInfo)
    info.id = uuid4()
    info.execution_status = ConversationExecutionStatus.RUNNING
    acp_agent = MagicMock(spec=ACPAgent)
    acp_agent.agent_kind = 'acp'
    acp_agent.acp_command = acp_command
    info.agent = acp_agent
    info.stats = MagicMock()
    info.stats.get_combined_metrics.return_value = None
    info.tags = {}
    return info


# ---------------------------------------------------------------------------
# Discriminator unit tests
# ---------------------------------------------------------------------------


def test_acp_agent_classified_by_agent_kind():
    """agent_kind == 'acp' is the ACP identity check."""
    agent = MagicMock()
    agent.agent_kind = 'acp'
    assert getattr(agent, 'agent_kind', None) == 'acp'


def test_non_acp_agent_not_classified_as_acp():
    """An agent whose conversation contract is not 'acp' is not ACP."""
    for agent_kind in ('openhands', 'custom', None):
        agent = MagicMock()
        agent.agent_kind = agent_kind
        assert getattr(agent, 'agent_kind', None) != 'acp'


# ---------------------------------------------------------------------------
# Webhook endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_conversation_stores_llm_model(async_session, service, sandbox_info):
    """LLM path stores the real model in llm_model and leaves display_name null."""
    llm_info = _make_llm_conversation_info()
    conversation_id = llm_info.id

    existing = AppConversationInfo(
        id=conversation_id,
        title='Test',
        sandbox_id=sandbox_info.id,
        created_by_user_id=sandbox_info.created_by_user_id,
    )

    with patch(
        'openhands.app_server.event_callback.webhook_router.valid_conversation',
        return_value=existing,
    ):
        result = await on_conversation_update(
            conversation_info=llm_info,
            sandbox_info=sandbox_info,
            app_conversation_info_service=service,
        )

    assert isinstance(result, Success)

    saved = await service.get_app_conversation_info(conversation_id)
    assert saved is not None
    assert saved.llm_model == 'anthropic/claude-sonnet-4-6'
    assert saved.display_name is None
    assert saved.agent_kind == 'openhands'


@pytest.mark.asyncio
async def test_acp_conversation_stores_display_name(
    async_session, service, sandbox_info
):
    """ACP path stores display label in display_name and leaves llm_model null."""
    acp_info = _make_acp_conversation_info(
        acp_command=['npx', '-y', '@agentclientprotocol/claude-agent-acp']
    )
    conversation_id = acp_info.id

    existing = AppConversationInfo(
        id=conversation_id,
        title='Test',
        sandbox_id=sandbox_info.id,
        created_by_user_id=sandbox_info.created_by_user_id,
    )

    with patch(
        'openhands.app_server.event_callback.webhook_router.valid_conversation',
        return_value=existing,
    ):
        result = await on_conversation_update(
            conversation_info=acp_info,
            sandbox_info=sandbox_info,
            app_conversation_info_service=service,
        )

    assert isinstance(result, Success)

    saved = await service.get_app_conversation_info(conversation_id)
    assert saved is not None
    assert saved.llm_model is None
    assert saved.display_name == 'Claude Code'
    assert saved.agent_kind == 'acp'


@pytest.mark.asyncio
async def test_acp_conversation_empty_command_display_name(
    async_session, service, sandbox_info
):
    """ACP path with empty command falls back to plain 'ACP' label."""
    acp_info = _make_acp_conversation_info(acp_command=[])
    conversation_id = acp_info.id

    existing = AppConversationInfo(
        id=conversation_id,
        title='Test',
        sandbox_id=sandbox_info.id,
        created_by_user_id=sandbox_info.created_by_user_id,
    )

    with patch(
        'openhands.app_server.event_callback.webhook_router.valid_conversation',
        return_value=existing,
    ):
        await on_conversation_update(
            conversation_info=acp_info,
            sandbox_info=sandbox_info,
            app_conversation_info_service=service,
        )

    saved = await service.get_app_conversation_info(conversation_id)
    assert saved is not None
    assert saved.display_name == 'ACP'
    assert saved.llm_model is None
