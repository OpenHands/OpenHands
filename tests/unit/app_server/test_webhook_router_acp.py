"""Tests for ACP agent discrimination in webhook_router.

Verifies that a ``ConversationInfo`` payload carrying an ``ACPAgent`` is
correctly discriminated from one carrying a regular ``Agent`` (via the
``AgentBase`` discriminated union the SDK exposes on the unified
``/api/conversations`` endpoint), and that ``agent_kind`` / ``llm_model``
are populated accordingly.
"""

from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from openhands.agent_server.models import ConversationInfo, Success
from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationInfo,
)
from openhands.app_server.app_conversation.sql_app_conversation_info_service import (
    SQLAppConversationInfoService,
)
from openhands.app_server.event_callback.webhook_router import on_conversation_update
from openhands.app_server.user.specifiy_user_context import SpecifyUserContext
from openhands.app_server.utils.sql_utils import Base
from openhands.sdk import Agent
from openhands.sdk.agent.acp_agent import ACPAgent
from openhands.sdk.llm import LLM

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
    """Build a real ``ConversationInfo`` with a real ``Agent``.

    Using real Pydantic models (rather than ``MagicMock(spec=...)``) exercises
    the discriminator (``isinstance(.agent, ACPAgent)``) and the serialization
    paths the production code relies on. The webhook only touches ``.id``,
    ``.execution_status``, ``.agent``, ``.stats``, ``.tags`` — everything else
    can ride on defaults.
    """
    agent = Agent(llm=LLM(model='anthropic/claude-sonnet-4-6', usage_id='test-usage'))
    return ConversationInfo.model_validate(
        {
            'id': str(uuid4()),
            'workspace': {'kind': 'LocalWorkspace', 'working_dir': '/tmp'},
            'persistence_dir': '/tmp/persist',
            'agent': agent.model_dump(mode='json'),
            'execution_status': 'running',
        }
    )


def _make_acp_conversation_info(acp_command: list[str]) -> ConversationInfo:
    """Build a real ``ConversationInfo`` with a real ``ACPAgent`` payload."""
    acp_agent = ACPAgent(acp_command=acp_command)
    return ConversationInfo.model_validate(
        {
            'id': str(uuid4()),
            'workspace': {'kind': 'LocalWorkspace', 'working_dir': '/tmp'},
            'persistence_dir': '/tmp/persist',
            'agent': acp_agent.model_dump(mode='json'),
            'execution_status': 'running',
        }
    )


# ---------------------------------------------------------------------------
# Webhook endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_conversation_stores_llm_model(async_session, service, sandbox_info):
    """LLM path stores the real model in llm_model and sets agent_kind='openhands'."""
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
    assert saved.agent_kind == 'openhands'


@pytest.mark.asyncio
async def test_acp_conversation_sets_agent_kind(async_session, service, sandbox_info):
    """ACP path sets agent_kind='acp' and leaves llm_model null."""
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
    assert saved.agent_kind == 'acp'


@pytest.mark.asyncio
async def test_acp_server_tag_preserved_on_webhook_update(
    async_session, service, sandbox_info
):
    """``tags['acp_server']`` set during creation must survive a webhook update.

    The live-status service stamps the active ACP provider key into
    ``tags['acp_server']`` when the conversation is first stored. Subsequent
    webhook updates merge incoming tags onto existing ones, so the provider
    key must still be present after a state-change webhook fires.
    """
    acp_info = _make_acp_conversation_info(acp_command=['my-acp'])
    conversation_id = acp_info.id

    existing = AppConversationInfo(
        id=conversation_id,
        title='Test',
        sandbox_id=sandbox_info.id,
        created_by_user_id=sandbox_info.created_by_user_id,
        tags={'acp_server': 'claude-code'},
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
    assert saved.tags.get('acp_server') == 'claude-code'


# ---------------------------------------------------------------------------
# Analytics — llm_model must not leak the ACP sentinel
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acp_conversation_analytics_llm_model_is_null(
    async_session, service, sandbox_info
):
    """``track_conversation_created`` must receive ``llm_model=None`` for ACP.

    Regression guard: ``ACPAgent.llm`` defaults to a dummy ``LLM(model='acp-managed')``
    sentinel, so reading ``conversation_info.agent.llm.model`` directly would
    record the literal string ``"acp-managed"`` in BIZZ-04 dashboards. The
    handler must use the agent-kind-aware ``llm_model`` variable instead.
    """
    acp_info = _make_acp_conversation_info(acp_command=['my-acp'])
    existing = AppConversationInfo(
        id=acp_info.id,
        title='Test',
        sandbox_id=sandbox_info.id,
        created_by_user_id=sandbox_info.created_by_user_id,
    )
    analytics = MagicMock()

    with (
        patch(
            'openhands.app_server.event_callback.webhook_router.valid_conversation',
            return_value=existing,
        ),
        patch(
            'openhands.app_server.event_callback.webhook_router.get_analytics_service',
            return_value=analytics,
        ),
        patch(
            'openhands.app_server.event_callback.webhook_router.resolve_analytics_context',
            new=AsyncMock(return_value=MagicMock()),
        ),
    ):
        await on_conversation_update(
            conversation_info=acp_info,
            sandbox_info=sandbox_info,
            app_conversation_info_service=service,
        )

    analytics.track_conversation_created.assert_called_once()
    kwargs = analytics.track_conversation_created.call_args.kwargs
    assert kwargs['llm_model'] is None


# ---------------------------------------------------------------------------
# Backward compatibility — discriminated union deserialisation
# ---------------------------------------------------------------------------


def test_legacy_llm_payload_deserialises_as_agent():
    """A legacy LLM webhook payload still routes to the ``Agent`` branch.

    The webhook signature accepts ``ConversationInfo`` whose ``agent`` field
    is the ``AgentBase`` discriminated union (``Agent | ACPAgent``). This
    test proves that an old-style payload (no ACP fields, ``kind='Agent'``)
    deserialises into the LLM branch and that ``isinstance(.agent, ACPAgent)``
    correctly returns ``False`` — i.e. the on_conversation_update branch
    that writes ``llm_model`` is selected.
    """
    from openhands.sdk.agent.agent import Agent
    from openhands.sdk.llm import LLM

    agent = Agent(llm=LLM(model='anthropic/claude-sonnet-4-6', usage_id='test-usage'))
    legacy_payload: dict = {
        'id': str(uuid4()),
        'workspace': {'kind': 'LocalWorkspace', 'working_dir': '/tmp'},
        'persistence_dir': '/tmp/persist',
        'agent': agent.model_dump(mode='json'),
        'execution_status': 'running',
    }

    parsed = ConversationInfo.model_validate(legacy_payload)

    assert parsed.agent.kind == 'Agent'
    assert not isinstance(parsed.agent, ACPAgent)
    # And the LLM model survived the round-trip — proves the field is reachable
    # the same way the webhook handler reads it (``conversation_info.agent.llm.model``).
    assert parsed.agent.llm.model == 'anthropic/claude-sonnet-4-6'
