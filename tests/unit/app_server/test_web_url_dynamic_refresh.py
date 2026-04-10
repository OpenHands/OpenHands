"""Tests for dynamic web_url refresh on sandbox restart (fixes #13861).

When a server's IP address changes between sandbox restarts, the MCP server
URL must reflect the current environment variables — not the stale value
that was captured when the config singleton was first created.
"""

import os
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from openhands.app_server.app_conversation.live_status_app_conversation_service import (
    LiveStatusAppConversationService,
)
from openhands.app_server.config import get_current_web_url
from openhands.app_server.sandbox.sandbox_models import SandboxStatus
from openhands.app_server.user.user_context import UserContext
from openhands.storage.data_models.settings import SandboxGroupingStrategy

# Env var used by openhands SDK LLM to skip context-window validation
_ALLOW_SHORT_CONTEXT_WINDOWS = 'ALLOW_SHORT_CONTEXT_WINDOWS'


@pytest.fixture(autouse=True)
def allow_short_context_windows():
    """Allow small context windows so unit tests can create LLM with gpt-4 etc."""
    old = os.environ.pop(_ALLOW_SHORT_CONTEXT_WINDOWS, None)
    os.environ[_ALLOW_SHORT_CONTEXT_WINDOWS] = 'true'
    try:
        yield
    finally:
        if old is not None:
            os.environ[_ALLOW_SHORT_CONTEXT_WINDOWS] = old
        else:
            os.environ.pop(_ALLOW_SHORT_CONTEXT_WINDOWS, None)


@pytest.fixture(autouse=True)
def clean_web_url_env():
    """Remove web URL env vars before/after each test to avoid cross-contamination."""
    old_oh = os.environ.pop('OH_WEB_URL', None)
    old_wh = os.environ.pop('WEB_HOST', None)
    yield
    if old_oh is not None:
        os.environ['OH_WEB_URL'] = old_oh
    else:
        os.environ.pop('OH_WEB_URL', None)
    if old_wh is not None:
        os.environ['WEB_HOST'] = old_wh
    else:
        os.environ.pop('WEB_HOST', None)


def _make_service(web_url: str | None = 'https://old-host.example.com') -> (
    LiveStatusAppConversationService
):
    """Create a LiveStatusAppConversationService with sensible test defaults."""
    mock_user_context = Mock(spec=UserContext)
    mock_user_context.user_auth = Mock()
    mock_user_context.get_mcp_api_key = AsyncMock(return_value=None)

    return LiveStatusAppConversationService(
        init_git_in_empty_workspace=True,
        user_context=mock_user_context,
        app_conversation_info_service=Mock(),
        app_conversation_start_task_service=Mock(),
        event_callback_service=Mock(),
        event_service=Mock(),
        sandbox_service=Mock(),
        sandbox_spec_service=Mock(),
        jwt_service=Mock(),
        pending_message_service=Mock(),
        sandbox_startup_timeout=30,
        sandbox_startup_poll_frequency=1,
        max_num_conversations_per_sandbox=20,
        httpx_client=Mock(),
        web_url=web_url,
        openhands_provider_base_url=None,
        access_token_hard_timeout=None,
        app_mode='test',
    )


# ---------------------------------------------------------------------------
# Tests for get_current_web_url()
# ---------------------------------------------------------------------------


class TestGetCurrentWebUrl:
    """Test the get_current_web_url helper that re-reads env vars."""

    def test_returns_none_when_no_env_vars_set(self):
        """Without OH_WEB_URL or WEB_HOST, should return None."""
        os.environ.pop('OH_WEB_URL', None)
        os.environ.pop('WEB_HOST', None)
        assert get_current_web_url() is None

    def test_returns_oh_web_url_when_set(self):
        """OH_WEB_URL takes priority."""
        os.environ['OH_WEB_URL'] = 'https://new-ip.example.com'
        assert get_current_web_url() == 'https://new-ip.example.com'

    def test_oh_web_url_takes_precedence_over_web_host(self):
        """OH_WEB_URL wins when both are set."""
        os.environ['OH_WEB_URL'] = 'https://via-oh-web-url.com'
        os.environ['WEB_HOST'] = 'via-web-host.com'
        assert get_current_web_url() == 'https://via-oh-web-url.com'

    def test_falls_back_to_web_host(self):
        """When only WEB_HOST is set, derive https URL from it."""
        os.environ.pop('OH_WEB_URL', None)
        os.environ['WEB_HOST'] = 'fallback-host.example.com'
        assert get_current_web_url() == 'https://fallback-host.example.com'

    def test_strips_whitespace_from_oh_web_url(self):
        """Trailing/leading whitespace should be stripped."""
        os.environ['OH_WEB_URL'] = '  https://stripped.example.com  '
        assert get_current_web_url() == 'https://stripped.example.com'

    def test_reflects_env_changes_immediately(self):
        """Changing the env var between calls should be reflected."""
        os.environ['OH_WEB_URL'] = 'https://first.example.com'
        assert get_current_web_url() == 'https://first.example.com'

        os.environ['OH_WEB_URL'] = 'https://second.example.com'
        assert get_current_web_url() == 'https://second.example.com'


# ---------------------------------------------------------------------------
# Tests for _resolve_web_url()
# ---------------------------------------------------------------------------


class TestResolveWebUrl:
    """Test that _resolve_web_url picks up environment changes."""

    def test_returns_env_value_when_env_set(self):
        """When the env var provides a new URL, use it."""
        service = _make_service(web_url='https://old.example.com')
        os.environ['OH_WEB_URL'] = 'https://new.example.com'

        result = service._resolve_web_url()
        assert result == 'https://new.example.com'

    def test_falls_back_to_init_web_url_when_no_env(self):
        """When no env var is set, fall back to the init-time web_url."""
        service = _make_service(web_url='https://init-time.example.com')
        os.environ.pop('OH_WEB_URL', None)
        os.environ.pop('WEB_HOST', None)

        result = service._resolve_web_url()
        assert result == 'https://init-time.example.com'

    def test_returns_none_when_both_are_none(self):
        """When neither env nor init value is set, return None."""
        service = _make_service(web_url=None)
        os.environ.pop('OH_WEB_URL', None)
        os.environ.pop('WEB_HOST', None)

        result = service._resolve_web_url()
        assert result is None

    def test_env_change_after_init_is_picked_up(self):
        """Simulate an IP change: env var changes after service creation."""
        service = _make_service(web_url='https://192.168.1.100')

        # Initially no env override — should use init value
        os.environ.pop('OH_WEB_URL', None)
        os.environ.pop('WEB_HOST', None)
        assert service._resolve_web_url() == 'https://192.168.1.100'

        # IP changes: env var is updated
        os.environ['OH_WEB_URL'] = 'https://192.168.1.200'
        assert service._resolve_web_url() == 'https://192.168.1.200'

        # IP changes again
        os.environ['OH_WEB_URL'] = 'https://10.0.0.5'
        assert service._resolve_web_url() == 'https://10.0.0.5'


# ---------------------------------------------------------------------------
# Tests for _add_system_mcp_servers using dynamic web_url
# ---------------------------------------------------------------------------


class TestAddSystemMcpServersDynamic:
    """Verify that _add_system_mcp_servers uses the dynamically resolved URL."""

    @pytest.mark.asyncio
    async def test_mcp_url_uses_current_env_not_init_value(self):
        """MCP URL should reflect the current OH_WEB_URL, not the stale init value."""
        service = _make_service(web_url='https://stale-host.example.com')

        # Simulate environment change (e.g., IP address changed)
        os.environ['OH_WEB_URL'] = 'https://fresh-host.example.com'

        mock_user = Mock()
        mock_user.search_api_key = None
        mock_user.mcp_config = None
        conversation_id = uuid4()

        mcp_servers: dict = {}
        await service._add_system_mcp_servers(mcp_servers, mock_user, conversation_id)

        assert 'default' in mcp_servers
        assert mcp_servers['default']['url'] == 'https://fresh-host.example.com/mcp/mcp'

    @pytest.mark.asyncio
    async def test_mcp_url_falls_back_to_init_value(self):
        """When no env var is set, MCP URL should use the init-time web_url."""
        service = _make_service(web_url='https://init-host.example.com')
        os.environ.pop('OH_WEB_URL', None)
        os.environ.pop('WEB_HOST', None)

        mock_user = Mock()
        mock_user.search_api_key = None
        mock_user.mcp_config = None
        conversation_id = uuid4()

        mcp_servers: dict = {}
        await service._add_system_mcp_servers(mcp_servers, mock_user, conversation_id)

        assert 'default' in mcp_servers
        assert mcp_servers['default']['url'] == 'https://init-host.example.com/mcp/mcp'

    @pytest.mark.asyncio
    async def test_no_mcp_servers_when_no_url_available(self):
        """When both env and init web_url are None, no MCP servers should be added."""
        service = _make_service(web_url=None)
        os.environ.pop('OH_WEB_URL', None)
        os.environ.pop('WEB_HOST', None)

        mock_user = Mock()
        mock_user.search_api_key = None
        mock_user.mcp_config = None
        conversation_id = uuid4()

        mcp_servers: dict = {}
        await service._add_system_mcp_servers(mcp_servers, mock_user, conversation_id)

        assert 'default' not in mcp_servers

    @pytest.mark.asyncio
    async def test_full_configure_llm_and_mcp_uses_dynamic_url(self):
        """End-to-end: _configure_llm_and_mcp should produce MCP config with fresh URL."""
        service = _make_service(web_url='https://old-ip.example.com')
        service._load_hooks_from_workspace = AsyncMock(return_value=None)

        # Simulate environment change
        os.environ['OH_WEB_URL'] = 'https://new-ip.example.com'

        mock_user = Mock()
        mock_user.llm_model = 'gpt-4'
        mock_user.llm_base_url = 'https://api.openai.com/v1'
        mock_user.llm_api_key = 'test_key'
        mock_user.sandbox_grouping_strategy = SandboxGroupingStrategy.ADD_TO_ANY
        mock_user.confirmation_mode = False
        mock_user.search_api_key = None
        mock_user.condenser_max_size = None
        mock_user.mcp_config = None
        conversation_id = uuid4()

        llm, mcp_config = await service._configure_llm_and_mcp(
            mock_user, None, conversation_id
        )

        assert 'mcpServers' in mcp_config
        assert 'default' in mcp_config['mcpServers']
        assert (
            mcp_config['mcpServers']['default']['url']
            == 'https://new-ip.example.com/mcp/mcp'
        )
