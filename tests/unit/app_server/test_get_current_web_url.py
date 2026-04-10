"""Unit tests for _get_current_web_url() and its integration with _add_system_mcp_servers().

These tests verify that MCP server URLs are dynamically resolved from environment
variables rather than using a stale snapshot, fixing the bug where IP changes after
sandbox restart would leave the agent unable to connect to the MCP server.

Integration / E2E verification plan (not automated here):
  1. Start OpenHands with OH_WEB_URL=http://ip-a:3000
  2. Create a conversation and verify the MCP default server URL is http://ip-a:3000/mcp/mcp
  3. Stop the service, change to OH_WEB_URL=http://ip-b:3000
  4. Restart the sandbox / resume the conversation
  5. Verify the agent's MCP default server URL is now http://ip-b:3000/mcp/mcp
"""

import os
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from openhands.app_server.app_conversation.live_status_app_conversation_service import (
    LiveStatusAppConversationService,
)
from openhands.app_server.sandbox.docker_sandbox_service import DockerSandboxService
from openhands.app_server.user.user_context import UserContext


class TestGetCurrentWebUrl:
    """Test cases for _get_current_web_url() dynamic URL resolution."""

    def setup_method(self):
        """Set up test fixtures matching the project's existing pattern."""
        self.mock_user_context = Mock(spec=UserContext)
        self.mock_user_context.user_auth = Mock()
        self.mock_sandbox_service = Mock()

        self.service = LiveStatusAppConversationService(
            init_git_in_empty_workspace=True,
            user_context=self.mock_user_context,
            app_conversation_info_service=Mock(),
            app_conversation_start_task_service=Mock(),
            event_callback_service=Mock(),
            event_service=Mock(),
            sandbox_service=self.mock_sandbox_service,
            sandbox_spec_service=Mock(),
            jwt_service=Mock(),
            pending_message_service=Mock(),
            sandbox_startup_timeout=30,
            sandbox_startup_poll_frequency=1,
            max_num_conversations_per_sandbox=20,
            httpx_client=Mock(),
            web_url='https://snapshot.example.com',
            openhands_provider_base_url=None,
            access_token_hard_timeout=None,
        )

        # Clean env vars that could interfere
        self._saved_env = {}
        for key in ('OH_WEB_URL', 'WEB_HOST'):
            self._saved_env[key] = os.environ.pop(key, None)

    def teardown_method(self):
        """Restore environment variables."""
        for key, value in self._saved_env.items():
            if value is not None:
                os.environ[key] = value
            else:
                os.environ.pop(key, None)

    def test_oh_web_url_takes_highest_priority(self):
        """OH_WEB_URL env var should be returned when set, regardless of other sources."""
        with patch.dict(
            os.environ,
            {'OH_WEB_URL': 'http://from-oh-env:3000', 'WEB_HOST': 'from-web-host'},
        ):
            result = self.service._get_current_web_url()

        assert result == 'http://from-oh-env:3000'

    def test_web_host_used_when_oh_web_url_absent(self):
        """WEB_HOST env var should be used (with https) when OH_WEB_URL is not set."""
        with patch.dict(os.environ, {'WEB_HOST': 'my-host.example.com'}, clear=False):
            # Ensure OH_WEB_URL is not set
            os.environ.pop('OH_WEB_URL', None)
            result = self.service._get_current_web_url()

        assert result == 'https://my-host.example.com'

    def test_fallback_to_self_web_url_when_no_env_vars(self):
        """When no env vars are set, should fall back to the snapshot self.web_url."""
        os.environ.pop('OH_WEB_URL', None)
        os.environ.pop('WEB_HOST', None)

        result = self.service._get_current_web_url()

        assert result == 'https://snapshot.example.com'

    def test_env_var_change_returns_new_value(self):
        """Changing OH_WEB_URL between calls should return the updated value."""
        with patch.dict(os.environ, {'OH_WEB_URL': 'http://old-ip:3000'}):
            first = self.service._get_current_web_url()

        with patch.dict(os.environ, {'OH_WEB_URL': 'http://new-ip:3000'}):
            second = self.service._get_current_web_url()

        assert first == 'http://old-ip:3000'
        assert second == 'http://new-ip:3000'

    def test_docker_fallback_when_no_url_available(self):
        """When no env vars and no self.web_url, Docker fallback should be used."""
        # Create a service with web_url=None and a DockerSandboxService mock
        mock_docker_service = Mock(spec=DockerSandboxService)
        mock_docker_service.host_port = 3000
        service = LiveStatusAppConversationService(
            init_git_in_empty_workspace=True,
            user_context=self.mock_user_context,
            app_conversation_info_service=Mock(),
            app_conversation_start_task_service=Mock(),
            event_callback_service=Mock(),
            event_service=Mock(),
            sandbox_service=mock_docker_service,
            sandbox_spec_service=Mock(),
            jwt_service=Mock(),
            pending_message_service=Mock(),
            sandbox_startup_timeout=30,
            sandbox_startup_poll_frequency=1,
            max_num_conversations_per_sandbox=20,
            httpx_client=Mock(),
            web_url=None,
            openhands_provider_base_url=None,
            access_token_hard_timeout=None,
        )

        os.environ.pop('OH_WEB_URL', None)
        os.environ.pop('WEB_HOST', None)

        result = service._get_current_web_url()

        assert result == 'http://host.docker.internal:3000'

    def test_returns_none_when_nothing_available(self):
        """When no env vars, no self.web_url, and not Docker, should return None."""
        service = LiveStatusAppConversationService(
            init_git_in_empty_workspace=True,
            user_context=self.mock_user_context,
            app_conversation_info_service=Mock(),
            app_conversation_start_task_service=Mock(),
            event_callback_service=Mock(),
            event_service=Mock(),
            sandbox_service=self.mock_sandbox_service,  # not DockerSandboxService
            sandbox_spec_service=Mock(),
            jwt_service=Mock(),
            pending_message_service=Mock(),
            sandbox_startup_timeout=30,
            sandbox_startup_poll_frequency=1,
            max_num_conversations_per_sandbox=20,
            httpx_client=Mock(),
            web_url=None,
            openhands_provider_base_url=None,
            access_token_hard_timeout=None,
        )

        os.environ.pop('OH_WEB_URL', None)
        os.environ.pop('WEB_HOST', None)

        result = service._get_current_web_url()

        assert result is None


class TestAddSystemMcpServersUsesCurrentUrl:
    """Verify _add_system_mcp_servers() uses _get_current_web_url(), not self.web_url."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_user_context = Mock(spec=UserContext)
        self.mock_user_context.user_auth = Mock()
        self.mock_user_context.get_mcp_api_key = AsyncMock(return_value=None)

        self.service = LiveStatusAppConversationService(
            init_git_in_empty_workspace=True,
            user_context=self.mock_user_context,
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
            web_url='https://stale-snapshot.example.com',
            openhands_provider_base_url=None,
            access_token_hard_timeout=None,
        )

        self.mock_user = Mock()
        self.mock_user.search_api_key = None
        self.conversation_id = uuid4()

    @pytest.mark.asyncio
    async def test_mcp_url_uses_dynamic_web_url_not_snapshot(self):
        """The default MCP server URL must come from _get_current_web_url(), not self.web_url."""
        dynamic_url = 'http://dynamic-ip:4000'

        with patch.object(
            self.service, '_get_current_web_url', return_value=dynamic_url
        ):
            mcp_servers: dict = {}
            await self.service._add_system_mcp_servers(
                mcp_servers, self.mock_user, self.conversation_id
            )

        assert mcp_servers['default']['url'] == f'{dynamic_url}/mcp/mcp'
        # Confirm it did NOT use the stale snapshot
        assert 'stale-snapshot' not in mcp_servers['default']['url']

    @pytest.mark.asyncio
    async def test_no_servers_added_when_url_is_none(self):
        """When _get_current_web_url() returns None, no default server should be added."""
        with patch.object(self.service, '_get_current_web_url', return_value=None):
            mcp_servers: dict = {}
            await self.service._add_system_mcp_servers(
                mcp_servers, self.mock_user, self.conversation_id
            )

        assert 'default' not in mcp_servers

    @pytest.mark.asyncio
    async def test_mcp_url_reflects_env_var_change(self):
        """End-to-end: changing OH_WEB_URL between calls changes the MCP server URL."""
        with patch.dict(os.environ, {'OH_WEB_URL': 'http://ip-a:3000'}):
            servers_a: dict = {}
            await self.service._add_system_mcp_servers(
                servers_a, self.mock_user, self.conversation_id
            )

        with patch.dict(os.environ, {'OH_WEB_URL': 'http://ip-b:3000'}):
            servers_b: dict = {}
            await self.service._add_system_mcp_servers(
                servers_b, self.mock_user, self.conversation_id
            )

        assert servers_a['default']['url'] == 'http://ip-a:3000/mcp/mcp'
        assert servers_b['default']['url'] == 'http://ip-b:3000/mcp/mcp'
