"""Tests for the Slack view classes and their v1 vs v0 conversation handling.

Focuses on the 3 essential scenarios:
1. V1 vs V0 decision logic based on user setting
2. Message routing to correct method based on conversation v1 flag
3. Paused sandbox resumption for V1 conversations
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from integrations.slack.slack_view import (
    SlackNewConversationView,
    SlackUpdateExistingConversationView,
)
from jinja2 import DictLoader, Environment
from storage.slack_conversation import SlackConversation
from storage.slack_user import SlackUser

from openhands.app_server.sandbox.sandbox_models import SandboxStatus
from openhands.server.user_auth.user_auth import UserAuth

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_jinja_env():
    """Create a mock Jinja environment with test templates."""
    templates = {
        'user_message_conversation_instructions.j2': 'Previous messages: {{ messages|join(", ") }}\nUser: {{ username }}\nURL: {{ conversation_url }}'
    }
    return Environment(loader=DictLoader(templates))


@pytest.fixture
def mock_slack_user():
    """Create a mock SlackUser."""
    user = SlackUser()
    user.slack_user_id = 'U1234567890'
    user.keycloak_user_id = 'test-user-123'
    user.slack_display_name = 'Test User'
    return user


@pytest.fixture
def mock_user_auth():
    """Create a mock UserAuth."""
    auth = MagicMock(spec=UserAuth)
    auth.get_provider_tokens = AsyncMock(return_value={})
    auth.get_secrets = AsyncMock(return_value=MagicMock(custom_secrets={}))
    return auth


@pytest.fixture
def slack_new_conversation_view(mock_slack_user, mock_user_auth):
    """Create a SlackNewConversationView instance."""
    return SlackNewConversationView(
        bot_access_token='xoxb-test-token',
        user_msg='Hello OpenHands!',
        slack_user_id='U1234567890',
        slack_to_openhands_user=mock_slack_user,
        saas_user_auth=mock_user_auth,
        channel_id='C1234567890',
        message_ts='1234567890.123456',
        thread_ts=None,
        selected_repo='owner/repo',
        should_extract=True,
        send_summary_instruction=True,
        conversation_id='',
        team_id='T1234567890',
        v1_enabled=False,
    )


@pytest.fixture
def slack_update_conversation_view_v0(mock_slack_user, mock_user_auth):
    """Create a SlackUpdateExistingConversationView instance for V0."""
    conversation_id = '87654321-4321-8765-4321-876543218765'
    mock_conversation = SlackConversation(
        conversation_id=conversation_id,
        channel_id='C1234567890',
        keycloak_user_id='test-user-123',
        parent_id='1234567890.123456',
        v1_enabled=False,
    )
    return SlackUpdateExistingConversationView(
        bot_access_token='xoxb-test-token',
        user_msg='Follow up message',
        slack_user_id='U1234567890',
        slack_to_openhands_user=mock_slack_user,
        saas_user_auth=mock_user_auth,
        channel_id='C1234567890',
        message_ts='1234567890.123457',
        thread_ts='1234567890.123456',
        selected_repo=None,
        should_extract=True,
        send_summary_instruction=True,
        conversation_id=conversation_id,
        slack_conversation=mock_conversation,
        team_id='T1234567890',
        v1_enabled=False,
    )


@pytest.fixture
def slack_update_conversation_view_v1(mock_slack_user, mock_user_auth):
    """Create a SlackUpdateExistingConversationView instance for V1."""
    conversation_id = '12345678-1234-5678-1234-567812345678'
    mock_conversation = SlackConversation(
        conversation_id=conversation_id,
        channel_id='C1234567890',
        keycloak_user_id='test-user-123',
        parent_id='1234567890.123456',
        v1_enabled=True,
    )
    return SlackUpdateExistingConversationView(
        bot_access_token='xoxb-test-token',
        user_msg='Follow up message',
        slack_user_id='U1234567890',
        slack_to_openhands_user=mock_slack_user,
        saas_user_auth=mock_user_auth,
        channel_id='C1234567890',
        message_ts='1234567890.123457',
        thread_ts='1234567890.123456',
        selected_repo=None,
        should_extract=True,
        send_summary_instruction=True,
        conversation_id=conversation_id,
        slack_conversation=mock_conversation,
        team_id='T1234567890',
        v1_enabled=True,
    )


# ---------------------------------------------------------------------------
# Test 1: V1 vs V0 Decision Logic Based on User Setting
# ---------------------------------------------------------------------------


class TestV1V0DecisionLogic:
    """Test the decision logic for choosing between V1 and V0 conversations based on user setting."""

    @pytest.mark.parametrize(
        'v1_enabled,expected_v1_flag',
        [
            (True, True),  # V1 enabled, use V1
            (False, False),  # V1 disabled, use V0
        ],
    )
    @patch('integrations.slack.slack_view.is_v1_enabled_for_slack_resolver')
    @patch.object(SlackNewConversationView, '_create_v1_conversation')
    @patch.object(SlackNewConversationView, '_create_v0_conversation')
    async def test_v1_v0_decision_logic(
        self,
        mock_create_v0,
        mock_create_v1,
        mock_is_v1_enabled,
        slack_new_conversation_view,
        mock_jinja_env,
        v1_enabled,
        expected_v1_flag,
    ):
        """Test the decision logic for V1 vs V0 conversation creation based on user setting."""
        # Setup mocks
        mock_is_v1_enabled.return_value = v1_enabled
        mock_create_v1.return_value = None
        mock_create_v0.return_value = None

        # Execute
        result = await slack_new_conversation_view.create_or_update_conversation(
            mock_jinja_env
        )

        # Verify
        assert result == slack_new_conversation_view.conversation_id
        assert slack_new_conversation_view.v1_enabled == expected_v1_flag

        if v1_enabled:
            mock_create_v1.assert_called_once()
            mock_create_v0.assert_not_called()
        else:
            mock_create_v1.assert_not_called()
            mock_create_v0.assert_called_once()


# ---------------------------------------------------------------------------
# Test 2: Message Routing Based on Conversation V1 Flag
# ---------------------------------------------------------------------------


class TestMessageRouting:
    """Test that message sending routes to correct method based on conversation v1 flag."""

    @patch.object(
        SlackUpdateExistingConversationView, 'send_message_to_v1_conversation'
    )
    @patch.object(
        SlackUpdateExistingConversationView, 'send_message_to_v0_conversation'
    )
    async def test_message_routing_to_v1(
        self,
        mock_send_v0,
        mock_send_v1,
        slack_update_conversation_view_v1,
        mock_jinja_env,
    ):
        """Test that V1 conversations route to V1 message sending method."""
        # Setup
        mock_send_v0.return_value = None
        mock_send_v1.return_value = None

        # Execute
        result = await slack_update_conversation_view_v1.create_or_update_conversation(
            mock_jinja_env
        )

        # Verify
        assert result == slack_update_conversation_view_v1.conversation_id
        mock_send_v1.assert_called_once_with(mock_jinja_env)
        mock_send_v0.assert_not_called()

    @patch.object(
        SlackUpdateExistingConversationView, 'send_message_to_v1_conversation'
    )
    @patch.object(
        SlackUpdateExistingConversationView, 'send_message_to_v0_conversation'
    )
    async def test_message_routing_to_v0(
        self,
        mock_send_v0,
        mock_send_v1,
        slack_update_conversation_view_v0,
        mock_jinja_env,
    ):
        """Test that V0 conversations route to V0 message sending method."""
        # Setup
        mock_send_v0.return_value = None
        mock_send_v1.return_value = None

        # Execute
        result = await slack_update_conversation_view_v0.create_or_update_conversation(
            mock_jinja_env
        )

        # Verify
        assert result == slack_update_conversation_view_v0.conversation_id
        mock_send_v0.assert_called_once_with(mock_jinja_env)
        mock_send_v1.assert_not_called()


# ---------------------------------------------------------------------------
# Test 3: Paused Sandbox Resumption for V1 Conversations
# ---------------------------------------------------------------------------


class TestPausedSandboxResumption:
    """Test that paused sandboxes are resumed when sending messages to V1 conversations."""

    @patch('openhands.app_server.config.get_sandbox_service')
    @patch('openhands.app_server.config.get_app_conversation_info_service')
    @patch('openhands.app_server.config.get_httpx_client')
    @patch('openhands.app_server.event_callback.util.ensure_running_sandbox')
    @patch('openhands.app_server.event_callback.util.get_agent_server_url_from_sandbox')
    @patch.object(SlackUpdateExistingConversationView, '_get_instructions')
    async def test_paused_sandbox_resumption(
        self,
        mock_get_instructions,
        mock_get_agent_server_url,
        mock_ensure_running_sandbox,
        mock_get_httpx_client,
        mock_get_app_info_service,
        mock_get_sandbox_service,
        slack_update_conversation_view_v1,
        mock_jinja_env,
    ):
        """Test that paused sandboxes are resumed when sending messages to V1 conversations."""
        # Setup mocks
        mock_get_instructions.return_value = ('User message', '')

        # Mock app conversation info service
        mock_app_info_service = AsyncMock()
        mock_app_info = MagicMock()
        mock_app_info.sandbox_id = 'sandbox-123'
        mock_app_info_service.get_app_conversation_info.return_value = mock_app_info
        mock_get_app_info_service.return_value.__aenter__.return_value = (
            mock_app_info_service
        )

        # Mock sandbox service with paused sandbox that gets resumed
        mock_sandbox_service = AsyncMock()
        mock_paused_sandbox = MagicMock()
        mock_paused_sandbox.status = SandboxStatus.PAUSED
        mock_paused_sandbox.session_api_key = 'test-api-key'
        mock_paused_sandbox.exposed_urls = [
            MagicMock(name='AGENT_SERVER', url='http://localhost:8000')
        ]

        # After resume, sandbox becomes running
        mock_running_sandbox = MagicMock()
        mock_running_sandbox.status = SandboxStatus.RUNNING
        mock_running_sandbox.session_api_key = 'test-api-key'
        mock_running_sandbox.exposed_urls = [
            MagicMock(name='AGENT_SERVER', url='http://localhost:8000')
        ]

        mock_sandbox_service.get_sandbox.side_effect = [
            mock_paused_sandbox,
            mock_running_sandbox,
        ]
        mock_sandbox_service.resume_sandbox = AsyncMock()
        mock_get_sandbox_service.return_value.__aenter__.return_value = (
            mock_sandbox_service
        )

        # Mock ensure_running_sandbox to first raise RuntimeError, then return running sandbox
        mock_ensure_running_sandbox.side_effect = [
            RuntimeError('Sandbox not running: sandbox-123'),
            mock_running_sandbox,
        ]

        # Mock agent server URL
        mock_get_agent_server_url.return_value = 'http://localhost:8000'

        # Mock HTTP client
        mock_httpx_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_httpx_client.post.return_value = mock_response
        mock_get_httpx_client.return_value.__aenter__.return_value = mock_httpx_client

        # Execute
        await slack_update_conversation_view_v1.send_message_to_v1_conversation(
            mock_jinja_env
        )

        # Verify sandbox was resumed
        mock_sandbox_service.resume_sandbox.assert_called_once_with('sandbox-123')
        mock_httpx_client.post.assert_called_once()
        mock_response.raise_for_status.assert_called_once()
