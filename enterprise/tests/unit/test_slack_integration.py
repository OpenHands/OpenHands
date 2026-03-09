from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from integrations.slack.slack_manager import SlackManager
from integrations.slack.slack_view import SlackNewConversationView
from storage.slack_user import SlackUser

from openhands.integrations.service_types import ProviderTimeoutError
from openhands.server.user_auth.user_auth import UserAuth


@pytest.fixture
def slack_manager():
    # Mock the token_manager constructor
    slack_manager = SlackManager(token_manager=MagicMock())
    return slack_manager


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
    """Create a SlackNewConversationView instance for testing."""
    return SlackNewConversationView(
        bot_access_token='xoxb-test-token',
        user_msg='Hello OpenHands!',
        slack_user_id='U1234567890',
        slack_to_openhands_user=mock_slack_user,
        saas_user_auth=mock_user_auth,
        channel_id='C1234567890',
        message_ts='1234567890.123456',
        thread_ts=None,
        selected_repo=None,
        should_extract=True,
        send_summary_instruction=True,
        conversation_id='',
        team_id='T1234567890',
        v1_enabled=False,
    )


@pytest.mark.parametrize(
    'message,expected',
    [
        ('OpenHands/Openhands', 'OpenHands/Openhands'),
        ('deploy repo', 'deploy'),
        ('use hello world', None),
    ],
)
def test_infer_repo_from_message(message, expected, slack_manager):
    # Test the extracted function
    result = slack_manager._infer_repo_from_message(message)
    assert result == expected


class TestRepoQueryTimeoutHandling:
    """Test timeout handling when fetching repositories for Slack integration."""

    @patch.object(SlackManager, 'send_message', new_callable=AsyncMock)
    @patch.object(SlackManager, '_get_repositories', new_callable=AsyncMock)
    async def test_timeout_sends_user_friendly_message(
        self,
        mock_get_repositories,
        mock_send_message,
        slack_manager,
        slack_new_conversation_view,
    ):
        """Test that when repository fetching times out, a user-friendly message is sent."""
        # Setup: _get_repositories raises ProviderTimeoutError
        mock_get_repositories.side_effect = ProviderTimeoutError(
            'github API request timed out: ConnectTimeout'
        )

        # Execute
        result = await slack_manager.is_job_requested(
            MagicMock(), slack_new_conversation_view
        )

        # Verify: should return False (job not started)
        assert result is False

        # Verify: send_message was called with the timeout message
        mock_send_message.assert_called_once()
        call_args = mock_send_message.call_args

        # Check the message content
        message = call_args[0][0]
        assert 'timed out' in message
        assert 'repository name' in message
        assert 'owner/repo-name' in message

        # Check it was sent as ephemeral
        assert call_args[1]['ephemeral'] is True

    @patch.object(SlackManager, 'send_message', new_callable=AsyncMock)
    @patch.object(SlackManager, '_get_repositories', new_callable=AsyncMock)
    async def test_successful_repo_fetch_does_not_send_timeout_message(
        self,
        mock_get_repositories,
        mock_send_message,
        slack_manager,
        slack_new_conversation_view,
    ):
        """Test that successful repo fetch shows repo selector, not timeout message."""
        # Setup: _get_repositories returns empty list (no repos, but no timeout)
        mock_get_repositories.return_value = []

        # Execute
        result = await slack_manager.is_job_requested(
            MagicMock(), slack_new_conversation_view
        )

        # Verify: should return False (no repo selected yet)
        assert result is False

        # Verify: send_message was called (for repo selector)
        mock_send_message.assert_called_once()
        call_args = mock_send_message.call_args

        # Check the message is NOT the timeout message
        message = call_args[0][0]
        assert 'timed out' not in str(message)
        # Should be the repo selection form
        assert isinstance(message, dict)
        assert message.get('text') == 'Choose a Repository:'
