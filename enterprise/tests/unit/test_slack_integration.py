from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from integrations.slack.slack_manager import (
    SLACK_USER_MSG_EXPIRATION,
    SLACK_USER_MSG_KEY_PREFIX,
    SlackManager,
)
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
    @patch.object(SlackManager, '_search_repositories', new_callable=AsyncMock)
    async def test_timeout_sends_user_friendly_message_and_shows_selector(
        self,
        mock_search_repositories,
        mock_send_message,
        slack_manager,
        slack_new_conversation_view,
    ):
        """Test that when repository search times out, a message is sent and selector is shown."""
        # Setup: Modify message to include a repo reference to trigger the search
        slack_new_conversation_view.user_msg = 'Help me with OpenHands/OpenHands repo'

        # Setup: _search_repositories raises ProviderTimeoutError
        mock_search_repositories.side_effect = ProviderTimeoutError(
            'github API request timed out: ConnectTimeout'
        )

        # Execute
        result = await slack_manager.is_job_requested(
            MagicMock(), slack_new_conversation_view
        )

        # Verify: should return False (job not started, but selector is shown)
        assert result is False

        # Verify: send_message was called twice:
        # 1. Timeout warning message
        # 2. Repository selection form
        assert mock_send_message.call_count == 2

        # Check the first call (timeout message)
        first_call_args = mock_send_message.call_args_list[0]
        timeout_message = first_call_args[0][0]
        assert 'timed out' in timeout_message
        assert first_call_args[1]['ephemeral'] is True

        # Check the second call (repo selector)
        second_call_args = mock_send_message.call_args_list[1]
        selector_message = second_call_args[0][0]
        assert isinstance(selector_message, dict)
        assert selector_message.get('text') == 'Choose a Repository:'

    @patch.object(SlackManager, 'send_message', new_callable=AsyncMock)
    @patch.object(SlackManager, '_search_repositories', new_callable=AsyncMock)
    async def test_successful_repo_fetch_shows_external_selector(
        self,
        mock_search_repositories,
        mock_send_message,
        slack_manager,
        slack_new_conversation_view,
    ):
        """Test that successful search shows external_select repo selector."""
        # Setup: _search_repositories returns empty list (no repos, but no timeout)
        mock_search_repositories.return_value = []

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
        # Should be the repo selection form with external_select
        assert isinstance(message, dict)
        assert message.get('text') == 'Choose a Repository:'
        # Verify it's using external_select
        blocks = message.get('blocks', [])
        actions_block = next((b for b in blocks if b.get('type') == 'actions'), None)
        assert actions_block is not None
        elements = actions_block.get('elements', [])
        assert len(elements) > 0
        assert elements[0].get('type') == 'external_select'


class TestBuildRepoOptions:
    """Test the _build_repo_options helper method."""

    def test_build_options_with_repos(self, slack_manager):
        """Test building options from a list of repositories."""
        from openhands.integrations.service_types import ProviderType, Repository

        repos = [
            Repository(
                id='1',
                full_name='owner/repo1',
                git_provider=ProviderType.GITHUB,
                is_public=True,
            ),
            Repository(
                id='2',
                full_name='owner/repo2',
                git_provider=ProviderType.GITHUB,
                is_public=False,
            ),
        ]

        options = slack_manager._build_repo_options(repos, include_no_repo=True)

        # Should have 3 options: "No Repository" + 2 repos
        assert len(options) == 3
        assert options[0]['value'] == '-'
        assert options[0]['text']['text'] == 'No Repository'
        assert options[1]['value'] == 'owner/repo1'
        assert options[2]['value'] == 'owner/repo2'

    def test_build_options_without_no_repo(self, slack_manager):
        """Test building options without the No Repository option."""
        from openhands.integrations.service_types import ProviderType, Repository

        repos = [
            Repository(
                id='1',
                full_name='owner/repo1',
                git_provider=ProviderType.GITHUB,
                is_public=True,
            ),
        ]

        options = slack_manager._build_repo_options(repos, include_no_repo=False)

        # Should have 1 option (just the repo)
        assert len(options) == 1
        assert options[0]['value'] == 'owner/repo1'

    def test_build_options_truncates_long_names(self, slack_manager):
        """Test that repo names longer than 75 chars are truncated."""
        from openhands.integrations.service_types import ProviderType, Repository

        long_name = 'a' * 100
        repos = [
            Repository(
                id='1',
                full_name=long_name,
                git_provider=ProviderType.GITHUB,
                is_public=True,
            ),
        ]

        options = slack_manager._build_repo_options(repos, include_no_repo=False)

        # Text should be truncated to 75 chars
        assert len(options[0]['text']['text']) == 75
        # But value should have full name
        assert options[0]['value'] == long_name


class TestUserMsgStorage:
    """Test the user message storage for repo selection form flow."""

    @patch('integrations.slack.slack_manager.sio')
    async def test_store_user_msg_for_form(self, mock_sio, slack_manager):
        """Test storing user message in Redis."""
        mock_redis = AsyncMock()
        mock_sio.manager.redis = mock_redis

        message_ts = '1234567890.123456'
        thread_ts = '1234567890.111111'
        user_msg = 'Hello OpenHands, help me with my code'

        await slack_manager.store_user_msg_for_form(message_ts, thread_ts, user_msg)

        expected_key = f'{SLACK_USER_MSG_KEY_PREFIX}:{message_ts}:{thread_ts}'
        mock_redis.set.assert_called_once_with(
            expected_key, user_msg, ex=SLACK_USER_MSG_EXPIRATION
        )

    @patch('integrations.slack.slack_manager.sio')
    async def test_store_user_msg_for_form_no_thread(self, mock_sio, slack_manager):
        """Test storing user message when there's no thread."""
        mock_redis = AsyncMock()
        mock_sio.manager.redis = mock_redis

        message_ts = '1234567890.123456'
        thread_ts = None
        user_msg = 'Hello OpenHands'

        await slack_manager.store_user_msg_for_form(message_ts, thread_ts, user_msg)

        expected_key = f'{SLACK_USER_MSG_KEY_PREFIX}:{message_ts}:{thread_ts}'
        mock_redis.set.assert_called_once_with(
            expected_key, user_msg, ex=SLACK_USER_MSG_EXPIRATION
        )

    @patch('integrations.slack.slack_manager.sio')
    async def test_retrieve_user_msg_for_form_found(self, mock_sio, slack_manager):
        """Test retrieving user message from Redis when it exists."""
        mock_redis = AsyncMock()
        mock_redis.get.return_value = b'Hello OpenHands, help me with my code'
        mock_sio.manager.redis = mock_redis

        message_ts = '1234567890.123456'
        thread_ts = '1234567890.111111'

        result = await slack_manager.retrieve_user_msg_for_form(message_ts, thread_ts)

        expected_key = f'{SLACK_USER_MSG_KEY_PREFIX}:{message_ts}:{thread_ts}'
        mock_redis.get.assert_called_once_with(expected_key)
        assert result == 'Hello OpenHands, help me with my code'

    @patch('integrations.slack.slack_manager.sio')
    async def test_retrieve_user_msg_for_form_not_found(self, mock_sio, slack_manager):
        """Test retrieving user message from Redis when it doesn't exist."""
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        mock_sio.manager.redis = mock_redis

        message_ts = '1234567890.123456'
        thread_ts = None

        result = await slack_manager.retrieve_user_msg_for_form(message_ts, thread_ts)

        expected_key = f'{SLACK_USER_MSG_KEY_PREFIX}:{message_ts}:{thread_ts}'
        mock_redis.get.assert_called_once_with(expected_key)
        assert result is None

    @patch('integrations.slack.slack_manager.sio')
    async def test_retrieve_user_msg_for_form_string_response(
        self, mock_sio, slack_manager
    ):
        """Test retrieving user message when Redis returns string instead of bytes."""
        mock_redis = AsyncMock()
        mock_redis.get.return_value = 'Hello OpenHands'
        mock_sio.manager.redis = mock_redis

        message_ts = '1234567890.123456'
        thread_ts = None

        result = await slack_manager.retrieve_user_msg_for_form(message_ts, thread_ts)

        assert result == 'Hello OpenHands'


class TestIsJobRequestedWithUserMsgStorage:
    """Test that is_job_requested properly stores user message for form flow."""

    @patch('integrations.slack.slack_manager.sio')
    @patch.object(SlackManager, 'send_message', new_callable=AsyncMock)
    @patch.object(SlackManager, '_search_repositories', new_callable=AsyncMock)
    async def test_stores_user_msg_when_showing_repo_selector(
        self,
        mock_search_repositories,
        mock_send_message,
        mock_sio,
        slack_manager,
        slack_new_conversation_view,
    ):
        """Test that user_msg is stored in Redis when repo selector is shown."""
        mock_redis = AsyncMock()
        mock_sio.manager.redis = mock_redis

        # Setup: _search_repositories returns empty list
        mock_search_repositories.return_value = []

        # Execute
        result = await slack_manager.is_job_requested(
            MagicMock(), slack_new_conversation_view
        )

        # Verify: should return False (no repo selected yet)
        assert result is False

        # Verify: Redis set was called to store the user message
        expected_key = f'{SLACK_USER_MSG_KEY_PREFIX}:{slack_new_conversation_view.message_ts}:{slack_new_conversation_view.thread_ts}'
        mock_redis.set.assert_called_once_with(
            expected_key,
            slack_new_conversation_view.user_msg,
            ex=SLACK_USER_MSG_EXPIRATION,
        )
