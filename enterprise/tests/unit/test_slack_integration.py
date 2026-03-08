from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openhands.integrations.service_types import ProviderTimeoutError
from openhands.server.user_auth.user_auth import UserAuth

from integrations.slack.slack_manager import (
    SLACK_USER_MSG_EXPIRATION,
    SLACK_USER_MSG_KEY_PREFIX,
    SlackManager,
)
from integrations.slack.slack_view import SlackNewConversationView
from storage.slack_user import SlackUser


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
    auth.get_provider_tokens = AsyncMock(return_value={'github': 'test-token'})
    auth.get_access_token = AsyncMock(return_value='access-token')
    auth.get_user_id = AsyncMock(return_value='user-123')
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
        ('OpenHands/Openhands', ['OpenHands/Openhands']),
        ('help me with repo', []),  # Updated: this pattern is not matched by infer_repo_from_message
        ('use hello world', []),
    ],
)
def test_infer_repo_from_message(message, expected):
    # Test the infer_repo_from_message function from utils
    from integrations.utils import infer_repo_from_message

    result = infer_repo_from_message(message)
    assert result == expected


class TestRepoVerificationHandling:
    """Test repo verification handling for Slack integration."""

    @patch('integrations.slack.slack_manager.sio')
    @patch('integrations.slack.slack_manager.ProviderHandler')
    @patch.object(SlackManager, 'send_message', new_callable=AsyncMock)
    async def test_timeout_during_verification_shows_selector(
        self,
        mock_send_message,
        mock_provider_handler_class,
        mock_sio,
        slack_manager,
        slack_new_conversation_view,
    ):
        """Test that when repo verification times out, selector is shown."""
        # Setup Redis mock
        mock_redis = AsyncMock()
        mock_sio.manager.redis = mock_redis

        # Setup: Modify message to include exactly one repo reference to trigger verification
        slack_new_conversation_view.user_msg = 'Help me with OpenHands/OpenHands repo'

        # Setup: verify_repo_provider raises ProviderTimeoutError
        mock_provider_handler = MagicMock()
        mock_provider_handler.verify_repo_provider = AsyncMock(
            side_effect=ProviderTimeoutError(
                'github API request timed out: ConnectTimeout'
            )
        )
        mock_provider_handler_class.return_value = mock_provider_handler

        # Execute
        result = await slack_manager.is_job_requested(
            MagicMock(), slack_new_conversation_view
        )

        # Verify: should return False (job not started, but selector is shown)
        assert result is False

        # Verify: send_message was called once (for repo selector)
        mock_send_message.assert_called_once()
        call_args = mock_send_message.call_args
        selector_message = call_args[0][0]
        assert isinstance(selector_message, dict)
        assert selector_message.get('text') == 'Choose a Repository:'

    @patch('integrations.slack.slack_manager.sio')
    @patch.object(SlackManager, 'send_message', new_callable=AsyncMock)
    async def test_no_repo_mentioned_shows_external_selector(
        self,
        mock_send_message,
        mock_sio,
        slack_manager,
        slack_new_conversation_view,
    ):
        """Test that when no repo is mentioned, external_select repo selector is shown."""
        # Setup Redis mock
        mock_redis = AsyncMock()
        mock_sio.manager.redis = mock_redis

        # Setup: user message without any repo mention
        slack_new_conversation_view.user_msg = 'Hello, can you help me?'

        # Execute
        result = await slack_manager.is_job_requested(
            MagicMock(), slack_new_conversation_view
        )

        # Verify: should return False (no repo selected yet)
        assert result is False

        # Verify: send_message was called (for repo selector)
        mock_send_message.assert_called_once()
        call_args = mock_send_message.call_args

        # Should be the repo selection form with external_select
        message = call_args[0][0]
        assert isinstance(message, dict)
        assert message.get('text') == 'Choose a Repository:'
        # Verify it's using external_select
        blocks = message.get('blocks', [])
        actions_block = next((b for b in blocks if b.get('type') == 'actions'), None)
        assert actions_block is not None
        elements = actions_block.get('elements', [])
        assert len(elements) > 0
        assert elements[0].get('type') == 'external_select'

    @patch('integrations.slack.slack_manager.sio')
    @patch('integrations.slack.slack_manager.ProviderHandler')
    @patch.object(SlackManager, 'send_message', new_callable=AsyncMock)
    async def test_verified_repo_starts_job(
        self,
        mock_send_message,
        mock_provider_handler_class,
        mock_sio,
        slack_manager,
        slack_new_conversation_view,
    ):
        """Test that when repo is successfully verified, job starts without selector."""
        from openhands.integrations.service_types import ProviderType, Repository

        # Setup Redis mock
        mock_redis = AsyncMock()
        mock_sio.manager.redis = mock_redis

        # Setup: Modify message to include exactly one repo reference
        slack_new_conversation_view.user_msg = 'Help me with OpenHands/OpenHands repo'

        # Setup: verify_repo_provider returns a valid repo
        mock_repo = Repository(
            id='123',
            full_name='OpenHands/OpenHands',
            git_provider=ProviderType.GITHUB,
            is_public=True,
        )
        mock_provider_handler = MagicMock()
        mock_provider_handler.verify_repo_provider = AsyncMock(return_value=mock_repo)
        mock_provider_handler_class.return_value = mock_provider_handler

        # Execute
        result = await slack_manager.is_job_requested(
            MagicMock(), slack_new_conversation_view
        )

        # Verify: should return True (job started)
        assert result is True

        # Verify: send_message was NOT called (no selector needed)
        mock_send_message.assert_not_called()

        # Verify: selected_repo was set
        assert slack_new_conversation_view.selected_repo == 'OpenHands/OpenHands'


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


class TestSearchRepositories:
    """Test the _search_repositories method with real repository filtering logic."""

    @patch('integrations.slack.slack_manager.ProviderHandler')
    async def test_search_repositories_returns_repos_from_provider(
        self, mock_provider_handler_class, slack_manager, mock_user_auth
    ):
        """Test that _search_repositories returns repositories from the provider."""
        from openhands.integrations.service_types import ProviderType, Repository

        # Setup: Create real Repository objects
        expected_repos = [
            Repository(
                id='1',
                full_name='owner/frontend-app',
                git_provider=ProviderType.GITHUB,
                is_public=True,
            ),
            Repository(
                id='2',
                full_name='owner/backend-api',
                git_provider=ProviderType.GITHUB,
                is_public=False,
            ),
            Repository(
                id='3',
                full_name='owner/shared-lib',
                git_provider=ProviderType.GITHUB,
                is_public=True,
            ),
        ]

        # Setup: Mock provider handler to return real repos
        mock_provider_handler = MagicMock()
        mock_provider_handler.search_repositories = AsyncMock(return_value=expected_repos)
        mock_provider_handler_class.return_value = mock_provider_handler

        # Setup: Mock user_auth to return valid tokens
        mock_user_auth.get_provider_tokens = AsyncMock(
            return_value={'github': 'test-token'}
        )
        mock_user_auth.get_access_token = AsyncMock(return_value='access-token')
        mock_user_auth.get_user_id = AsyncMock(return_value='user-123')

        # Execute: Search with a query
        result = await slack_manager._search_repositories(
            mock_user_auth, query='frontend', per_page=20
        )

        # Verify: The correct parameters were passed to search_repositories
        mock_provider_handler.search_repositories.assert_called_once()
        call_kwargs = mock_provider_handler.search_repositories.call_args[1]
        assert call_kwargs['query'] == 'frontend'
        assert call_kwargs['per_page'] == 20
        assert call_kwargs['sort'] == 'pushed'
        assert call_kwargs['order'] == 'desc'

        # Verify: All repos are returned
        assert len(result) == 3
        assert result[0].full_name == 'owner/frontend-app'
        assert result[1].full_name == 'owner/backend-api'
        assert result[2].full_name == 'owner/shared-lib'

    @patch('integrations.slack.slack_manager.ProviderHandler')
    async def test_search_repositories_returns_empty_when_no_tokens(
        self, mock_provider_handler_class, slack_manager, mock_user_auth
    ):
        """Test that _search_repositories returns empty list when user has no provider tokens."""
        # Setup: User has no provider tokens
        mock_user_auth.get_provider_tokens = AsyncMock(return_value=None)

        # Execute
        result = await slack_manager._search_repositories(mock_user_auth, query='test')

        # Verify: Returns empty list, doesn't call ProviderHandler
        assert result == []
        mock_provider_handler_class.assert_not_called()

    @patch('integrations.slack.slack_manager.ProviderHandler')
    async def test_search_and_build_options_integration(
        self, mock_provider_handler_class, slack_manager, mock_user_auth
    ):
        """Test the full flow: search repositories and build options for Slack.

        This exercises the full code path from search → filter → options building.
        """
        from openhands.integrations.service_types import ProviderType, Repository

        # Setup: Create a realistic repository list
        repos = [
            Repository(
                id='1',
                full_name='myorg/react-dashboard',
                git_provider=ProviderType.GITHUB,
                is_public=True,
            ),
            Repository(
                id='2',
                full_name='myorg/python-api',
                git_provider=ProviderType.GITHUB,
                is_public=False,
            ),
            Repository(
                id='3',
                full_name='myorg/docs-site',
                git_provider=ProviderType.GITHUB,
                is_public=True,
            ),
        ]

        mock_provider_handler = MagicMock()
        mock_provider_handler.search_repositories = AsyncMock(return_value=repos)
        mock_provider_handler_class.return_value = mock_provider_handler

        mock_user_auth.get_provider_tokens = AsyncMock(
            return_value={'github': 'test-token'}
        )
        mock_user_auth.get_access_token = AsyncMock(return_value='access-token')
        mock_user_auth.get_user_id = AsyncMock(return_value='user-123')

        # Execute: Search and build options (simulating what slack route does)
        search_results = await slack_manager._search_repositories(
            mock_user_auth, query='', per_page=100
        )
        options = slack_manager._build_repo_options(search_results, include_no_repo=True)

        # Verify: Options are correctly built from search results
        assert len(options) == 4  # "No Repository" + 3 repos

        # First option should be "No Repository"
        assert options[0]['value'] == '-'
        assert options[0]['text']['text'] == 'No Repository'

        # Remaining options should be the repos in order
        assert options[1]['value'] == 'myorg/react-dashboard'
        assert options[1]['text']['text'] == 'myorg/react-dashboard'
        assert options[2]['value'] == 'myorg/python-api'
        assert options[3]['value'] == 'myorg/docs-site'

    @patch('integrations.slack.slack_manager.ProviderHandler')
    async def test_search_with_empty_results_builds_no_repo_only_option(
        self, mock_provider_handler_class, slack_manager, mock_user_auth
    ):
        """Test that when search returns no results, only 'No Repository' option is shown."""
        # Setup: No matching repos
        mock_provider_handler = MagicMock()
        mock_provider_handler.search_repositories = AsyncMock(return_value=[])
        mock_provider_handler_class.return_value = mock_provider_handler

        mock_user_auth.get_provider_tokens = AsyncMock(
            return_value={'github': 'test-token'}
        )
        mock_user_auth.get_access_token = AsyncMock(return_value='access-token')
        mock_user_auth.get_user_id = AsyncMock(return_value='user-123')

        # Execute
        search_results = await slack_manager._search_repositories(
            mock_user_auth, query='nonexistent-repo', per_page=100
        )
        options = slack_manager._build_repo_options(search_results, include_no_repo=True)

        # Verify: Only "No Repository" option
        assert len(options) == 1
        assert options[0]['value'] == '-'
        assert options[0]['text']['text'] == 'No Repository'


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
    async def test_stores_user_msg_when_showing_repo_selector(
        self,
        mock_send_message,
        mock_sio,
        slack_manager,
        slack_new_conversation_view,
    ):
        """Test that user_msg is stored in Redis when repo selector is shown."""
        mock_redis = AsyncMock()
        mock_sio.manager.redis = mock_redis

        # Setup: user message without any repo mention (no repo inferred)
        slack_new_conversation_view.user_msg = 'Hello, can you help me?'

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
