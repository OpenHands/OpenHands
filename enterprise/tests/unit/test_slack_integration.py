import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import BackgroundTasks
from integrations.slack.slack_manager import (
    SLACK_USER_MSG_EXPIRATION,
    SLACK_USER_MSG_KEY_PREFIX,
    SlackManager,
)
from integrations.slack.slack_view import SlackNewConversationView
from storage.slack_user import SlackUser

from openhands.integrations.service_types import (
    ProviderTimeoutError,
    ProviderType,
    Repository,
)
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
        (
            'help me with repo',
            [],
        ),  # Updated: this pattern is not matched by infer_repo_from_message
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
    """Test the _build_repo_options helper method.

    Note: _build_repo_options always includes the "No Repository" option at the top.
    This is by design for the external_select dropdown.
    """

    def test_build_options_with_repos(self, slack_manager):
        """Test building options from a list of repositories."""

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

        options = slack_manager._build_repo_options(repos)

        # Should have 3 options: "No Repository" + 2 repos
        assert len(options) == 3
        assert options[0]['value'] == '-'
        assert options[0]['text']['text'] == 'No Repository'
        assert options[1]['value'] == 'owner/repo1'
        assert options[2]['value'] == 'owner/repo2'

    def test_build_options_empty_repos(self, slack_manager):
        """Test building options with empty repo list still includes No Repository."""
        options = slack_manager._build_repo_options([])

        # Should have 1 option: just "No Repository"
        assert len(options) == 1
        assert options[0]['value'] == '-'
        assert options[0]['text']['text'] == 'No Repository'

    def test_build_options_truncates_long_names(self, slack_manager):
        """Test that repo names longer than 75 chars are truncated."""

        long_name = 'a' * 100
        repos = [
            Repository(
                id='1',
                full_name=long_name,
                git_provider=ProviderType.GITHUB,
                is_public=True,
            ),
        ]

        options = slack_manager._build_repo_options(repos)

        # First option is "No Repository", second is the repo
        assert len(options) == 2
        # Text should be truncated to 75 chars
        assert len(options[1]['text']['text']) == 75
        # But value should have full name
        assert options[1]['value'] == long_name


class TestSearchRepositories:
    """Test the _search_repositories method with real repository filtering logic."""

    @patch('integrations.slack.slack_manager.ProviderHandler')
    async def test_search_repositories_returns_repos_from_provider(
        self, mock_provider_handler_class, slack_manager, mock_user_auth
    ):
        """Test that _search_repositories returns repositories from the provider."""

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
        mock_provider_handler.search_repositories = AsyncMock(
            return_value=expected_repos
        )
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
        options = slack_manager._build_repo_options(search_results)

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
        options = slack_manager._build_repo_options(search_results)

        # Verify: Only "No Repository" option
        assert len(options) == 1
        assert options[0]['value'] == '-'
        assert options[0]['text']['text'] == 'No Repository'


class TestUserMsgStorage:
    """Test the user message storage for repo selection form flow.

    Note: _store_user_msg_for_form and _retrieve_user_msg_for_form are private methods
    that raise SlackError on failure instead of returning True/False.
    """

    @pytest.mark.parametrize(
        'message_ts,thread_ts,user_msg',
        [
            (
                '1234567890.123456',
                '1234567890.111111',
                'Hello OpenHands, help me with my code',
            ),
            ('1234567890.123456', None, 'Hello OpenHands'),
            ('9999999999.999999', '8888888888.888888', 'Another test message'),
        ],
        ids=['with_thread', 'without_thread', 'different_timestamps'],
    )
    @patch('integrations.slack.slack_manager.sio')
    async def test_store_user_msg_for_form(
        self, mock_sio, slack_manager, message_ts, thread_ts, user_msg
    ):
        """Test storing user message in Redis with various timestamp combinations."""
        mock_redis = AsyncMock()
        mock_sio.manager.redis = mock_redis

        # Should not raise an exception on success
        await slack_manager._store_user_msg_for_form(message_ts, thread_ts, user_msg)

        expected_key = f'{SLACK_USER_MSG_KEY_PREFIX}:{message_ts}:{thread_ts}'
        mock_redis.set.assert_called_once_with(
            expected_key, user_msg, ex=SLACK_USER_MSG_EXPIRATION
        )

    @pytest.mark.parametrize(
        'exception_type,exception_msg',
        [
            (ConnectionError, 'Connection refused'),
            (TimeoutError, 'Redis operation timed out'),
            (Exception, 'Redis internal error'),
        ],
        ids=['connection_error', 'timeout_error', 'generic_exception'],
    )
    @patch('integrations.slack.slack_manager.sio')
    async def test_store_user_msg_for_form_redis_failure(
        self, mock_sio, slack_manager, exception_type, exception_msg
    ):
        """Test that Redis failures during store raise SlackError."""
        from integrations.slack.slack_errors import SlackError, SlackErrorCode

        mock_redis = AsyncMock()
        mock_redis.set.side_effect = exception_type(exception_msg)
        mock_sio.manager.redis = mock_redis

        message_ts = '1234567890.123456'
        thread_ts = '1234567890.111111'
        user_msg = 'Hello OpenHands'

        # Should raise SlackError when Redis fails
        with pytest.raises(SlackError) as exc_info:
            await slack_manager._store_user_msg_for_form(
                message_ts, thread_ts, user_msg
            )

        assert exc_info.value.code == SlackErrorCode.REDIS_STORE_FAILED

    @pytest.mark.parametrize(
        'redis_return_value,expected_result',
        [
            (
                b'Hello OpenHands, help me with my code',
                'Hello OpenHands, help me with my code',
            ),
            ('Hello OpenHands', 'Hello OpenHands'),  # String instead of bytes
        ],
        ids=['bytes_response', 'string_response'],
    )
    @patch('integrations.slack.slack_manager.sio')
    async def test_retrieve_user_msg_for_form(
        self, mock_sio, slack_manager, redis_return_value, expected_result
    ):
        """Test retrieving user message from Redis with various response types."""
        mock_redis = AsyncMock()
        mock_redis.get.return_value = redis_return_value
        mock_sio.manager.redis = mock_redis

        message_ts = '1234567890.123456'
        thread_ts = '1234567890.111111'

        result = await slack_manager._retrieve_user_msg_for_form(message_ts, thread_ts)

        expected_key = f'{SLACK_USER_MSG_KEY_PREFIX}:{message_ts}:{thread_ts}'
        mock_redis.get.assert_called_once_with(expected_key)
        assert result == expected_result

    @patch('integrations.slack.slack_manager.sio')
    async def test_retrieve_user_msg_for_form_key_not_found(
        self, mock_sio, slack_manager
    ):
        """Test that missing key raises SlackError with SESSION_EXPIRED."""
        from integrations.slack.slack_errors import SlackError, SlackErrorCode

        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        mock_sio.manager.redis = mock_redis

        message_ts = '1234567890.123456'
        thread_ts = '1234567890.111111'

        # Should raise SlackError when key not found
        with pytest.raises(SlackError) as exc_info:
            await slack_manager._retrieve_user_msg_for_form(message_ts, thread_ts)

        assert exc_info.value.code == SlackErrorCode.SESSION_EXPIRED

    @pytest.mark.parametrize(
        'exception_type,exception_msg',
        [
            (ConnectionError, 'Connection refused'),
            (TimeoutError, 'Redis operation timed out'),
        ],
        ids=['connection_error', 'timeout_error'],
    )
    @patch('integrations.slack.slack_manager.sio')
    async def test_retrieve_user_msg_for_form_redis_failure(
        self, mock_sio, slack_manager, exception_type, exception_msg
    ):
        """Test that Redis failures during retrieve raise SlackError."""
        from integrations.slack.slack_errors import SlackError, SlackErrorCode

        mock_redis = AsyncMock()
        mock_redis.get.side_effect = exception_type(exception_msg)
        mock_sio.manager.redis = mock_redis

        message_ts = '1234567890.123456'
        thread_ts = '1234567890.111111'

        # Should raise SlackError when Redis fails
        with pytest.raises(SlackError) as exc_info:
            await slack_manager._retrieve_user_msg_for_form(message_ts, thread_ts)

        assert exc_info.value.code == SlackErrorCode.REDIS_RETRIEVE_FAILED


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


class TestOnOptionsLoadEndpoint:
    """Test the /on-options-load endpoint for external_select repo search."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock Request object."""
        request = MagicMock()
        request.headers = {
            'X-Slack-Request-Timestamp': '1234567890',
            'X-Slack-Signature': 'v0=test_signature',
        }
        return request

    @pytest.fixture
    def valid_block_suggestion_payload(self):
        """Create a valid block_suggestion payload from Slack."""
        return {
            'type': 'block_suggestion',
            'user': {'id': 'U1234567890'},
            'value': 'test-query',
            'team': {'id': 'T1234567890'},
            'container': {'channel_id': 'C1234567890'},
        }

    @pytest.fixture
    def background_tasks(self):
        """Create mock BackgroundTasks."""
        return MagicMock(spec=BackgroundTasks)

    @pytest.mark.asyncio
    @patch('server.routes.integration.slack.SLACK_WEBHOOKS_ENABLED', False)
    async def test_on_options_load_disabled_returns_empty_options(
        self, mock_request, background_tasks
    ):
        """Test that when webhooks are disabled, empty options are returned."""
        from server.routes.integration.slack import on_options_load

        response = await on_options_load(mock_request, background_tasks)

        assert response.status_code == 200
        body = json.loads(response.body)
        assert body == {'options': []}

    @pytest.mark.asyncio
    @patch('server.routes.integration.slack.SLACK_WEBHOOKS_ENABLED', True)
    async def test_on_options_load_no_payload_returns_empty_options(
        self, mock_request, background_tasks
    ):
        """Test that when no payload is in request, empty options are returned."""
        from server.routes.integration.slack import on_options_load

        mock_request.body = AsyncMock(return_value=b'')
        mock_form = MagicMock()
        mock_form.get.return_value = None
        mock_request.form = AsyncMock(return_value=mock_form)

        response = await on_options_load(mock_request, background_tasks)

        assert response.status_code == 200
        body = json.loads(response.body)
        assert body == {'options': []}

    @pytest.mark.asyncio
    @patch('server.routes.integration.slack.SLACK_WEBHOOKS_ENABLED', True)
    @patch('server.routes.integration.slack.signature_verifier')
    async def test_on_options_load_invalid_signature_raises_403(
        self,
        mock_signature_verifier,
        mock_request,
        background_tasks,
        valid_block_suggestion_payload,
    ):
        """Test that invalid Slack signature raises 403 HTTPException."""
        from fastapi import HTTPException
        from server.routes.integration.slack import on_options_load

        payload_str = json.dumps(valid_block_suggestion_payload)
        mock_request.body = AsyncMock(return_value=payload_str.encode())
        mock_form = MagicMock()
        mock_form.get.return_value = payload_str
        mock_request.form = AsyncMock(return_value=mock_form)

        mock_signature_verifier.is_valid.return_value = False

        with pytest.raises(HTTPException) as exc_info:
            await on_options_load(mock_request, background_tasks)

        assert exc_info.value.status_code == 403
        assert exc_info.value.detail == 'invalid_request'

    @pytest.mark.asyncio
    @patch('server.routes.integration.slack.SLACK_WEBHOOKS_ENABLED', True)
    @patch('server.routes.integration.slack.signature_verifier')
    async def test_on_options_load_wrong_payload_type_returns_empty_options(
        self, mock_signature_verifier, mock_request, background_tasks
    ):
        """Test that non-block_suggestion payload returns empty options."""
        from server.routes.integration.slack import on_options_load

        payload = {
            'type': 'interactive_message',  # Wrong type
            'user': {'id': 'U1234567890'},
        }
        payload_str = json.dumps(payload)
        mock_request.body = AsyncMock(return_value=payload_str.encode())
        mock_form = MagicMock()
        mock_form.get.return_value = payload_str
        mock_request.form = AsyncMock(return_value=mock_form)

        mock_signature_verifier.is_valid.return_value = True

        response = await on_options_load(mock_request, background_tasks)

        assert response.status_code == 200
        body = json.loads(response.body)
        assert body == {'options': []}

    @pytest.mark.asyncio
    @patch('server.routes.integration.slack.SLACK_WEBHOOKS_ENABLED', True)
    @patch('server.routes.integration.slack.signature_verifier')
    @patch('server.routes.integration.slack.slack_manager')
    async def test_on_options_load_unauthenticated_user_returns_empty_options(
        self,
        mock_slack_manager,
        mock_signature_verifier,
        mock_request,
        background_tasks,
        valid_block_suggestion_payload,
    ):
        """Test that unauthenticated users get empty options and linking message is queued."""
        from server.routes.integration.slack import on_options_load

        payload_str = json.dumps(valid_block_suggestion_payload)
        mock_request.body = AsyncMock(return_value=payload_str.encode())
        mock_form = MagicMock()
        mock_form.get.return_value = payload_str
        mock_request.form = AsyncMock(return_value=mock_form)

        mock_signature_verifier.is_valid.return_value = True
        mock_slack_manager.authenticate_user = AsyncMock(return_value=(None, None))

        response = await on_options_load(mock_request, background_tasks)

        assert response.status_code == 200
        body = json.loads(response.body)
        assert body == {'options': []}

        # Verify background task was queued for account linking message
        background_tasks.add_task.assert_called_once()

    @pytest.mark.asyncio
    @patch('server.routes.integration.slack.SLACK_WEBHOOKS_ENABLED', True)
    @patch('server.routes.integration.slack.signature_verifier')
    @patch('server.routes.integration.slack.slack_manager')
    async def test_on_options_load_successful_search_with_repos(
        self,
        mock_slack_manager,
        mock_signature_verifier,
        mock_request,
        background_tasks,
        valid_block_suggestion_payload,
        mock_slack_user,
        mock_user_auth,
    ):
        """Test successful repository search returns properly formatted options.

        This test verifies the endpoint calls search_repos_for_slack with the
        correct parameters. The actual formatting is tested in TestBuildRepoOptions.
        """
        from server.routes.integration.slack import on_options_load

        payload_str = json.dumps(valid_block_suggestion_payload)
        mock_request.body = AsyncMock(return_value=payload_str.encode())
        mock_form = MagicMock()
        mock_form.get.return_value = payload_str
        mock_request.form = AsyncMock(return_value=mock_form)

        mock_signature_verifier.is_valid.return_value = True
        mock_slack_manager.authenticate_user = AsyncMock(
            return_value=(mock_slack_user, mock_user_auth)
        )

        # Expected options from search_repos_for_slack
        expected_options = [
            {'text': {'type': 'plain_text', 'text': 'No Repository'}, 'value': '-'},
            {
                'text': {'type': 'plain_text', 'text': 'owner/repo1'},
                'value': 'owner/repo1',
            },
            {
                'text': {'type': 'plain_text', 'text': 'owner/repo2'},
                'value': 'owner/repo2',
            },
        ]
        mock_slack_manager.search_repos_for_slack = AsyncMock(
            return_value=expected_options
        )

        response = await on_options_load(mock_request, background_tasks)

        assert response.status_code == 200
        body = json.loads(response.body)
        assert body == {'options': expected_options}

        # Verify search_repos_for_slack was called with correct parameters
        mock_slack_manager.search_repos_for_slack.assert_called_once_with(
            mock_user_auth, query='test-query', per_page=20
        )

    @pytest.mark.asyncio
    @patch('server.routes.integration.slack.SLACK_WEBHOOKS_ENABLED', True)
    @patch('server.routes.integration.slack.signature_verifier')
    @patch('server.routes.integration.slack.slack_manager')
    async def test_on_options_load_empty_query_search(
        self,
        mock_slack_manager,
        mock_signature_verifier,
        mock_request,
        background_tasks,
        mock_slack_user,
        mock_user_auth,
    ):
        """Test search with empty query (min_query_length: 0 in external_select)."""
        from server.routes.integration.slack import on_options_load

        # Payload with empty value (no search text entered yet)
        payload = {
            'type': 'block_suggestion',
            'user': {'id': 'U1234567890'},
            'value': '',  # Empty search
            'team': {'id': 'T1234567890'},
            'container': {'channel_id': 'C1234567890'},
        }
        payload_str = json.dumps(payload)
        mock_request.body = AsyncMock(return_value=payload_str.encode())
        mock_form = MagicMock()
        mock_form.get.return_value = payload_str
        mock_request.form = AsyncMock(return_value=mock_form)

        mock_signature_verifier.is_valid.return_value = True
        mock_slack_manager.authenticate_user = AsyncMock(
            return_value=(mock_slack_user, mock_user_auth)
        )
        mock_slack_manager.search_repos_for_slack = AsyncMock(
            return_value=[
                {'text': {'type': 'plain_text', 'text': 'No Repository'}, 'value': '-'}
            ]
        )

        response = await on_options_load(mock_request, background_tasks)

        assert response.status_code == 200

        # Verify search_repos_for_slack was called with empty query
        mock_slack_manager.search_repos_for_slack.assert_called_once_with(
            mock_user_auth, query='', per_page=20
        )

    @pytest.mark.asyncio
    @patch('server.routes.integration.slack.SLACK_WEBHOOKS_ENABLED', True)
    @patch('server.routes.integration.slack.signature_verifier')
    @patch('server.routes.integration.slack.slack_manager')
    async def test_on_options_load_search_exception_returns_empty_options(
        self,
        mock_slack_manager,
        mock_signature_verifier,
        mock_request,
        background_tasks,
        valid_block_suggestion_payload,
        mock_slack_user,
        mock_user_auth,
    ):
        """Test that when search raises an exception, empty options are returned gracefully."""
        from server.routes.integration.slack import on_options_load

        payload_str = json.dumps(valid_block_suggestion_payload)
        mock_request.body = AsyncMock(return_value=payload_str.encode())
        mock_form = MagicMock()
        mock_form.get.return_value = payload_str
        mock_request.form = AsyncMock(return_value=mock_form)

        mock_signature_verifier.is_valid.return_value = True
        mock_slack_manager.authenticate_user = AsyncMock(
            return_value=(mock_slack_user, mock_user_auth)
        )
        # Simulate search error (e.g., provider timeout)
        mock_slack_manager.search_repos_for_slack = AsyncMock(
            side_effect=Exception('GitHub API timeout')
        )
        mock_slack_manager.handle_slack_error = AsyncMock()

        response = await on_options_load(mock_request, background_tasks)

        assert response.status_code == 200
        body = json.loads(response.body)
        assert body == {'options': []}

    @pytest.mark.asyncio
    @patch('server.routes.integration.slack.SLACK_WEBHOOKS_ENABLED', True)
    @patch('server.routes.integration.slack.signature_verifier')
    @patch('server.routes.integration.slack.slack_manager')
    async def test_on_options_load_missing_value_field_defaults_to_empty(
        self,
        mock_slack_manager,
        mock_signature_verifier,
        mock_request,
        background_tasks,
        mock_slack_user,
        mock_user_auth,
    ):
        """Test that missing 'value' field in payload defaults to empty string."""
        from server.routes.integration.slack import on_options_load

        # Payload without 'value' key
        payload = {
            'type': 'block_suggestion',
            'user': {'id': 'U1234567890'},
            # 'value' is missing
            'team': {'id': 'T1234567890'},
            'container': {'channel_id': 'C1234567890'},
        }
        payload_str = json.dumps(payload)
        mock_request.body = AsyncMock(return_value=payload_str.encode())
        mock_form = MagicMock()
        mock_form.get.return_value = payload_str
        mock_request.form = AsyncMock(return_value=mock_form)

        mock_signature_verifier.is_valid.return_value = True
        mock_slack_manager.authenticate_user = AsyncMock(
            return_value=(mock_slack_user, mock_user_auth)
        )
        mock_slack_manager.search_repos_for_slack = AsyncMock(return_value=[])

        response = await on_options_load(mock_request, background_tasks)

        assert response.status_code == 200

        # Should default to empty string for search
        mock_slack_manager.search_repos_for_slack.assert_called_once_with(
            mock_user_auth, query='', per_page=20
        )

    @pytest.mark.asyncio
    @patch('server.routes.integration.slack.SLACK_WEBHOOKS_ENABLED', True)
    @patch('server.routes.integration.slack.signature_verifier')
    @patch('server.routes.integration.slack.slack_manager')
    async def test_on_options_load_truncates_long_repo_names(
        self,
        mock_slack_manager,
        mock_signature_verifier,
        mock_request,
        background_tasks,
        valid_block_suggestion_payload,
        mock_slack_user,
        mock_user_auth,
    ):
        """Test that options with long repo names are properly handled.

        Note: The actual truncation logic is tested in TestBuildRepoOptions.
        This test just verifies the endpoint correctly passes through the formatted options.
        """
        from server.routes.integration.slack import on_options_load

        payload_str = json.dumps(valid_block_suggestion_payload)
        mock_request.body = AsyncMock(return_value=payload_str.encode())
        mock_form = MagicMock()
        mock_form.get.return_value = payload_str
        mock_request.form = AsyncMock(return_value=mock_form)

        mock_signature_verifier.is_valid.return_value = True
        mock_slack_manager.authenticate_user = AsyncMock(
            return_value=(mock_slack_user, mock_user_auth)
        )

        # Mock the formatted options that would come from search_repos_for_slack
        expected_options = [
            {'text': {'type': 'plain_text', 'text': 'No Repository'}, 'value': '-'},
            {
                'text': {
                    'type': 'plain_text',
                    'text': 'verylongorganizationname/very-long-repository-name-tha',
                },
                'value': 'verylongorganizationname/very-long-repository-name-that-exceeds-normal-length',
            },
        ]
        mock_slack_manager.search_repos_for_slack = AsyncMock(
            return_value=expected_options
        )

        response = await on_options_load(mock_request, background_tasks)

        assert response.status_code == 200
        body = json.loads(response.body)
        assert 'options' in body
        assert len(body['options']) == 2


class TestHandleSlackError:
    """Test the handle_slack_error method on SlackManager.

    Note: Error handling now goes through SlackManager.handle_slack_error method
    instead of a standalone _send_slack_error function.
    """

    @pytest.mark.asyncio
    @patch('integrations.slack.slack_manager.SlackMessageView.from_payload')
    @patch.object(SlackManager, 'send_message', new_callable=AsyncMock)
    async def test_handle_slack_error_success(
        self, mock_send_message, mock_from_payload, slack_manager
    ):
        """Test successful sending of error message to Slack user."""
        from integrations.slack.slack_errors import SlackError, SlackErrorCode

        payload = {
            'team': {'id': 'T1234567890'},
            'container': {'channel_id': 'C1234567890'},
            'user': {'id': 'U1234567890'},
        }
        error = SlackError(
            SlackErrorCode.USER_NOT_AUTHENTICATED,
            message_kwargs={'login_link': 'https://test.link'},
            log_context={'slack_user_id': 'U1234567890'},
        )

        # Mock the view creation
        mock_view = MagicMock()
        mock_view.to_log_context.return_value = {}
        mock_from_payload.return_value = mock_view

        await slack_manager.handle_slack_error(payload, error)

        mock_send_message.assert_called_once()
        # Verify ephemeral=True is passed
        call_args = mock_send_message.call_args
        assert call_args.kwargs.get('ephemeral') is True

    @pytest.mark.asyncio
    @patch('integrations.slack.slack_manager.SlackMessageView.from_payload')
    @patch.object(SlackManager, 'send_message', new_callable=AsyncMock)
    async def test_handle_slack_error_no_view(
        self, mock_send_message, mock_from_payload, slack_manager
    ):
        """Test handling when view creation fails (returns None)."""
        from integrations.slack.slack_errors import SlackError, SlackErrorCode

        payload = {
            'team': {},  # Invalid - missing id
            'container': {},
            'user': {},
        }
        error = SlackError(
            SlackErrorCode.SESSION_EXPIRED,
            log_context={'test': 'context'},
        )

        # Mock view creation returning None
        mock_from_payload.return_value = None

        # Should handle gracefully without raising
        await slack_manager.handle_slack_error(payload, error)

        # send_message should not be called when view creation fails
        mock_send_message.assert_not_called()

    @pytest.mark.asyncio
    @patch('integrations.slack.slack_manager.SlackMessageView.from_payload')
    @patch.object(SlackManager, 'send_message', new_callable=AsyncMock)
    async def test_handle_slack_error_various_error_codes(
        self, mock_send_message, mock_from_payload, slack_manager
    ):
        """Test that different error codes produce appropriate messages."""
        from integrations.slack.slack_errors import SlackError, SlackErrorCode

        payload = {
            'team': {'id': 'T1234567890'},
            'container': {'channel_id': 'C1234567890'},
            'user': {'id': 'U1234567890'},
        }

        # Mock the view creation
        mock_view = MagicMock()
        mock_view.to_log_context.return_value = {}
        mock_from_payload.return_value = mock_view

        # Test different error codes
        error_codes = [
            SlackErrorCode.SESSION_EXPIRED,
            SlackErrorCode.PROVIDER_TIMEOUT,
            SlackErrorCode.REDIS_STORE_FAILED,
            SlackErrorCode.UNEXPECTED_ERROR,
        ]

        for code in error_codes:
            mock_send_message.reset_mock()
            error = SlackError(code, log_context={'test': 'context'})

            await slack_manager.handle_slack_error(payload, error)

            mock_send_message.assert_called_once()
            call_args = mock_send_message.call_args
            message = call_args.args[0]
            # Verify message is not empty
            assert message
            assert isinstance(message, str)
