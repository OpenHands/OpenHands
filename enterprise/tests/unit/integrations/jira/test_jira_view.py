"""
Tests for Jira view classes and factory.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from integrations.jira.jira_payload import (
    JiraEventType,
    JiraPayloadError,
    JiraPayloadParser,
    JiraPayloadSkipped,
    JiraPayloadSuccess,
)
from integrations.jira.jira_types import RepositoryNotFoundError, StartingConvoException
from integrations.jira.jira_view import (
    JiraFactory,
    JiraNewConversationView,
)


class TestJiraNewConversationView:
    """Tests for JiraNewConversationView"""

    @pytest.mark.asyncio
    async def test_get_issue_details_success(
        self, new_conversation_view, sample_jira_workspace
    ):
        """Test successful issue details retrieval."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'fields': {'summary': 'Test Issue', 'description': 'Test description'}
        }
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            title, description = await new_conversation_view.get_issue_details()

            assert title == 'Test Issue'
            assert description == 'Test description'

    @pytest.mark.asyncio
    async def test_get_issue_details_cached(self, new_conversation_view):
        """Test issue details are cached after first call."""
        new_conversation_view._issue_title = 'Cached Title'
        new_conversation_view._issue_description = 'Cached Description'

        title, description = await new_conversation_view.get_issue_details()

        assert title == 'Cached Title'
        assert description == 'Cached Description'

    @pytest.mark.asyncio
    async def test_get_issue_details_no_title(self, new_conversation_view):
        """Test issue details with no title raises error."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'fields': {'summary': '', 'description': 'Test description'}
        }
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(StartingConvoException, match='does not have a title'):
                await new_conversation_view.get_issue_details()

    @pytest.mark.asyncio
    async def test_get_instructions(self, new_conversation_view, mock_jinja_env):
        """Test _get_instructions method fetches issue details."""
        new_conversation_view._issue_title = 'Test Issue'
        new_conversation_view._issue_description = 'This is a test issue'

        instructions, user_msg = await new_conversation_view._get_instructions(
            mock_jinja_env
        )

        assert instructions == 'Test Jira instructions template'
        assert 'TEST-123' in user_msg
        assert 'Test Issue' in user_msg

    @pytest.mark.asyncio
    @patch('integrations.jira.jira_view.create_new_conversation')
    @patch('integrations.jira.jira_view.integration_store')
    async def test_create_or_update_conversation_success(
        self,
        mock_store,
        mock_create_conversation,
        new_conversation_view,
        mock_jinja_env,
        mock_agent_loop_info,
    ):
        """Test successful conversation creation"""
        new_conversation_view._issue_title = 'Test Issue'
        new_conversation_view._issue_description = 'Test description'
        mock_create_conversation.return_value = mock_agent_loop_info
        mock_store.create_conversation = AsyncMock()

        result = await new_conversation_view.create_or_update_conversation(
            mock_jinja_env
        )

        assert result == 'conv-123'
        mock_create_conversation.assert_called_once()
        mock_store.create_conversation.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_or_update_conversation_no_repo(
        self, new_conversation_view, mock_jinja_env
    ):
        """Test conversation creation without selected repo"""
        new_conversation_view.selected_repo = None

        with pytest.raises(StartingConvoException, match='No repository selected'):
            await new_conversation_view.create_or_update_conversation(mock_jinja_env)

    def test_get_response_msg(self, new_conversation_view):
        """Test get_response_msg method"""
        response = new_conversation_view.get_response_msg()

        assert "I'm on it!" in response
        assert 'Test User' in response
        assert 'track my progress here' in response
        assert 'conv-123' in response


class TestJiraFactory:
    """Tests for JiraFactory"""

    @pytest.mark.asyncio
    @patch('integrations.jira.jira_view.JiraFactory._create_provider_handler')
    @patch('integrations.jira.jira_view.infer_repo_from_message')
    async def test_create_view_success(
        self,
        mock_infer_repos,
        mock_create_handler,
        sample_webhook_payload,
        sample_user_auth,
        sample_jira_user,
        sample_jira_workspace,
        sample_repositories,
    ):
        """Test factory creating view with repo selection."""
        # Setup mock provider handler
        mock_handler = MagicMock()
        mock_handler.verify_repo_provider = AsyncMock(
            return_value=sample_repositories[0]
        )
        mock_create_handler.return_value = mock_handler

        # Mock repo inference to return a repo name
        mock_infer_repos.return_value = ['test/repo1']

        mock_response = MagicMock()
        mock_response.json.return_value = {
            'fields': {'summary': 'Test Issue', 'description': 'Test description'}
        }
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            view = await JiraFactory.create_view(
                payload=sample_webhook_payload,
                workspace=sample_jira_workspace,
                user=sample_jira_user,
                user_auth=sample_user_auth,
                decrypted_api_key='test_api_key',
            )

            assert isinstance(view, JiraNewConversationView)
            assert view.selected_repo == 'test/repo1'
            mock_handler.verify_repo_provider.assert_called_once_with('test/repo1')

    @pytest.mark.asyncio
    @patch('integrations.jira.jira_view.JiraFactory._create_provider_handler')
    @patch('integrations.jira.jira_view.infer_repo_from_message')
    async def test_create_view_no_repo_in_text(
        self,
        mock_infer_repos,
        mock_create_handler,
        sample_webhook_payload,
        sample_user_auth,
        sample_jira_user,
        sample_jira_workspace,
    ):
        """Test factory raises error when no repo mentioned in text."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        # No repos found in text
        mock_infer_repos.return_value = []

        mock_response = MagicMock()
        mock_response.json.return_value = {
            'fields': {'summary': 'Test Issue', 'description': 'Test description'}
        }
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(
                RepositoryNotFoundError, match='Could not determine which repository'
            ):
                await JiraFactory.create_view(
                    payload=sample_webhook_payload,
                    workspace=sample_jira_workspace,
                    user=sample_jira_user,
                    user_auth=sample_user_auth,
                    decrypted_api_key='test_api_key',
                )

    @pytest.mark.asyncio
    @patch('integrations.jira.jira_view.JiraFactory._create_provider_handler')
    @patch('integrations.jira.jira_view.infer_repo_from_message')
    async def test_create_view_repo_verification_fails(
        self,
        mock_infer_repos,
        mock_create_handler,
        sample_webhook_payload,
        sample_user_auth,
        sample_jira_user,
        sample_jira_workspace,
    ):
        """Test factory raises error when repo verification fails."""
        mock_handler = MagicMock()
        mock_handler.verify_repo_provider = AsyncMock(
            side_effect=Exception('Repository not found')
        )
        mock_create_handler.return_value = mock_handler

        # Repos found in text but verification fails
        mock_infer_repos.return_value = ['test/repo1', 'test/repo2']

        mock_response = MagicMock()
        mock_response.json.return_value = {
            'fields': {'summary': 'Test Issue', 'description': 'Test description'}
        }
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(
                RepositoryNotFoundError,
                match='Could not access any of the mentioned repositories',
            ):
                await JiraFactory.create_view(
                    payload=sample_webhook_payload,
                    workspace=sample_jira_workspace,
                    user=sample_jira_user,
                    user_auth=sample_user_auth,
                    decrypted_api_key='test_api_key',
                )

    @pytest.mark.asyncio
    @patch('integrations.jira.jira_view.JiraFactory._create_provider_handler')
    @patch('integrations.jira.jira_view.infer_repo_from_message')
    async def test_create_view_multiple_repos_verified(
        self,
        mock_infer_repos,
        mock_create_handler,
        sample_webhook_payload,
        sample_user_auth,
        sample_jira_user,
        sample_jira_workspace,
        sample_repositories,
    ):
        """Test factory raises error when multiple repos are verified."""
        mock_handler = MagicMock()
        # Both repos verify successfully
        mock_handler.verify_repo_provider = AsyncMock(
            side_effect=[sample_repositories[0], sample_repositories[1]]
        )
        mock_create_handler.return_value = mock_handler

        # Multiple repos found in text
        mock_infer_repos.return_value = ['test/repo1', 'test/repo2']

        mock_response = MagicMock()
        mock_response.json.return_value = {
            'fields': {'summary': 'Test Issue', 'description': 'Test description'}
        }
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(
                RepositoryNotFoundError, match='Multiple repositories found'
            ):
                await JiraFactory.create_view(
                    payload=sample_webhook_payload,
                    workspace=sample_jira_workspace,
                    user=sample_jira_user,
                    user_auth=sample_user_auth,
                    decrypted_api_key='test_api_key',
                )

    @pytest.mark.asyncio
    @patch('integrations.jira.jira_view.JiraFactory._create_provider_handler')
    async def test_create_view_no_provider(
        self,
        mock_create_handler,
        sample_webhook_payload,
        sample_user_auth,
        sample_jira_user,
        sample_jira_workspace,
    ):
        """Test factory raises error when no provider is connected."""
        mock_create_handler.return_value = None

        mock_response = MagicMock()
        mock_response.json.return_value = {
            'fields': {'summary': 'Test Issue', 'description': 'Test description'}
        }
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(
                RepositoryNotFoundError, match='No Git provider connected'
            ):
                await JiraFactory.create_view(
                    payload=sample_webhook_payload,
                    workspace=sample_jira_workspace,
                    user=sample_jira_user,
                    user_auth=sample_user_auth,
                    decrypted_api_key='test_api_key',
                )


class TestJiraPayloadParser:
    """Tests for JiraPayloadParser"""

    @pytest.fixture
    def parser(self):
        """Create a parser for testing."""
        return JiraPayloadParser(oh_label='openhands', inline_oh_label='@openhands')

    def test_parse_label_event_success(
        self, parser, sample_issue_update_webhook_payload
    ):
        """Test parsing label event."""
        result = parser.parse(sample_issue_update_webhook_payload)

        assert isinstance(result, JiraPayloadSuccess)
        assert result.payload.event_type == JiraEventType.LABELED_TICKET
        assert result.payload.issue_key == 'PROJ-123'

    def test_parse_comment_event_success(self, parser, sample_comment_webhook_payload):
        """Test parsing comment event."""
        result = parser.parse(sample_comment_webhook_payload)

        assert isinstance(result, JiraPayloadSuccess)
        assert result.payload.event_type == JiraEventType.COMMENT_MENTION
        assert result.payload.issue_key == 'TEST-123'
        assert '@openhands' in result.payload.comment_body

    def test_parse_unknown_event_skipped(self, parser):
        """Test unknown event is skipped."""
        payload = {'webhookEvent': 'unknown_event'}
        result = parser.parse(payload)

        assert isinstance(result, JiraPayloadSkipped)
        assert 'Unhandled webhook event type' in result.skip_reason

    def test_parse_label_event_wrong_label_skipped(self, parser):
        """Test label event without OH label is skipped."""
        payload = {
            'webhookEvent': 'jira:issue_updated',
            'changelog': {'items': [{'field': 'labels', 'toString': 'other-label'}]},
        }
        result = parser.parse(payload)

        assert isinstance(result, JiraPayloadSkipped)
        assert 'does not contain' in result.skip_reason

    def test_parse_comment_event_no_mention_skipped(self, parser):
        """Test comment without mention is skipped."""
        payload = {
            'webhookEvent': 'comment_created',
            'comment': {
                'body': 'Regular comment',
                'author': {'emailAddress': 'test@test.com'},
            },
        }
        result = parser.parse(payload)

        assert isinstance(result, JiraPayloadSkipped)
        assert 'does not mention' in result.skip_reason

    def test_parse_missing_fields_error(self, parser):
        """Test missing required fields returns error."""
        payload = {
            'webhookEvent': 'jira:issue_updated',
            'changelog': {'items': [{'field': 'labels', 'toString': 'openhands'}]},
            'issue': {'id': '123'},  # Missing key
            'user': {'emailAddress': 'test@test.com'},  # Missing other fields
        }
        result = parser.parse(payload)

        assert isinstance(result, JiraPayloadError)
        assert 'Missing required fields' in result.error


class TestJiraPayloadParserStagingLabels:
    """Tests for JiraPayloadParser with staging labels."""

    @pytest.fixture
    def staging_parser(self):
        """Create a parser with staging labels."""
        return JiraPayloadParser(
            oh_label='openhands-exp', inline_oh_label='@openhands-exp'
        )

    def test_parse_staging_label(self, staging_parser):
        """Test parsing with staging label."""
        payload = {
            'webhookEvent': 'jira:issue_updated',
            'changelog': {'items': [{'field': 'labels', 'toString': 'openhands-exp'}]},
            'issue': {
                'id': '123',
                'key': 'TEST-1',
                'self': 'https://test.atlassian.net/rest/api/2/issue/123',
            },
            'user': {
                'emailAddress': 'test@test.com',
                'displayName': 'Test',
                'accountId': 'acc123',
                'self': 'https://test.atlassian.net/rest/api/2/user',
            },
        }
        result = staging_parser.parse(payload)

        assert isinstance(result, JiraPayloadSuccess)
        assert result.payload.event_type == JiraEventType.LABELED_TICKET

    def test_parse_prod_label_in_staging_skipped(self, staging_parser):
        """Test prod label is skipped in staging environment."""
        payload = {
            'webhookEvent': 'jira:issue_updated',
            'changelog': {'items': [{'field': 'labels', 'toString': 'openhands'}]},
        }
        result = staging_parser.parse(payload)

        assert isinstance(result, JiraPayloadSkipped)
