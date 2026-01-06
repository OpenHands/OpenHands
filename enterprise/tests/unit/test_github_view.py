from unittest import TestCase, mock
from unittest.mock import MagicMock, patch

import pytest
from integrations.github.github_view import GithubFactory, GithubIssue, get_oh_labels
from integrations.models import Message, SourceType
from integrations.types import UserData


class TestGithubLabels(TestCase):
    def test_labels_with_staging(self):
        oh_label, inline_oh_label = get_oh_labels('staging.all-hands.dev')
        self.assertEqual(oh_label, 'openhands-exp')
        self.assertEqual(inline_oh_label, '@openhands-exp')

    def test_labels_with_staging_v2(self):
        oh_label, inline_oh_label = get_oh_labels('main.staging.all-hands.dev')
        self.assertEqual(oh_label, 'openhands-exp')
        self.assertEqual(inline_oh_label, '@openhands-exp')

    def test_labels_with_local(self):
        oh_label, inline_oh_label = get_oh_labels('localhost:3000')
        self.assertEqual(oh_label, 'openhands-exp')
        self.assertEqual(inline_oh_label, '@openhands-exp')

    def test_labels_with_prod(self):
        oh_label, inline_oh_label = get_oh_labels('app.all-hands.dev')
        self.assertEqual(oh_label, 'openhands')
        self.assertEqual(inline_oh_label, '@openhands')

    def test_labels_with_spaces(self):
        """Test that spaces are properly stripped"""
        oh_label, inline_oh_label = get_oh_labels('  local  ')
        self.assertEqual(oh_label, 'openhands-exp')
        self.assertEqual(inline_oh_label, '@openhands-exp')


class TestGithubCommentCaseInsensitivity(TestCase):
    @mock.patch('integrations.github.github_view.INLINE_OH_LABEL', '@openhands')
    def test_issue_comment_case_insensitivity(self):
        # Test with lowercase mention
        message_lower = Message(
            source=SourceType.GITHUB,
            message={
                'payload': {
                    'action': 'created',
                    'comment': {'body': 'hello @openhands please help'},
                    'issue': {'number': 1},
                }
            },
        )

        # Test with uppercase mention
        message_upper = Message(
            source=SourceType.GITHUB,
            message={
                'payload': {
                    'action': 'created',
                    'comment': {'body': 'hello @OPENHANDS please help'},
                    'issue': {'number': 1},
                }
            },
        )

        # Test with mixed case mention
        message_mixed = Message(
            source=SourceType.GITHUB,
            message={
                'payload': {
                    'action': 'created',
                    'comment': {'body': 'hello @OpenHands please help'},
                    'issue': {'number': 1},
                }
            },
        )

        # All should be detected as issue comments with mentions
        self.assertTrue(GithubFactory.is_issue_comment(message_lower))
        self.assertTrue(GithubFactory.is_issue_comment(message_upper))
        self.assertTrue(GithubFactory.is_issue_comment(message_mixed))


class TestGithubV1ConversationRouting(TestCase):
    """Test V1 conversation routing logic in GitHub integration."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a proper UserData instance instead of MagicMock
        self.user_data = UserData(
            user_id=123, username='testuser', keycloak_user_id='test-keycloak-id'
        )

        # Create a mock raw_payload
        self.raw_payload = Message(
            source=SourceType.GITHUB,
            message={
                'payload': {
                    'action': 'opened',
                    'issue': {'number': 123},
                }
            },
        )

    def _create_github_issue(self):
        """Create a GithubIssue instance for testing."""
        return GithubIssue(
            user_info=self.user_data,
            full_repo_name='test/repo',
            issue_number=123,
            installation_id=456,
            conversation_id='test-conversation-id',
            should_extract=True,
            send_summary_instruction=False,
            is_public_repo=True,
            raw_payload=self.raw_payload,
            uuid='test-uuid',
            title='Test Issue',
            description='Test issue description',
            previous_comments=[],
            v1_enabled=False,
        )

    @pytest.mark.asyncio
    @patch('integrations.github.github_view.initialize_conversation')
    @patch('integrations.github.github_view.get_user_v1_enabled_setting')
    async def test_initialize_sets_v1_enabled_from_setting_when_false(
        self, mock_get_v1_setting, mock_initialize_conversation
    ):
        """Test that initialize_new_conversation sets v1_enabled from get_user_v1_enabled_setting."""
        mock_get_v1_setting.return_value = False
        mock_initialize_conversation.return_value = MagicMock(
            conversation_id='new-conversation-id'
        )

        github_issue = self._create_github_issue()
        await github_issue.initialize_new_conversation()

        # Verify get_user_v1_enabled_setting was called with correct user ID
        mock_get_v1_setting.assert_called_once_with('test-keycloak-id')
        # Verify v1_enabled was set to False
        self.assertFalse(github_issue.v1_enabled)

    @pytest.mark.asyncio
    @patch('integrations.github.github_view.get_user_v1_enabled_setting')
    async def test_initialize_sets_v1_enabled_from_setting_when_true(
        self, mock_get_v1_setting
    ):
        """Test that initialize_new_conversation sets v1_enabled to True when setting returns True."""
        mock_get_v1_setting.return_value = True

        github_issue = self._create_github_issue()
        await github_issue.initialize_new_conversation()

        # Verify get_user_v1_enabled_setting was called with correct user ID
        mock_get_v1_setting.assert_called_once_with('test-keycloak-id')
        # Verify v1_enabled was set to True
        self.assertTrue(github_issue.v1_enabled)

    @pytest.mark.asyncio
    @patch.object(GithubIssue, '_create_v0_conversation')
    @patch.object(GithubIssue, '_create_v1_conversation')
    async def test_create_new_conversation_routes_to_v0_when_disabled(
        self, mock_create_v1, mock_create_v0
    ):
        """Test that conversation creation routes to V0 when v1_enabled is False."""
        mock_create_v0.return_value = None
        mock_create_v1.return_value = None

        github_issue = self._create_github_issue()
        github_issue.v1_enabled = False

        # Mock parameters
        jinja_env = MagicMock()
        git_provider_tokens = MagicMock()
        conversation_metadata = MagicMock()
        saas_user_auth = MagicMock()

        # Call the method
        await github_issue.create_new_conversation(
            jinja_env, git_provider_tokens, conversation_metadata, saas_user_auth
        )

        # Verify V0 was called and V1 was not
        mock_create_v0.assert_called_once_with(
            jinja_env, git_provider_tokens, conversation_metadata
        )
        mock_create_v1.assert_not_called()

    @pytest.mark.asyncio
    @patch.object(GithubIssue, '_create_v0_conversation')
    @patch.object(GithubIssue, '_create_v1_conversation')
    async def test_create_new_conversation_routes_to_v1_when_enabled(
        self, mock_create_v1, mock_create_v0
    ):
        """Test that conversation creation routes to V1 when v1_enabled is True."""
        mock_create_v0.return_value = None
        mock_create_v1.return_value = None

        github_issue = self._create_github_issue()
        github_issue.v1_enabled = True

        # Mock parameters
        jinja_env = MagicMock()
        git_provider_tokens = MagicMock()
        conversation_metadata = MagicMock()
        saas_user_auth = MagicMock()

        # Call the method
        await github_issue.create_new_conversation(
            jinja_env, git_provider_tokens, conversation_metadata, saas_user_auth
        )

        # Verify V1 was called and V0 was not
        mock_create_v1.assert_called_once_with(
            jinja_env, saas_user_auth, conversation_metadata
        )
        mock_create_v0.assert_not_called()
