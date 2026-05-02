from unittest import TestCase, mock
from unittest.mock import MagicMock, patch
from uuid import UUID

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
        )

    @pytest.mark.asyncio
    @patch.object(GithubIssue, '_create_v1_conversation')
    async def test_create_new_conversation_routes_to_v1(self, mock_create_v1):
        """Test that conversation creation routes to V1."""
        mock_create_v1.return_value = None

        github_issue = self._create_github_issue()

        # Mock parameters
        jinja_env = MagicMock()
        git_provider_tokens = MagicMock()
        conversation_metadata = MagicMock()
        saas_user_auth = MagicMock()

        # Call the method
        await github_issue.create_new_conversation(
            jinja_env, git_provider_tokens, conversation_metadata, saas_user_auth
        )

        # Verify V1 was called
        mock_create_v1.assert_called_once_with(
            jinja_env, saas_user_auth, conversation_metadata
        )


class TestGithubOrgRouting(TestCase):
    """Test org routing for GitHub resolver conversations."""

    def setUp(self):
        self.user_data = UserData(
            user_id=123, username='testuser', keycloak_user_id='test-keycloak-id'
        )
        self.raw_payload = Message(
            source=SourceType.GITHUB,
            message={
                'payload': {
                    'action': 'opened',
                    'issue': {'number': 42},
                }
            },
        )
        self.resolved_org_id = UUID('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa')

    def _create_github_issue(self):
        return GithubIssue(
            user_info=self.user_data,
            full_repo_name='ClaimedOrg/repo',
            issue_number=42,
            installation_id=456,
            conversation_id='',
            should_extract=True,
            send_summary_instruction=False,
            is_public_repo=True,
            raw_payload=self.raw_payload,
            uuid='test-uuid',
            title='',
            description='',
            previous_comments=[],
        )

    @pytest.mark.asyncio
    @patch('integrations.github.github_view.get_app_conversation_service')
    @patch('integrations.github.github_view.resolve_org_for_repo')
    async def test_v1_passes_resolver_org_id_to_resolver_user_context(
        self, mock_resolve_org, mock_get_service
    ):
        """V1 path passes resolved org_id to ResolverUserContext."""
        # Arrange
        mock_resolve_org.return_value = self.resolved_org_id

        github_issue = self._create_github_issue()

        # Initialize to set resolved_org_id
        await github_issue.initialize_new_conversation()

        # Assert
        assert github_issue.resolved_org_id == self.resolved_org_id

    @pytest.mark.asyncio
    @patch('integrations.github.github_view.get_app_conversation_service')
    @patch('integrations.github.github_view.resolve_org_for_repo')
    async def test_no_claim_passes_none_resolver_org_id(
        self, mock_resolve_org, mock_get_service
    ):
        """When no claim exists, resolver_org_id is None (falls back to personal workspace)."""
        # Arrange
        mock_resolve_org.return_value = None

        github_issue = self._create_github_issue()

        # Act
        await github_issue.initialize_new_conversation()

        # Assert
        assert github_issue.resolved_org_id is None
