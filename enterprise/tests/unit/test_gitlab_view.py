"""Tests for GitLab resolver org routing logic.

Tests that the GitLab resolver correctly resolves the target organization
and passes resolver_org_id through V0 and V1 conversation paths.
"""

from unittest import TestCase
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from integrations.gitlab.gitlab_view import GitlabIssue
from integrations.models import Message, SourceType
from integrations.types import UserData


class TestGitlabOrgRouting(TestCase):
    """Test org routing for GitLab resolver conversations."""

    def setUp(self):
        self.user_data = UserData(
            user_id=123, username='testuser', keycloak_user_id='test-keycloak-id'
        )
        self.raw_payload = Message(
            source=SourceType.GITLAB,
            message={
                'payload': {
                    'object_kind': 'issue',
                    'object_attributes': {'action': 'open', 'iid': 42},
                }
            },
        )
        self.resolved_org_id = UUID('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa')

    def _create_gitlab_issue(self):
        return GitlabIssue(
            user_info=self.user_data,
            full_repo_name='ClaimedOrg/repo',
            issue_number=42,
            project_id=100,
            installation_id='install-123',
            conversation_id='',
            should_extract=True,
            send_summary_instruction=False,
            is_public_repo=True,
            raw_payload=self.raw_payload,
            title='',
            description='',
            previous_comments=[],
            is_mr=False,
            v1_enabled=False,
        )

    @pytest.mark.asyncio
    @patch(
        'integrations.gitlab.gitlab_view.SaasConversationStore.get_resolver_instance'
    )
    @patch('integrations.gitlab.gitlab_view.resolve_org_for_repo')
    async def test_v0_passes_resolver_org_id_to_get_resolver_instance(
        self, mock_resolve_org, mock_get_resolver
    ):
        """V0 path creates store via get_resolver_instance with resolver_org_id."""
        # Arrange
        mock_resolve_org.return_value = self.resolved_org_id
        mock_store = MagicMock()
        mock_store.save_metadata = AsyncMock()
        mock_get_resolver.return_value = mock_store

        gitlab_issue = self._create_gitlab_issue()

        # Act
        await gitlab_issue.initialize_new_conversation()

        # Assert
        mock_resolve_org.assert_called_once_with(
            provider='gitlab',
            full_repo_name='ClaimedOrg/repo',
            keycloak_user_id='test-keycloak-id',
        )
        # get_resolver_instance(config, user_id, resolver_org_id)
        args, _ = mock_get_resolver.call_args
        assert args[1] == 'test-keycloak-id'
        assert args[2] == self.resolved_org_id

    @pytest.mark.asyncio
    @patch('integrations.gitlab.gitlab_view.get_app_conversation_service')
    @patch('integrations.gitlab.gitlab_view.resolve_org_for_repo')
    async def test_v1_passes_resolver_org_id_to_resolver_user_context(
        self, mock_resolve_org, mock_get_service
    ):
        """V1 path passes resolved org_id to ResolverUserContext."""
        # Arrange
        mock_resolve_org.return_value = self.resolved_org_id

        gitlab_issue = self._create_gitlab_issue()
        gitlab_issue.v1_enabled = True

        # Initialize to set resolved_org_id
        await gitlab_issue.initialize_new_conversation()

        # Assert
        assert gitlab_issue.resolved_org_id == self.resolved_org_id

    @pytest.mark.asyncio
    @patch(
        'integrations.gitlab.gitlab_view.SaasConversationStore.get_resolver_instance'
    )
    @patch('integrations.gitlab.gitlab_view.resolve_org_for_repo')
    async def test_no_claim_passes_none_resolver_org_id(
        self, mock_resolve_org, mock_get_resolver
    ):
        """When no claim exists, resolver_org_id is None (falls back to personal workspace)."""
        # Arrange
        mock_resolve_org.return_value = None
        mock_store = MagicMock()
        mock_store.save_metadata = AsyncMock()
        mock_get_resolver.return_value = mock_store

        gitlab_issue = self._create_gitlab_issue()

        # Act
        await gitlab_issue.initialize_new_conversation()

        # Assert
        args, _ = mock_get_resolver.call_args
        assert args[2] is None
