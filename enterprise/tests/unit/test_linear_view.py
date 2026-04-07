"""Tests for Linear resolver org routing logic.

Tests that the LinearNewConversationView correctly resolves the target
organization and passes resolver_org_id through the V0 conversation path.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from integrations.linear.linear_view import LinearNewConversationView
from integrations.models import JobContext
from storage.linear_user import LinearUser
from storage.linear_workspace import LinearWorkspace

from openhands.integrations.service_types import ProviderType
from openhands.server.user_auth.user_auth import UserAuth
from openhands.storage.data_models.conversation_metadata import ConversationTrigger

CLAIMING_ORG_ID = UUID('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa')
KEYCLOAK_USER_ID = 'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb'


@pytest.fixture
def mock_linear_user():
    user = LinearUser()
    user.id = 1
    user.keycloak_user_id = KEYCLOAK_USER_ID
    user.linear_user_id = 'linear-user-123'
    user.linear_workspace_id = 1
    user.status = 'active'
    return user


@pytest.fixture
def mock_linear_workspace():
    workspace = LinearWorkspace()
    workspace.id = 1
    workspace.name = 'test-workspace'
    workspace.linear_org_id = 'linear-org-123'
    workspace.admin_user_id = 'admin-123'
    workspace.webhook_secret = 'secret'
    workspace.svc_acc_email = 'svc@test.com'
    workspace.svc_acc_api_key = 'api-key'
    workspace.status = 'active'
    return workspace


@pytest.fixture
def mock_user_auth():
    auth = MagicMock(spec=UserAuth)
    auth.get_provider_tokens = AsyncMock(
        return_value={ProviderType.GITHUB: MagicMock()}
    )
    auth.get_secrets = AsyncMock(return_value=MagicMock(custom_secrets={}))
    return auth


@pytest.fixture
def job_context():
    return JobContext(
        issue_id='issue-123',
        issue_key='PROJ-42',
        issue_title='Test issue',
        issue_description='Test description',
        user_msg='@openhands fix this',
        user_email='user@test.com',
        platform_user_id='linear-user-123',
        workspace_name='test-workspace',
        display_name='Test User',
    )


@pytest.fixture
def linear_view(job_context, mock_user_auth, mock_linear_user, mock_linear_workspace):
    return LinearNewConversationView(
        job_context=job_context,
        saas_user_auth=mock_user_auth,
        linear_user=mock_linear_user,
        linear_workspace=mock_linear_workspace,
        selected_repo='OpenHands/foo',
        conversation_id='',
    )


class TestLinearV0OrgRouting:
    """Test V0 conversation routing logic for Linear resolver."""

    @pytest.mark.asyncio
    @patch(
        'integrations.linear.linear_view.resolve_org_for_repo', new_callable=AsyncMock
    )
    @patch('integrations.linear.linear_view.ProviderHandler')
    @patch(
        'integrations.linear.linear_view.SaasConversationStore.get_resolver_instance',
        new_callable=AsyncMock,
    )
    @patch('integrations.linear.linear_view.start_conversation', new_callable=AsyncMock)
    @patch(
        'integrations.linear.linear_view.integration_store',
    )
    async def test_v0_passes_resolver_org_id_to_get_resolver_instance(
        self,
        mock_integration_store,
        mock_start_convo,
        mock_get_resolver_instance,
        mock_provider_handler_cls,
        mock_resolve_org,
        linear_view,
    ):
        """V0 path should resolve org and pass resolver_org_id to get_resolver_instance."""
        # Arrange
        mock_repo = MagicMock()
        mock_repo.git_provider = ProviderType.GITHUB
        mock_handler = MagicMock()
        mock_handler.verify_repo_provider = AsyncMock(return_value=mock_repo)
        mock_provider_handler_cls.return_value = mock_handler

        mock_resolve_org.return_value = CLAIMING_ORG_ID
        mock_store = MagicMock()
        mock_store.save_metadata = AsyncMock()
        mock_get_resolver_instance.return_value = mock_store
        mock_integration_store.create_conversation = AsyncMock()

        mock_jinja = MagicMock()

        # Act
        with patch.object(
            linear_view,
            '_get_instructions',
            new_callable=AsyncMock,
            return_value=('instructions', 'user_msg'),
        ):
            await linear_view.create_or_update_conversation(mock_jinja)

        # Assert
        mock_resolve_org.assert_called_once_with(
            provider='github',
            full_repo_name='OpenHands/foo',
            keycloak_user_id=KEYCLOAK_USER_ID,
        )
        call_args = mock_get_resolver_instance.call_args
        assert call_args[0][1] == KEYCLOAK_USER_ID
        assert call_args[0][2] == CLAIMING_ORG_ID
        saved_metadata = mock_store.save_metadata.call_args[0][0]
        assert saved_metadata.trigger == ConversationTrigger.LINEAR
        assert saved_metadata.git_provider == ProviderType.GITHUB

    @pytest.mark.asyncio
    @patch(
        'integrations.linear.linear_view.resolve_org_for_repo', new_callable=AsyncMock
    )
    @patch('integrations.linear.linear_view.ProviderHandler')
    @patch(
        'integrations.linear.linear_view.SaasConversationStore.get_resolver_instance',
        new_callable=AsyncMock,
    )
    @patch('integrations.linear.linear_view.start_conversation', new_callable=AsyncMock)
    @patch(
        'integrations.linear.linear_view.integration_store',
    )
    async def test_v0_passes_none_when_no_claim(
        self,
        mock_integration_store,
        mock_start_convo,
        mock_get_resolver_instance,
        mock_provider_handler_cls,
        mock_resolve_org,
        linear_view,
    ):
        """When no claim exists, resolver_org_id should be None (personal workspace)."""
        # Arrange
        mock_repo = MagicMock()
        mock_repo.git_provider = ProviderType.GITHUB
        mock_handler = MagicMock()
        mock_handler.verify_repo_provider = AsyncMock(return_value=mock_repo)
        mock_provider_handler_cls.return_value = mock_handler

        mock_resolve_org.return_value = None
        mock_store = MagicMock()
        mock_store.save_metadata = AsyncMock()
        mock_get_resolver_instance.return_value = mock_store
        mock_integration_store.create_conversation = AsyncMock()

        mock_jinja = MagicMock()

        # Act
        with patch.object(
            linear_view,
            '_get_instructions',
            new_callable=AsyncMock,
            return_value=('instructions', 'user_msg'),
        ):
            await linear_view.create_or_update_conversation(mock_jinja)

        # Assert
        call_args = mock_get_resolver_instance.call_args
        assert call_args[0][2] is None

    @pytest.mark.asyncio
    @patch(
        'integrations.linear.linear_view.resolve_org_for_repo', new_callable=AsyncMock
    )
    @patch(
        'integrations.linear.linear_view.SaasConversationStore.get_resolver_instance',
        new_callable=AsyncMock,
    )
    @patch('integrations.linear.linear_view.start_conversation', new_callable=AsyncMock)
    @patch(
        'integrations.linear.linear_view.integration_store',
    )
    async def test_no_provider_tokens_skips_org_resolution(
        self,
        mock_integration_store,
        mock_start_convo,
        mock_get_resolver_instance,
        mock_resolve_org,
        linear_view,
        mock_user_auth,
    ):
        """When provider tokens are None, org resolution should be skipped."""
        # Arrange
        mock_user_auth.get_provider_tokens = AsyncMock(return_value=None)
        mock_store = MagicMock()
        mock_store.save_metadata = AsyncMock()
        mock_get_resolver_instance.return_value = mock_store
        mock_integration_store.create_conversation = AsyncMock()

        mock_jinja = MagicMock()

        # Act
        with patch.object(
            linear_view,
            '_get_instructions',
            new_callable=AsyncMock,
            return_value=('instructions', 'user_msg'),
        ):
            await linear_view.create_or_update_conversation(mock_jinja)

        # Assert
        mock_resolve_org.assert_not_called()
        call_args = mock_get_resolver_instance.call_args
        assert call_args[0][2] is None
        saved_metadata = mock_store.save_metadata.call_args[0][0]
        assert saved_metadata.git_provider is None

    @pytest.mark.asyncio
    @patch(
        'integrations.linear.linear_view.resolve_org_for_repo', new_callable=AsyncMock
    )
    @patch('integrations.linear.linear_view.ProviderHandler')
    @patch(
        'integrations.linear.linear_view.SaasConversationStore.get_resolver_instance',
        new_callable=AsyncMock,
    )
    @patch('integrations.linear.linear_view.start_conversation', new_callable=AsyncMock)
    @patch(
        'integrations.linear.linear_view.integration_store',
    )
    async def test_verify_repo_provider_failure_falls_back_to_personal_workspace(
        self,
        mock_integration_store,
        mock_start_convo,
        mock_get_resolver_instance,
        mock_provider_handler_cls,
        mock_resolve_org,
        linear_view,
    ):
        """When verify_repo_provider fails, should fall back to personal workspace."""
        # Arrange
        mock_handler = MagicMock()
        mock_handler.verify_repo_provider = AsyncMock(
            side_effect=Exception('Repository not found')
        )
        mock_provider_handler_cls.return_value = mock_handler

        mock_store = MagicMock()
        mock_store.save_metadata = AsyncMock()
        mock_get_resolver_instance.return_value = mock_store
        mock_integration_store.create_conversation = AsyncMock()

        mock_jinja = MagicMock()

        # Act
        with patch.object(
            linear_view,
            '_get_instructions',
            new_callable=AsyncMock,
            return_value=('instructions', 'user_msg'),
        ):
            await linear_view.create_or_update_conversation(mock_jinja)

        # Assert - org resolution should be skipped, conversation created in personal workspace
        mock_resolve_org.assert_not_called()
        call_args = mock_get_resolver_instance.call_args
        assert call_args[0][2] is None

    @pytest.mark.asyncio
    @patch(
        'integrations.linear.linear_view.resolve_org_for_repo', new_callable=AsyncMock
    )
    @patch('integrations.linear.linear_view.ProviderHandler')
    @patch(
        'integrations.linear.linear_view.SaasConversationStore.get_resolver_instance',
        new_callable=AsyncMock,
    )
    @patch('integrations.linear.linear_view.start_conversation', new_callable=AsyncMock)
    @patch(
        'integrations.linear.linear_view.integration_store',
    )
    async def test_resolve_org_failure_falls_back_to_personal_workspace(
        self,
        mock_integration_store,
        mock_start_convo,
        mock_get_resolver_instance,
        mock_provider_handler_cls,
        mock_resolve_org,
        linear_view,
    ):
        """When resolve_org_for_repo fails, should fall back to personal workspace."""
        # Arrange
        mock_repo = MagicMock()
        mock_repo.git_provider = ProviderType.GITHUB
        mock_handler = MagicMock()
        mock_handler.verify_repo_provider = AsyncMock(return_value=mock_repo)
        mock_provider_handler_cls.return_value = mock_handler

        mock_resolve_org.side_effect = Exception('Database connection failed')
        mock_store = MagicMock()
        mock_store.save_metadata = AsyncMock()
        mock_get_resolver_instance.return_value = mock_store
        mock_integration_store.create_conversation = AsyncMock()

        mock_jinja = MagicMock()

        # Act
        with patch.object(
            linear_view,
            '_get_instructions',
            new_callable=AsyncMock,
            return_value=('instructions', 'user_msg'),
        ):
            await linear_view.create_or_update_conversation(mock_jinja)

        # Assert - conversation should be created with resolver_org_id=None
        call_args = mock_get_resolver_instance.call_args
        assert call_args[0][2] is None
