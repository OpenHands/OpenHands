"""Tests for Slack view org routing logic.

Tests that the SlackNewConversationView correctly resolves the target org
based on claimed git organizations and passes it through V0/V1 paths.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from integrations.slack.slack_view import SlackNewConversationView
from storage.slack_user import SlackUser

from openhands.integrations.service_types import ProviderType
from openhands.server.user_auth.user_auth import UserAuth

CLAIMING_ORG_ID = UUID('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa')
KEYCLOAK_USER_ID = 'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb'


@pytest.fixture
def mock_slack_user():
    """Create a mock SlackUser."""
    user = SlackUser()
    user.slack_user_id = 'U1234567890'
    user.keycloak_user_id = KEYCLOAK_USER_ID
    user.slack_display_name = 'Test User'
    user.org_id = UUID('cccccccc-cccc-cccc-cccc-cccccccccccc')
    return user


@pytest.fixture
def mock_user_auth():
    """Create a mock UserAuth."""
    auth = MagicMock(spec=UserAuth)
    auth.get_provider_tokens = AsyncMock(
        return_value={ProviderType.GITHUB: MagicMock()}
    )
    auth.get_secrets = AsyncMock(return_value=MagicMock(custom_secrets={}))
    auth.get_access_token = AsyncMock(return_value='access-token')
    auth.get_user_id = AsyncMock(return_value=KEYCLOAK_USER_ID)
    return auth


@pytest.fixture
def slack_view(mock_slack_user, mock_user_auth):
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
        selected_repo='OpenHands/foo',
        should_extract=True,
        send_summary_instruction=True,
        conversation_id='',
        team_id='T1234567890',
        v1_enabled=False,
    )


@pytest.fixture
def slack_view_no_repo(mock_slack_user, mock_user_auth):
    """Create a SlackNewConversationView with no selected repo."""
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


class TestSlackV0ConversationRouting:
    """Test V0 conversation routing logic in Slack integration."""

    @pytest.mark.asyncio
    @patch(
        'integrations.slack.slack_view.is_v1_enabled_for_slack_resolver',
        new_callable=AsyncMock,
        return_value=False,
    )
    @patch('integrations.slack.slack_view.resolve_org_for_repo', new_callable=AsyncMock)
    @patch('integrations.slack.slack_view.ProviderHandler')
    @patch(
        'integrations.slack.slack_view.SaasConversationStore.get_resolver_instance',
        new_callable=AsyncMock,
    )
    @patch('integrations.slack.slack_view.start_conversation', new_callable=AsyncMock)
    async def test_v0_passes_resolver_org_id(
        self,
        mock_start_convo,
        mock_get_resolver_instance,
        mock_provider_handler_cls,
        mock_resolve_org,
        mock_v1_enabled,
        slack_view,
    ):
        """V0 path should pass resolver_org_id to SaasConversationStore.get_resolver_instance."""
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
        mock_start_convo.return_value = MagicMock(conversation_id='test-conv-id')

        mock_jinja = MagicMock()

        # Act
        with (
            patch.object(
                slack_view,
                '_get_instructions',
                new_callable=AsyncMock,
                return_value=('msg', 'instructions'),
            ),
            patch.object(slack_view, 'save_slack_convo', new_callable=AsyncMock),
        ):
            await slack_view.create_or_update_conversation(mock_jinja)

        # Assert
        mock_resolve_org.assert_called_once_with(
            provider='github',
            full_repo_name='OpenHands/foo',
            keycloak_user_id=KEYCLOAK_USER_ID,
        )
        mock_get_resolver_instance.assert_called_once()
        call_args = mock_get_resolver_instance.call_args
        assert call_args[0][1] == KEYCLOAK_USER_ID  # user_id
        assert call_args[0][2] == CLAIMING_ORG_ID  # resolver_org_id
        mock_store.save_metadata.assert_called_once()
        saved_metadata = mock_store.save_metadata.call_args[0][0]
        assert saved_metadata.git_provider == ProviderType.GITHUB

    @pytest.mark.asyncio
    @patch(
        'integrations.slack.slack_view.is_v1_enabled_for_slack_resolver',
        new_callable=AsyncMock,
        return_value=False,
    )
    @patch('integrations.slack.slack_view.resolve_org_for_repo', new_callable=AsyncMock)
    @patch('integrations.slack.slack_view.ProviderHandler')
    @patch(
        'integrations.slack.slack_view.SaasConversationStore.get_resolver_instance',
        new_callable=AsyncMock,
    )
    @patch('integrations.slack.slack_view.start_conversation', new_callable=AsyncMock)
    async def test_v0_passes_none_when_no_claim(
        self,
        mock_start_convo,
        mock_get_resolver_instance,
        mock_provider_handler_cls,
        mock_resolve_org,
        mock_v1_enabled,
        slack_view,
    ):
        """V0 path should pass resolver_org_id=None when no claim exists."""
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
        mock_start_convo.return_value = MagicMock(conversation_id='test-conv-id')

        mock_jinja = MagicMock()

        # Act
        with (
            patch.object(
                slack_view,
                '_get_instructions',
                new_callable=AsyncMock,
                return_value=('msg', 'instructions'),
            ),
            patch.object(slack_view, 'save_slack_convo', new_callable=AsyncMock),
        ):
            await slack_view.create_or_update_conversation(mock_jinja)

        # Assert
        call_args = mock_get_resolver_instance.call_args
        assert call_args[0][2] is None  # resolver_org_id is None


class TestSlackV1ConversationRouting:
    """Test V1 conversation routing logic in Slack integration."""

    @pytest.mark.asyncio
    @patch(
        'integrations.slack.slack_view.is_v1_enabled_for_slack_resolver',
        new_callable=AsyncMock,
        return_value=True,
    )
    @patch('integrations.slack.slack_view.resolve_org_for_repo', new_callable=AsyncMock)
    @patch('integrations.slack.slack_view.ProviderHandler')
    @patch('integrations.slack.slack_view.get_app_conversation_service')
    @patch('integrations.slack.slack_view.ResolverUserContext')
    async def test_v1_passes_resolver_org_id_to_context(
        self,
        mock_resolver_ctx_cls,
        mock_get_service,
        mock_provider_handler_cls,
        mock_resolve_org,
        mock_v1_enabled,
        slack_view,
    ):
        """V1 path should pass resolver_org_id to ResolverUserContext."""
        # Arrange
        mock_repo = MagicMock()
        mock_repo.git_provider = ProviderType.GITHUB
        mock_handler = MagicMock()
        mock_handler.verify_repo_provider = AsyncMock(return_value=mock_repo)
        mock_provider_handler_cls.return_value = mock_handler

        mock_resolve_org.return_value = CLAIMING_ORG_ID
        mock_resolver_ctx_cls.return_value = MagicMock()

        # Mock the async context manager for app_conversation_service
        mock_service = MagicMock()
        mock_service.start_app_conversation = MagicMock(return_value=aiter_empty())
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_service)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_get_service.return_value = mock_ctx

        mock_jinja = MagicMock()

        # Act
        with patch.object(
            slack_view,
            '_get_instructions',
            new_callable=AsyncMock,
            return_value=('msg', 'instructions'),
        ):
            with patch.object(slack_view, 'save_slack_convo', new_callable=AsyncMock):
                await slack_view.create_or_update_conversation(mock_jinja)

        # Assert
        mock_resolve_org.assert_called_once_with(
            provider='github',
            full_repo_name='OpenHands/foo',
            keycloak_user_id=KEYCLOAK_USER_ID,
        )
        mock_resolver_ctx_cls.assert_called_once_with(
            saas_user_auth=slack_view.saas_user_auth,
            resolver_org_id=CLAIMING_ORG_ID,
        )


class TestSlackNoRepoRouting:
    """Test routing when no repository is selected."""

    @pytest.mark.asyncio
    @patch(
        'integrations.slack.slack_view.is_v1_enabled_for_slack_resolver',
        new_callable=AsyncMock,
        return_value=False,
    )
    @patch('integrations.slack.slack_view.resolve_org_for_repo', new_callable=AsyncMock)
    @patch(
        'integrations.slack.slack_view.SaasConversationStore.get_resolver_instance',
        new_callable=AsyncMock,
    )
    @patch('integrations.slack.slack_view.start_conversation', new_callable=AsyncMock)
    async def test_no_repo_skips_org_resolution(
        self,
        mock_start_convo,
        mock_get_resolver_instance,
        mock_resolve_org,
        mock_v1_enabled,
        slack_view_no_repo,
    ):
        """When selected_repo is None, org resolution should be skipped."""
        # Arrange
        mock_store = MagicMock()
        mock_store.save_metadata = AsyncMock()
        mock_get_resolver_instance.return_value = mock_store
        mock_start_convo.return_value = MagicMock(conversation_id='test-conv-id')
        mock_jinja = MagicMock()

        # Act
        with (
            patch.object(
                slack_view_no_repo,
                '_get_instructions',
                new_callable=AsyncMock,
                return_value=('msg', 'instructions'),
            ),
            patch.object(
                slack_view_no_repo, 'save_slack_convo', new_callable=AsyncMock
            ),
            patch.object(slack_view_no_repo, '_verify_necessary_values_are_set'),
        ):
            await slack_view_no_repo.create_or_update_conversation(mock_jinja)

        # Assert
        mock_resolve_org.assert_not_called()
        call_args = mock_get_resolver_instance.call_args
        assert call_args[0][2] is None  # resolver_org_id is None
        saved_metadata = mock_store.save_metadata.call_args[0][0]
        assert saved_metadata.git_provider is None


async def aiter_empty():
    """Helper: empty async iterator."""
    return
    yield  # noqa: unreachable - makes this an async generator
