"""Unit tests for install_gitlab_webhooks module."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from integrations.gitlab.webhook_installation import (
    BreakLoopException,
    install_webhook_on_resource,
    verify_webhook_conditions,
)
from integrations.types import GitLabResourceType
from integrations.utils import GITLAB_WEBHOOK_URL
from storage.gitlab_webhook import GitlabWebhook, WebhookStatus


@pytest.fixture
def mock_gitlab_service():
    """Create a mock GitLab service."""
    service = MagicMock()
    service.check_resource_exists = AsyncMock(return_value=(True, None))
    service.check_user_has_admin_access_to_resource = AsyncMock(
        return_value=(True, None)
    )
    service.check_webhook_exists_on_resource = AsyncMock(return_value=(False, None))
    service.install_webhook = AsyncMock(return_value=('webhook-id-123', None))
    return service


@pytest.fixture
def mock_webhook_store():
    """Create a mock webhook store."""
    store = MagicMock()
    store.delete_webhook = AsyncMock()
    store.update_webhook = AsyncMock()
    return store


@pytest.fixture
def sample_webhook():
    """Create a sample webhook object."""
    webhook = MagicMock(spec=GitlabWebhook)
    webhook.user_id = 'test_user_id'
    webhook.webhook_exists = False
    webhook.webhook_uuid = None
    return webhook


class TestVerifyWebhookConditions:
    """Test cases for verify_webhook_conditions function."""

    @pytest.mark.asyncio
    async def test_verify_conditions_all_pass(
        self, mock_gitlab_service, mock_webhook_store, sample_webhook
    ):
        """Test when all conditions are met for webhook installation."""
        # Arrange
        resource_type = GitLabResourceType.PROJECT
        resource_id = 'project-123'

        # Act
        # Should not raise any exception
        await verify_webhook_conditions(
            gitlab_service=mock_gitlab_service,
            resource_type=resource_type,
            resource_id=resource_id,
            webhook_store=mock_webhook_store,
            webhook=sample_webhook,
        )

        # Assert
        mock_gitlab_service.check_resource_exists.assert_called_once_with(
            resource_type, resource_id
        )
        mock_gitlab_service.check_user_has_admin_access_to_resource.assert_called_once_with(
            resource_type, resource_id
        )
        mock_gitlab_service.check_webhook_exists_on_resource.assert_called_once_with(
            resource_type, resource_id, GITLAB_WEBHOOK_URL
        )
        mock_webhook_store.delete_webhook.assert_not_called()

    @pytest.mark.asyncio
    async def test_verify_conditions_resource_does_not_exist(
        self, mock_gitlab_service, mock_webhook_store, sample_webhook
    ):
        """Test when resource does not exist."""
        # Arrange
        resource_type = GitLabResourceType.PROJECT
        resource_id = 'project-999'
        mock_gitlab_service.check_resource_exists = AsyncMock(
            return_value=(False, None)
        )

        # Act & Assert
        with pytest.raises(BreakLoopException):
            await verify_webhook_conditions(
                gitlab_service=mock_gitlab_service,
                resource_type=resource_type,
                resource_id=resource_id,
                webhook_store=mock_webhook_store,
                webhook=sample_webhook,
            )

        # Assert webhook is deleted
        mock_webhook_store.delete_webhook.assert_called_once_with(sample_webhook)

    @pytest.mark.asyncio
    async def test_verify_conditions_rate_limited_on_resource_check(
        self, mock_gitlab_service, mock_webhook_store, sample_webhook
    ):
        """Test when rate limited during resource existence check."""
        # Arrange
        resource_type = GitLabResourceType.PROJECT
        resource_id = 'project-123'
        mock_gitlab_service.check_resource_exists = AsyncMock(
            return_value=(False, WebhookStatus.RATE_LIMITED)
        )

        # Act & Assert
        with pytest.raises(BreakLoopException):
            await verify_webhook_conditions(
                gitlab_service=mock_gitlab_service,
                resource_type=resource_type,
                resource_id=resource_id,
                webhook_store=mock_webhook_store,
                webhook=sample_webhook,
            )

        # Should not delete webhook on rate limit
        mock_webhook_store.delete_webhook.assert_not_called()

    @pytest.mark.asyncio
    async def test_verify_conditions_user_no_admin_access(
        self, mock_gitlab_service, mock_webhook_store, sample_webhook
    ):
        """Test when user does not have admin access."""
        # Arrange
        resource_type = GitLabResourceType.GROUP
        resource_id = 'group-456'
        mock_gitlab_service.check_user_has_admin_access_to_resource = AsyncMock(
            return_value=(False, None)
        )

        # Act & Assert
        with pytest.raises(BreakLoopException):
            await verify_webhook_conditions(
                gitlab_service=mock_gitlab_service,
                resource_type=resource_type,
                resource_id=resource_id,
                webhook_store=mock_webhook_store,
                webhook=sample_webhook,
            )

        # Assert webhook is deleted
        mock_webhook_store.delete_webhook.assert_called_once_with(sample_webhook)

    @pytest.mark.asyncio
    async def test_verify_conditions_rate_limited_on_admin_check(
        self, mock_gitlab_service, mock_webhook_store, sample_webhook
    ):
        """Test when rate limited during admin access check."""
        # Arrange
        resource_type = GitLabResourceType.PROJECT
        resource_id = 'project-123'
        mock_gitlab_service.check_user_has_admin_access_to_resource = AsyncMock(
            return_value=(False, WebhookStatus.RATE_LIMITED)
        )

        # Act & Assert
        with pytest.raises(BreakLoopException):
            await verify_webhook_conditions(
                gitlab_service=mock_gitlab_service,
                resource_type=resource_type,
                resource_id=resource_id,
                webhook_store=mock_webhook_store,
                webhook=sample_webhook,
            )

        # Should not delete webhook on rate limit
        mock_webhook_store.delete_webhook.assert_not_called()

    @pytest.mark.asyncio
    async def test_verify_conditions_webhook_already_exists(
        self, mock_gitlab_service, mock_webhook_store, sample_webhook
    ):
        """Test when webhook already exists on resource."""
        # Arrange
        resource_type = GitLabResourceType.PROJECT
        resource_id = 'project-123'
        mock_gitlab_service.check_webhook_exists_on_resource = AsyncMock(
            return_value=(True, None)
        )

        # Act & Assert
        with pytest.raises(BreakLoopException):
            await verify_webhook_conditions(
                gitlab_service=mock_gitlab_service,
                resource_type=resource_type,
                resource_id=resource_id,
                webhook_store=mock_webhook_store,
                webhook=sample_webhook,
            )

    @pytest.mark.asyncio
    async def test_verify_conditions_rate_limited_on_webhook_check(
        self, mock_gitlab_service, mock_webhook_store, sample_webhook
    ):
        """Test when rate limited during webhook existence check."""
        # Arrange
        resource_type = GitLabResourceType.PROJECT
        resource_id = 'project-123'
        mock_gitlab_service.check_webhook_exists_on_resource = AsyncMock(
            return_value=(False, WebhookStatus.RATE_LIMITED)
        )

        # Act & Assert
        with pytest.raises(BreakLoopException):
            await verify_webhook_conditions(
                gitlab_service=mock_gitlab_service,
                resource_type=resource_type,
                resource_id=resource_id,
                webhook_store=mock_webhook_store,
                webhook=sample_webhook,
            )

    @pytest.mark.asyncio
    async def test_verify_conditions_updates_webhook_status_mismatch(
        self, mock_gitlab_service, mock_webhook_store, sample_webhook
    ):
        """Test that webhook status is updated when database and API don't match."""
        # Arrange
        resource_type = GitLabResourceType.PROJECT
        resource_id = 'project-123'
        sample_webhook.webhook_exists = True  # DB says exists
        mock_gitlab_service.check_webhook_exists_on_resource = AsyncMock(
            return_value=(False, None)  # API says doesn't exist
        )

        # Act
        # Should not raise BreakLoopException when webhook doesn't exist (allows installation)
        await verify_webhook_conditions(
            gitlab_service=mock_gitlab_service,
            resource_type=resource_type,
            resource_id=resource_id,
            webhook_store=mock_webhook_store,
            webhook=sample_webhook,
        )

        # Assert webhook status was updated to match API
        mock_webhook_store.update_webhook.assert_called_once_with(
            sample_webhook, {'webhook_exists': False}
        )


class TestInstallWebhookOnResource:
    """Test cases for install_webhook_on_resource function."""

    @pytest.mark.asyncio
    async def test_install_webhook_success(
        self, mock_gitlab_service, mock_webhook_store, sample_webhook
    ):
        """Test successful webhook installation."""
        # Arrange
        resource_type = GitLabResourceType.PROJECT
        resource_id = 'project-123'

        # Act
        webhook_id, status = await install_webhook_on_resource(
            gitlab_service=mock_gitlab_service,
            resource_type=resource_type,
            resource_id=resource_id,
            webhook_store=mock_webhook_store,
            webhook=sample_webhook,
        )

        # Assert
        assert webhook_id == 'webhook-id-123'
        assert status is None
        mock_gitlab_service.install_webhook.assert_called_once()
        mock_webhook_store.update_webhook.assert_called_once()
        # Verify update_webhook was called with correct fields (using keyword arguments)
        call_args = mock_webhook_store.update_webhook.call_args
        assert call_args[1]['webhook'] == sample_webhook
        update_fields = call_args[1]['update_fields']
        assert update_fields['webhook_exists'] is True
        assert update_fields['webhook_url'] == GITLAB_WEBHOOK_URL
        assert 'webhook_secret' in update_fields
        assert 'webhook_uuid' in update_fields
        assert 'scopes' in update_fields

    @pytest.mark.asyncio
    async def test_install_webhook_group_resource(
        self, mock_gitlab_service, mock_webhook_store, sample_webhook
    ):
        """Test webhook installation for a group resource."""
        # Arrange
        resource_type = GitLabResourceType.GROUP
        resource_id = 'group-456'

        # Act
        webhook_id, status = await install_webhook_on_resource(
            gitlab_service=mock_gitlab_service,
            resource_type=resource_type,
            resource_id=resource_id,
            webhook_store=mock_webhook_store,
            webhook=sample_webhook,
        )

        # Assert
        assert webhook_id == 'webhook-id-123'
        # Verify install_webhook was called with GROUP type
        call_args = mock_gitlab_service.install_webhook.call_args
        assert call_args[1]['resource_type'] == resource_type
        assert call_args[1]['resource_id'] == resource_id

    @pytest.mark.asyncio
    async def test_install_webhook_rate_limited(
        self, mock_gitlab_service, mock_webhook_store, sample_webhook
    ):
        """Test when installation is rate limited."""
        # Arrange
        resource_type = GitLabResourceType.PROJECT
        resource_id = 'project-123'
        mock_gitlab_service.install_webhook = AsyncMock(
            return_value=(None, WebhookStatus.RATE_LIMITED)
        )

        # Act & Assert
        with pytest.raises(BreakLoopException):
            await install_webhook_on_resource(
                gitlab_service=mock_gitlab_service,
                resource_type=resource_type,
                resource_id=resource_id,
                webhook_store=mock_webhook_store,
                webhook=sample_webhook,
            )

        # Should not update webhook on rate limit
        mock_webhook_store.update_webhook.assert_not_called()

    @pytest.mark.asyncio
    async def test_install_webhook_installation_fails(
        self, mock_gitlab_service, mock_webhook_store, sample_webhook
    ):
        """Test when webhook installation fails."""
        # Arrange
        resource_type = GitLabResourceType.PROJECT
        resource_id = 'project-123'
        mock_gitlab_service.install_webhook = AsyncMock(return_value=(None, None))

        # Act
        webhook_id, status = await install_webhook_on_resource(
            gitlab_service=mock_gitlab_service,
            resource_type=resource_type,
            resource_id=resource_id,
            webhook_store=mock_webhook_store,
            webhook=sample_webhook,
        )

        # Assert
        assert webhook_id is None
        assert status is None
        # Should not update webhook when installation fails
        mock_webhook_store.update_webhook.assert_not_called()

    @pytest.mark.asyncio
    async def test_install_webhook_generates_unique_secrets(
        self, mock_gitlab_service, mock_webhook_store, sample_webhook
    ):
        """Test that unique webhook secrets and UUIDs are generated."""
        # Arrange
        resource_type = GitLabResourceType.PROJECT
        resource_id = 'project-123'

        # Act - First call
        webhook_id1, _ = await install_webhook_on_resource(
            gitlab_service=mock_gitlab_service,
            resource_type=resource_type,
            resource_id=resource_id,
            webhook_store=mock_webhook_store,
            webhook=sample_webhook,
        )

        # Capture first call's values before resetting
        call1_secret = mock_webhook_store.update_webhook.call_args_list[0][1][
            'update_fields'
        ]['webhook_secret']
        call1_uuid = mock_webhook_store.update_webhook.call_args_list[0][1][
            'update_fields'
        ]['webhook_uuid']

        # Reset mocks and call again
        mock_gitlab_service.install_webhook.reset_mock()
        mock_webhook_store.update_webhook.reset_mock()

        # Act - Second call
        webhook_id2, _ = await install_webhook_on_resource(
            gitlab_service=mock_gitlab_service,
            resource_type=resource_type,
            resource_id=resource_id,
            webhook_store=mock_webhook_store,
            webhook=sample_webhook,
        )

        # Capture second call's values
        call2_secret = mock_webhook_store.update_webhook.call_args_list[0][1][
            'update_fields'
        ]['webhook_secret']
        call2_uuid = mock_webhook_store.update_webhook.call_args_list[0][1][
            'update_fields'
        ]['webhook_uuid']

        # Assert - Secrets and UUIDs should be different
        assert call1_secret != call2_secret
        assert call1_uuid != call2_uuid

    @pytest.mark.asyncio
    async def test_install_webhook_uses_correct_webhook_name_and_url(
        self, mock_gitlab_service, mock_webhook_store, sample_webhook
    ):
        """Test that correct webhook name and URL are used."""
        # Arrange
        resource_type = GitLabResourceType.PROJECT
        resource_id = 'project-123'

        # Act
        await install_webhook_on_resource(
            gitlab_service=mock_gitlab_service,
            resource_type=resource_type,
            resource_id=resource_id,
            webhook_store=mock_webhook_store,
            webhook=sample_webhook,
        )

        # Assert
        call_args = mock_gitlab_service.install_webhook.call_args
        assert call_args[1]['webhook_name'] == 'OpenHands Resolver'
        assert call_args[1]['webhook_url'] == GITLAB_WEBHOOK_URL
