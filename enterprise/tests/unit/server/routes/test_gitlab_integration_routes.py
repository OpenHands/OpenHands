"""Unit tests for GitLab integration routes."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, status
from integrations.gitlab.gitlab_service import SaaSGitLabService
from integrations.gitlab.webhook_installation import BreakLoopException
from integrations.types import GitLabResourceType
from server.routes.integration.gitlab import (
    ReinstallWebhookRequest,
    ResourceIdentifier,
    get_gitlab_resources,
    reinstall_gitlab_webhook,
)
from storage.gitlab_webhook import GitlabWebhook


@pytest.fixture
def mock_gitlab_service():
    """Create a mock SaaSGitLabService instance."""
    service = MagicMock(spec=SaaSGitLabService)
    service.get_user_resources_with_admin_access = AsyncMock(
        return_value=(
            [
                {
                    'id': 1,
                    'name': 'Test Project',
                    'path_with_namespace': 'user/test-project',
                    'namespace': {'kind': 'user'},
                },
                {
                    'id': 2,
                    'name': 'Group Project',
                    'path_with_namespace': 'group/group-project',
                    'namespace': {'kind': 'group'},
                },
            ],
            [
                {
                    'id': 10,
                    'name': 'Test Group',
                    'full_path': 'test-group',
                },
            ],
        )
    )
    service.check_webhook_exists_on_resource = AsyncMock(return_value=(True, None))
    service.check_user_has_admin_access_to_resource = AsyncMock(
        return_value=(True, None)
    )
    return service


@pytest.fixture
def mock_webhook():
    """Create a mock webhook object."""
    webhook = MagicMock(spec=GitlabWebhook)
    webhook.webhook_uuid = 'test-uuid'
    webhook.last_synced = None
    return webhook


class TestGetGitLabResources:
    """Test cases for get_gitlab_resources endpoint."""

    @pytest.mark.asyncio
    @patch('server.routes.integration.gitlab.GitLabServiceImpl')
    @patch('server.routes.integration.gitlab.webhook_store')
    @patch('server.routes.integration.gitlab.isinstance')
    async def test_get_resources_success(
        self,
        mock_isinstance,
        mock_webhook_store,
        mock_gitlab_service_impl,
        mock_gitlab_service,
    ):
        """Test successfully retrieving GitLab resources with webhook status."""
        # Arrange
        user_id = 'test_user_id'
        mock_gitlab_service_impl.return_value = mock_gitlab_service
        mock_isinstance.return_value = True
        mock_webhook_store.get_webhooks_by_resources = AsyncMock(
            return_value=({}, {})  # Empty maps for simplicity
        )

        # Act
        response = await get_gitlab_resources(user_id=user_id)

        # Assert
        assert len(response.resources) == 2  # 1 project (filtered) + 1 group
        assert response.resources[0].type == 'project'
        assert response.resources[0].id == '1'
        assert response.resources[0].name == 'Test Project'
        assert response.resources[1].type == 'group'
        assert response.resources[1].id == '10'
        mock_gitlab_service.get_user_resources_with_admin_access.assert_called_once()
        mock_webhook_store.get_webhooks_by_resources.assert_called_once()

    @pytest.mark.asyncio
    @patch('server.routes.integration.gitlab.GitLabServiceImpl')
    @patch('server.routes.integration.gitlab.webhook_store')
    @patch('server.routes.integration.gitlab.isinstance')
    async def test_get_resources_filters_nested_projects(
        self,
        mock_isinstance,
        mock_webhook_store,
        mock_gitlab_service_impl,
        mock_gitlab_service,
    ):
        """Test that projects nested under groups are filtered out."""
        # Arrange
        user_id = 'test_user_id'
        mock_gitlab_service_impl.return_value = mock_gitlab_service
        mock_isinstance.return_value = True
        mock_webhook_store.get_webhooks_by_resources = AsyncMock(return_value=({}, {}))

        # Act
        response = await get_gitlab_resources(user_id=user_id)

        # Assert
        # Should only include the user project, not the group project
        project_resources = [r for r in response.resources if r.type == 'project']
        assert len(project_resources) == 1
        assert project_resources[0].id == '1'
        assert project_resources[0].name == 'Test Project'

    @pytest.mark.asyncio
    @patch('server.routes.integration.gitlab.GitLabServiceImpl')
    @patch('server.routes.integration.gitlab.webhook_store')
    @patch('server.routes.integration.gitlab.isinstance')
    async def test_get_resources_includes_webhook_metadata(
        self,
        mock_isinstance,
        mock_webhook_store,
        mock_gitlab_service_impl,
        mock_gitlab_service,
        mock_webhook,
    ):
        """Test that webhook metadata is included in the response."""
        # Arrange
        user_id = 'test_user_id'
        mock_gitlab_service_impl.return_value = mock_gitlab_service
        mock_isinstance.return_value = True
        mock_webhook_store.get_webhooks_by_resources = AsyncMock(
            return_value=({'1': mock_webhook}, {'10': mock_webhook})
        )

        # Act
        response = await get_gitlab_resources(user_id=user_id)

        # Assert
        assert response.resources[0].webhook_uuid == 'test-uuid'
        assert response.resources[1].webhook_uuid == 'test-uuid'

    @pytest.mark.asyncio
    @patch('server.routes.integration.gitlab.GitLabServiceImpl')
    async def test_get_resources_non_saas_service(
        self, mock_gitlab_service_impl, mock_gitlab_service
    ):
        """Test that non-SaaS GitLab service raises an error."""
        # Arrange
        user_id = 'test_user_id'
        non_saas_service = AsyncMock()
        mock_gitlab_service_impl.return_value = non_saas_service

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await get_gitlab_resources(user_id=user_id)

        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert 'Only SaaS GitLab service is supported' in exc_info.value.detail

    @pytest.mark.asyncio
    @patch('server.routes.integration.gitlab.GitLabServiceImpl')
    @patch('server.routes.integration.gitlab.webhook_store')
    @patch('server.routes.integration.gitlab.isinstance')
    async def test_get_resources_parallel_api_calls(
        self,
        mock_isinstance,
        mock_webhook_store,
        mock_gitlab_service_impl,
        mock_gitlab_service,
    ):
        """Test that webhook status checks are made in parallel."""
        # Arrange
        user_id = 'test_user_id'
        mock_gitlab_service_impl.return_value = mock_gitlab_service
        mock_isinstance.return_value = True
        mock_webhook_store.get_webhooks_by_resources = AsyncMock(return_value=({}, {}))
        call_count = 0

        async def track_calls(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return (True, None)

        mock_gitlab_service.check_webhook_exists_on_resource = AsyncMock(
            side_effect=track_calls
        )

        # Act
        await get_gitlab_resources(user_id=user_id)

        # Assert
        # Should be called for each resource (1 project + 1 group)
        assert call_count == 2


class TestReinstallGitLabWebhook:
    """Test cases for reinstall_gitlab_webhook endpoint."""

    @pytest.mark.asyncio
    @patch('server.routes.integration.gitlab.install_webhook_on_resource')
    @patch('server.routes.integration.gitlab.verify_webhook_conditions')
    @patch('server.routes.integration.gitlab.GitLabServiceImpl')
    @patch('server.routes.integration.gitlab.webhook_store')
    @patch('server.routes.integration.gitlab.isinstance')
    async def test_reinstall_webhook_success_existing_webhook(
        self,
        mock_isinstance,
        mock_webhook_store,
        mock_gitlab_service_impl,
        mock_verify_conditions,
        mock_install_webhook,
        mock_gitlab_service,
        mock_webhook,
    ):
        """Test successful webhook reinstallation when webhook record exists."""
        # Arrange
        user_id = 'test_user_id'
        resource_id = 'project-123'
        resource_type = GitLabResourceType.PROJECT

        mock_gitlab_service_impl.return_value = mock_gitlab_service
        mock_isinstance.return_value = True
        mock_webhook_store.reset_webhook_for_reinstallation_by_resource = AsyncMock(
            return_value=True
        )
        mock_webhook_store.get_webhook_by_resource_only = AsyncMock(
            return_value=mock_webhook
        )
        mock_verify_conditions.return_value = None
        mock_install_webhook.return_value = ('webhook-id-123', None)

        body = ReinstallWebhookRequest(
            resource=ResourceIdentifier(type=resource_type, id=resource_id)
        )

        # Act
        result = await reinstall_gitlab_webhook(body=body, user_id=user_id)

        # Assert
        assert result.success is True
        assert result.resource_id == resource_id
        assert result.resource_type == resource_type.value
        assert result.error is None
        mock_gitlab_service.check_user_has_admin_access_to_resource.assert_called_once_with(
            resource_type, resource_id
        )
        mock_webhook_store.reset_webhook_for_reinstallation_by_resource.assert_called_once_with(
            resource_type, resource_id, user_id
        )
        mock_verify_conditions.assert_called_once()
        mock_install_webhook.assert_called_once()

    @pytest.mark.asyncio
    @patch('server.routes.integration.gitlab.install_webhook_on_resource')
    @patch('server.routes.integration.gitlab.verify_webhook_conditions')
    @patch('server.routes.integration.gitlab.GitLabServiceImpl')
    @patch('server.routes.integration.gitlab.webhook_store')
    @patch('server.routes.integration.gitlab.isinstance')
    async def test_reinstall_webhook_success_new_webhook_record(
        self,
        mock_isinstance,
        mock_webhook_store,
        mock_gitlab_service_impl,
        mock_verify_conditions,
        mock_install_webhook,
        mock_gitlab_service,
    ):
        """Test successful webhook reinstallation when webhook record doesn't exist."""
        # Arrange
        user_id = 'test_user_id'
        resource_id = 'project-456'
        resource_type = GitLabResourceType.PROJECT

        mock_gitlab_service_impl.return_value = mock_gitlab_service
        mock_isinstance.return_value = True
        mock_webhook_store.reset_webhook_for_reinstallation_by_resource = (
            AsyncMock(return_value=False)  # No existing webhook to reset
        )
        mock_webhook_store.get_webhook_by_resource_only = AsyncMock(
            side_effect=[
                None,
                MagicMock(),
            ]  # First call returns None, second returns new webhook
        )
        mock_webhook_store.store_webhooks = AsyncMock()
        mock_verify_conditions.return_value = None
        mock_install_webhook.return_value = ('webhook-id-456', None)

        body = ReinstallWebhookRequest(
            resource=ResourceIdentifier(type=resource_type, id=resource_id)
        )

        # Act
        result = await reinstall_gitlab_webhook(body=body, user_id=user_id)

        # Assert
        assert result.success is True
        mock_webhook_store.store_webhooks.assert_called_once()
        # Should fetch webhook twice: once to check, once after creating
        assert mock_webhook_store.get_webhook_by_resource_only.call_count == 2

    @pytest.mark.asyncio
    @patch('server.routes.integration.gitlab.GitLabServiceImpl')
    @patch('server.routes.integration.gitlab.isinstance')
    async def test_reinstall_webhook_no_admin_access(
        self, mock_isinstance, mock_gitlab_service_impl, mock_gitlab_service
    ):
        """Test reinstallation when user doesn't have admin access."""
        # Arrange
        user_id = 'test_user_id'
        resource_id = 'project-789'
        resource_type = GitLabResourceType.PROJECT

        mock_gitlab_service_impl.return_value = mock_gitlab_service
        mock_isinstance.return_value = True
        mock_gitlab_service.check_user_has_admin_access_to_resource = AsyncMock(
            return_value=(False, None)
        )

        body = ReinstallWebhookRequest(
            resource=ResourceIdentifier(type=resource_type, id=resource_id)
        )

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await reinstall_gitlab_webhook(body=body, user_id=user_id)

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert 'does not have admin access' in exc_info.value.detail

    @pytest.mark.asyncio
    @patch('server.routes.integration.gitlab.GitLabServiceImpl')
    async def test_reinstall_webhook_non_saas_service(self, mock_gitlab_service_impl):
        """Test reinstallation with non-SaaS GitLab service."""
        # Arrange
        user_id = 'test_user_id'
        resource_id = 'project-999'
        resource_type = GitLabResourceType.PROJECT

        non_saas_service = AsyncMock()
        mock_gitlab_service_impl.return_value = non_saas_service

        body = ReinstallWebhookRequest(
            resource=ResourceIdentifier(type=resource_type, id=resource_id)
        )

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await reinstall_gitlab_webhook(body=body, user_id=user_id)

        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert 'Only SaaS GitLab service is supported' in exc_info.value.detail

    @pytest.mark.asyncio
    @patch('server.routes.integration.gitlab.install_webhook_on_resource')
    @patch('server.routes.integration.gitlab.verify_webhook_conditions')
    @patch('server.routes.integration.gitlab.GitLabServiceImpl')
    @patch('server.routes.integration.gitlab.webhook_store')
    @patch('server.routes.integration.gitlab.isinstance')
    async def test_reinstall_webhook_conditions_not_met(
        self,
        mock_isinstance,
        mock_webhook_store,
        mock_gitlab_service_impl,
        mock_verify_conditions,
        mock_install_webhook,
        mock_gitlab_service,
        mock_webhook,
    ):
        """Test reinstallation when webhook conditions are not met."""
        # Arrange
        user_id = 'test_user_id'
        resource_id = 'project-111'
        resource_type = GitLabResourceType.PROJECT

        mock_gitlab_service_impl.return_value = mock_gitlab_service
        mock_isinstance.return_value = True
        mock_webhook_store.reset_webhook_for_reinstallation_by_resource = AsyncMock(
            return_value=True
        )
        mock_webhook_store.get_webhook_by_resource_only = AsyncMock(
            return_value=mock_webhook
        )
        mock_verify_conditions.side_effect = BreakLoopException()

        body = ReinstallWebhookRequest(
            resource=ResourceIdentifier(type=resource_type, id=resource_id)
        )

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await reinstall_gitlab_webhook(body=body, user_id=user_id)

        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert 'conditions not met' in exc_info.value.detail.lower()
        mock_install_webhook.assert_not_called()

    @pytest.mark.asyncio
    @patch('server.routes.integration.gitlab.install_webhook_on_resource')
    @patch('server.routes.integration.gitlab.verify_webhook_conditions')
    @patch('server.routes.integration.gitlab.GitLabServiceImpl')
    @patch('server.routes.integration.gitlab.webhook_store')
    @patch('server.routes.integration.gitlab.isinstance')
    async def test_reinstall_webhook_installation_fails(
        self,
        mock_isinstance,
        mock_webhook_store,
        mock_gitlab_service_impl,
        mock_verify_conditions,
        mock_install_webhook,
        mock_gitlab_service,
        mock_webhook,
    ):
        """Test reinstallation when webhook installation fails."""
        # Arrange
        user_id = 'test_user_id'
        resource_id = 'project-222'
        resource_type = GitLabResourceType.PROJECT

        mock_gitlab_service_impl.return_value = mock_gitlab_service
        mock_isinstance.return_value = True
        mock_webhook_store.reset_webhook_for_reinstallation_by_resource = AsyncMock(
            return_value=True
        )
        mock_webhook_store.get_webhook_by_resource_only = AsyncMock(
            return_value=mock_webhook
        )
        mock_verify_conditions.return_value = None
        mock_install_webhook.return_value = (None, None)  # Installation failed

        body = ReinstallWebhookRequest(
            resource=ResourceIdentifier(type=resource_type, id=resource_id)
        )

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await reinstall_gitlab_webhook(body=body, user_id=user_id)

        assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert 'Failed to install webhook' in exc_info.value.detail

    @pytest.mark.asyncio
    @patch('server.routes.integration.gitlab.install_webhook_on_resource')
    @patch('server.routes.integration.gitlab.verify_webhook_conditions')
    @patch('server.routes.integration.gitlab.GitLabServiceImpl')
    @patch('server.routes.integration.gitlab.webhook_store')
    @patch('server.routes.integration.gitlab.isinstance')
    async def test_reinstall_webhook_group_resource(
        self,
        mock_isinstance,
        mock_webhook_store,
        mock_gitlab_service_impl,
        mock_verify_conditions,
        mock_install_webhook,
        mock_gitlab_service,
        mock_webhook,
    ):
        """Test reinstallation for a group resource."""
        # Arrange
        user_id = 'test_user_id'
        resource_id = 'group-333'
        resource_type = GitLabResourceType.GROUP

        mock_gitlab_service_impl.return_value = mock_gitlab_service
        mock_isinstance.return_value = True
        mock_webhook_store.reset_webhook_for_reinstallation_by_resource = AsyncMock(
            return_value=True
        )
        mock_webhook_store.get_webhook_by_resource_only = AsyncMock(
            return_value=mock_webhook
        )
        mock_verify_conditions.return_value = None
        mock_install_webhook.return_value = ('webhook-id-group', None)

        body = ReinstallWebhookRequest(
            resource=ResourceIdentifier(type=resource_type, id=resource_id)
        )

        # Act
        result = await reinstall_gitlab_webhook(body=body, user_id=user_id)

        # Assert
        assert result.success is True
        assert result.resource_id == resource_id
        assert result.resource_type == resource_type.value
        mock_webhook_store.reset_webhook_for_reinstallation_by_resource.assert_called_once_with(
            resource_type, resource_id, user_id
        )
