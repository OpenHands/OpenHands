"""
Tests for JiraIntegrationStore async methods.

The store uses async database sessions (a_session_maker) for all operations,
which is critical for avoiding asyncpg event loop issues when called from
FastAPI async endpoints.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock, patch

import pytest
from storage.jira_integration_store import JiraIntegrationStore
from storage.jira_user import JiraUser
from storage.jira_workspace import JiraWorkspace


@pytest.fixture
def store():
    """Create a JiraIntegrationStore instance."""
    return JiraIntegrationStore()


@pytest.fixture
def create_mock_async_session():
    """Factory to create properly mocked async session context manager."""

    def _create(query_result=None, all_results=None):
        mock_session = Mock()
        mock_result = Mock()

        if all_results is not None:
            mock_result.scalars.return_value.all.return_value = all_results
        else:
            mock_result.scalars.return_value.first.return_value = query_result

        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.add = Mock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        @asynccontextmanager
        async def mock_context_manager():
            yield mock_session

        return mock_context_manager, mock_session

    return _create


class TestJiraIntegrationStoreAsyncMethods:
    """Tests verifying JiraIntegrationStore methods use async sessions correctly."""

    @pytest.mark.asyncio
    async def test_get_workspace_by_id_returns_workspace(
        self, store, create_mock_async_session
    ):
        """Test get_workspace_by_id returns workspace when found."""
        # Arrange
        mock_workspace = Mock(spec=JiraWorkspace)
        mock_workspace.id = 1
        mock_workspace.name = 'test-workspace'

        mock_context_manager, mock_session = create_mock_async_session(mock_workspace)

        # Act
        with patch(
            'storage.jira_integration_store.a_session_maker', mock_context_manager
        ):
            result = await store.get_workspace_by_id(1)

        # Assert
        assert result == mock_workspace
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_workspace_by_id_returns_none_when_not_found(
        self, store, create_mock_async_session
    ):
        """Test get_workspace_by_id returns None when workspace not found."""
        # Arrange
        mock_context_manager, mock_session = create_mock_async_session(None)

        # Act
        with patch(
            'storage.jira_integration_store.a_session_maker', mock_context_manager
        ):
            result = await store.get_workspace_by_id(999)

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_get_workspace_by_name_normalizes_to_lowercase(
        self, store, create_mock_async_session
    ):
        """Test get_workspace_by_name converts name to lowercase for query."""
        # Arrange
        mock_workspace = Mock(spec=JiraWorkspace)
        mock_workspace.name = 'test-workspace'

        mock_context_manager, mock_session = create_mock_async_session(mock_workspace)

        # Act
        with patch(
            'storage.jira_integration_store.a_session_maker', mock_context_manager
        ):
            result = await store.get_workspace_by_name('TEST-WORKSPACE')

        # Assert
        assert result == mock_workspace
        # Verify the query was executed (filter includes lowercase conversion)
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_active_user_filters_by_status(
        self, store, create_mock_async_session
    ):
        """Test get_active_user only returns users with active status."""
        # Arrange
        mock_user = Mock(spec=JiraUser)
        mock_user.jira_user_id = 'jira-123'
        mock_user.jira_workspace_id = 1
        mock_user.status = 'active'

        mock_context_manager, mock_session = create_mock_async_session(mock_user)

        # Act
        with patch(
            'storage.jira_integration_store.a_session_maker', mock_context_manager
        ):
            result = await store.get_active_user('jira-123', 1)

        # Assert
        assert result == mock_user
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_workspace_adds_and_commits(
        self, store, create_mock_async_session
    ):
        """Test create_workspace properly adds, commits, and refreshes."""
        # Arrange
        mock_context_manager, mock_session = create_mock_async_session(None)

        # Act
        with patch(
            'storage.jira_integration_store.a_session_maker', mock_context_manager
        ):
            await store.create_workspace(
                name='TEST-WORKSPACE',
                jira_cloud_id='cloud-123',
                admin_user_id='admin-user',
                encrypted_webhook_secret='encrypted-secret',
                svc_acc_email='svc@test.com',
                encrypted_svc_acc_api_key='encrypted-key',
                status='active',
            )

        # Assert
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.refresh.assert_called_once()

        # Verify workspace was created with lowercase name
        added_workspace = mock_session.add.call_args[0][0]
        assert added_workspace.name == 'test-workspace'

    @pytest.mark.asyncio
    async def test_update_user_integration_status_raises_if_not_found(
        self, store, create_mock_async_session
    ):
        """Test update_user_integration_status raises ValueError if user not found."""
        # Arrange
        mock_context_manager, mock_session = create_mock_async_session(None)

        # Act & Assert
        with patch(
            'storage.jira_integration_store.a_session_maker', mock_context_manager
        ):
            with pytest.raises(ValueError) as exc_info:
                await store.update_user_integration_status('unknown-user', 'inactive')

            assert 'Jira user not found' in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_deactivate_workspace_deactivates_all_users(
        self, store, create_mock_async_session
    ):
        """Test deactivate_workspace sets all users and workspace to inactive."""
        # Arrange
        mock_user1 = Mock(spec=JiraUser)
        mock_user1.status = 'active'
        mock_user2 = Mock(spec=JiraUser)
        mock_user2.status = 'active'

        mock_workspace = Mock(spec=JiraWorkspace)
        mock_workspace.status = 'active'

        mock_session = Mock()

        # First execute returns users, second returns workspace
        call_count = [0]

        def execute_side_effect(*args, **kwargs):
            result = Mock()
            if call_count[0] == 0:
                result.scalars.return_value.all.return_value = [mock_user1, mock_user2]
            else:
                result.scalars.return_value.first.return_value = mock_workspace
            call_count[0] += 1
            return result

        mock_session.execute = AsyncMock(side_effect=execute_side_effect)
        mock_session.add = Mock()
        mock_session.commit = AsyncMock()

        @asynccontextmanager
        async def mock_context_manager():
            yield mock_session

        # Act
        with patch(
            'storage.jira_integration_store.a_session_maker', mock_context_manager
        ):
            await store.deactivate_workspace(1)

        # Assert
        assert mock_user1.status == 'inactive'
        assert mock_user2.status == 'inactive'
        assert mock_workspace.status == 'inactive'
        mock_session.commit.assert_called_once()
