"""Unit tests for SaaSGitLabService."""

from unittest.mock import patch

import pytest
from integrations.gitlab.gitlab_service import SaaSGitLabService


@pytest.fixture
def gitlab_service():
    """Create a SaaSGitLabService instance for testing."""
    return SaaSGitLabService(external_auth_id='test_user_id')


class TestGetUserResourcesWithAdminAccess:
    """Test cases for get_user_resources_with_admin_access method."""

    @pytest.mark.asyncio
    async def test_get_resources_single_page_projects_and_groups(self, gitlab_service):
        """Test fetching resources when all data fits in a single page."""
        # Arrange
        mock_projects = [
            {'id': 1, 'name': 'Project 1'},
            {'id': 2, 'name': 'Project 2'},
        ]
        mock_groups = [
            {'id': 10, 'name': 'Group 1'},
        ]

        with patch.object(gitlab_service, '_make_request') as mock_request:
            # First call for projects, second for groups
            mock_request.side_effect = [
                (mock_projects, {'Link': ''}),  # No next page
                (mock_groups, {'Link': ''}),  # No next page
            ]

            # Act
            (
                projects,
                groups,
            ) = await gitlab_service.get_user_resources_with_admin_access()

            # Assert
            assert len(projects) == 2
            assert len(groups) == 1
            assert projects[0]['id'] == 1
            assert projects[1]['id'] == 2
            assert groups[0]['id'] == 10
            assert mock_request.call_count == 2

    @pytest.mark.asyncio
    async def test_get_resources_multiple_pages_projects(self, gitlab_service):
        """Test fetching projects across multiple pages."""
        # Arrange
        page1_projects = [{'id': i, 'name': f'Project {i}'} for i in range(1, 101)]
        page2_projects = [{'id': i, 'name': f'Project {i}'} for i in range(101, 151)]

        with patch.object(gitlab_service, '_make_request') as mock_request:
            mock_request.side_effect = [
                (page1_projects, {'Link': '<url>; rel="next"'}),  # Has next page
                (page2_projects, {'Link': ''}),  # Last page
                ([], {'Link': ''}),  # Groups (empty)
            ]

            # Act
            (
                projects,
                groups,
            ) = await gitlab_service.get_user_resources_with_admin_access()

            # Assert
            assert len(projects) == 150
            assert len(groups) == 0
            assert mock_request.call_count == 3

    @pytest.mark.asyncio
    async def test_get_resources_multiple_pages_groups(self, gitlab_service):
        """Test fetching groups across multiple pages."""
        # Arrange
        page1_groups = [{'id': i, 'name': f'Group {i}'} for i in range(1, 101)]
        page2_groups = [{'id': i, 'name': f'Group {i}'} for i in range(101, 151)]

        with patch.object(gitlab_service, '_make_request') as mock_request:
            mock_request.side_effect = [
                ([], {'Link': ''}),  # Projects (empty)
                (page1_groups, {'Link': '<url>; rel="next"'}),  # Has next page
                (page2_groups, {'Link': ''}),  # Last page
            ]

            # Act
            (
                projects,
                groups,
            ) = await gitlab_service.get_user_resources_with_admin_access()

            # Assert
            assert len(projects) == 0
            assert len(groups) == 150
            assert mock_request.call_count == 3

    @pytest.mark.asyncio
    async def test_get_resources_empty_response(self, gitlab_service):
        """Test when user has no projects or groups with admin access."""
        # Arrange
        with patch.object(gitlab_service, '_make_request') as mock_request:
            mock_request.side_effect = [
                ([], {'Link': ''}),  # No projects
                ([], {'Link': ''}),  # No groups
            ]

            # Act
            (
                projects,
                groups,
            ) = await gitlab_service.get_user_resources_with_admin_access()

            # Assert
            assert len(projects) == 0
            assert len(groups) == 0
            assert mock_request.call_count == 2

    @pytest.mark.asyncio
    async def test_get_resources_uses_correct_params_for_projects(self, gitlab_service):
        """Test that projects API is called with correct parameters."""
        # Arrange
        with patch.object(gitlab_service, '_make_request') as mock_request:
            mock_request.side_effect = [
                ([], {'Link': ''}),  # Projects
                ([], {'Link': ''}),  # Groups
            ]

            # Act
            await gitlab_service.get_user_resources_with_admin_access()

            # Assert
            # Check first call (projects)
            first_call = mock_request.call_args_list[0]
            assert 'projects' in first_call[0][0]
            assert first_call[0][1]['membership'] == 1
            assert first_call[0][1]['min_access_level'] == 40
            assert first_call[0][1]['per_page'] == '100'

    @pytest.mark.asyncio
    async def test_get_resources_uses_correct_params_for_groups(self, gitlab_service):
        """Test that groups API is called with correct parameters."""
        # Arrange
        with patch.object(gitlab_service, '_make_request') as mock_request:
            mock_request.side_effect = [
                ([], {'Link': ''}),  # Projects
                ([], {'Link': ''}),  # Groups
            ]

            # Act
            await gitlab_service.get_user_resources_with_admin_access()

            # Assert
            # Check second call (groups)
            second_call = mock_request.call_args_list[1]
            assert 'groups' in second_call[0][0]
            assert second_call[0][1]['min_access_level'] == 40
            assert second_call[0][1]['top_level_only'] == 'true'
            assert second_call[0][1]['per_page'] == '100'

    @pytest.mark.asyncio
    async def test_get_resources_handles_api_error_gracefully(self, gitlab_service):
        """Test that API errors are handled gracefully and don't crash."""
        # Arrange
        with patch.object(gitlab_service, '_make_request') as mock_request:
            # First call succeeds, second call fails
            mock_request.side_effect = [
                ([{'id': 1, 'name': 'Project 1'}], {'Link': ''}),
                Exception('API Error'),
            ]

            # Act
            (
                projects,
                groups,
            ) = await gitlab_service.get_user_resources_with_admin_access()

            # Assert
            # Should return what was fetched before the error
            assert len(projects) == 1
            assert len(groups) == 0

    @pytest.mark.asyncio
    async def test_get_resources_stops_on_empty_response(self, gitlab_service):
        """Test that pagination stops when API returns empty response."""
        # Arrange
        with patch.object(gitlab_service, '_make_request') as mock_request:
            mock_request.side_effect = [
                (None, {'Link': ''}),  # Empty response stops pagination
                ([], {'Link': ''}),  # Groups
            ]

            # Act
            (
                projects,
                groups,
            ) = await gitlab_service.get_user_resources_with_admin_access()

            # Assert
            assert len(projects) == 0
            assert mock_request.call_count == 2  # Should not continue pagination
