"""Unit tests for the GET /users/git-info endpoint in user_router.py.

Tests:
- GET /api/v1/users/git-info
- get_current_user_git_info function
"""

from unittest.mock import AsyncMock

import pytest


class TestGetCurrentUserGitInfo:
    """Test suite for get_current_user_git_info function."""

    @pytest.fixture
    def mock_user_context(self):
        """Create a mock user context."""
        return AsyncMock()

    async def test_returns_user_git_info_when_authenticated(self, mock_user_context):
        """Authenticated user with git info returns the git info."""
        from openhands.integrations.service_types import UserGitInfo

        expected_git_info = UserGitInfo(
            id='user-123',
            login='testuser',
            avatar_url='https://example.com/avatar.png',
            company='Test Company',
            name='Test User',
            email='test@example.com',
        )
        mock_user_context.get_user_git_info = AsyncMock(return_value=expected_git_info)

        # Import inside test to avoid circular imports at collection time
        from openhands.app_server.user.user_router import get_current_user_git_info

        result = await get_current_user_git_info(user_context=mock_user_context)

        mock_user_context.get_user_git_info.assert_called_once()
        assert result.id == 'user-123'
        assert result.login == 'testuser'
        assert result.avatar_url == 'https://example.com/avatar.png'
        assert result.company == 'Test Company'
        assert result.name == 'Test User'
        assert result.email == 'test@example.com'

    async def test_returns_user_git_info_with_minimal_fields(self, mock_user_context):
        """User git info with only required fields returns successfully."""
        from openhands.integrations.service_types import UserGitInfo

        minimal_git_info = UserGitInfo(
            id='user-456',
            login='minimaluser',
            avatar_url='https://example.com/default.png',
        )
        mock_user_context.get_user_git_info = AsyncMock(return_value=minimal_git_info)

        from openhands.app_server.user.user_router import get_current_user_git_info

        result = await get_current_user_git_info(user_context=mock_user_context)

        assert result.id == 'user-456'
        assert result.login == 'minimaluser'
        assert result.avatar_url == 'https://example.com/default.png'
        assert result.company is None
        assert result.name is None
        assert result.email is None

    async def test_raises_401_when_user_git_info_is_none(self, mock_user_context):
        """When get_user_git_info returns None, raises 401 Unauthorized."""
        from fastapi import HTTPException

        mock_user_context.get_user_git_info = AsyncMock(return_value=None)

        from openhands.app_server.user.user_router import get_current_user_git_info

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user_git_info(user_context=mock_user_context)

        assert exc_info.value.status_code == 401
        assert 'Not authenticated' in exc_info.value.detail
        mock_user_context.get_user_git_info.assert_called_once()

    async def test_raises_401_when_user_git_info_returns_none_for_company(
        self, mock_user_context
    ):
        """Ensure None values in optional fields don't trigger 401."""
        from openhands.integrations.service_types import UserGitInfo

        git_info = UserGitInfo(
            id='user-789',
            login='companyuser',
            avatar_url='https://example.com/company.png',
            company=None,
            name=None,
            email=None,
        )
        mock_user_context.get_user_git_info = AsyncMock(return_value=git_info)

        from openhands.app_server.user.user_router import get_current_user_git_info

        result = await get_current_user_git_info(user_context=mock_user_context)

        # Should NOT raise 401 - None optional fields are valid
        assert result.id == 'user-789'
        assert result.login == 'companyuser'

    async def test_propagates_exceptions_from_user_context(self, mock_user_context):
        """When get_user_git_info raises an exception, it should propagate."""
        mock_user_context.get_user_git_info = AsyncMock(
            side_effect=Exception('Database error')
        )

        from openhands.app_server.user.user_router import get_current_user_git_info

        with pytest.raises(Exception, match='Database error'):
            await get_current_user_git_info(user_context=mock_user_context)


class TestUserGitInfoModel:
    """Tests for the UserGitInfo model itself."""

    def test_user_git_info_full_fields(self):
        """Test UserGitInfo with all fields populated."""
        from openhands.integrations.service_types import UserGitInfo

        git_info = UserGitInfo(
            id='full-user',
            login='fulllogin',
            avatar_url='https://example.com/full.png',
            company='Full Company',
            name='Full Name',
            email='full@example.com',
        )

        assert git_info.id == 'full-user'
        assert git_info.login == 'fulllogin'
        assert git_info.avatar_url == 'https://example.com/full.png'
        assert git_info.company == 'Full Company'
        assert git_info.name == 'Full Name'
        assert git_info.email == 'full@example.com'

    def test_user_git_info_model_dump_json(self):
        """Test UserGitInfo serializes correctly to JSON."""
        from openhands.integrations.service_types import UserGitInfo

        git_info = UserGitInfo(
            id='json-user',
            login='jsonlogin',
            avatar_url='https://example.com/json.png',
            company='JSON Corp',
            name='JSON Name',
            email='json@example.com',
        )

        json_str = git_info.model_dump_json()
        # Check for key fields in the JSON output (Pydantic uses compact JSON by default)
        assert '"id":"json-user"' in json_str
        assert '"login":"jsonlogin"' in json_str
        assert '"company":"JSON Corp"' in json_str
        assert '"name":"JSON Name"' in json_str
        assert '"email":"json@example.com"' in json_str

    def test_user_git_info_model_dump(self):
        """Test UserGitInfo model_dump works correctly."""
        from openhands.integrations.service_types import UserGitInfo

        git_info = UserGitInfo(
            id='dump-user',
            login='dumplogin',
            avatar_url='https://example.com/dump.png',
        )

        data = git_info.model_dump()
        assert data['id'] == 'dump-user'
        assert data['login'] == 'dumplogin'
        assert data['avatar_url'] == 'https://example.com/dump.png'
        assert data['company'] is None
        assert data['name'] is None
        assert data['email'] is None
