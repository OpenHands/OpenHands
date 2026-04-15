"""Unit tests for SAAS-specific /api/v1/users endpoints.

Tests:
- SaasUserInfo model with org fields
- get_current_user_saas endpoint with org info
- _get_org_info_from_context helper function
"""

from unittest.mock import AsyncMock, MagicMock

import pytest


class TestSaasUserInfoModel:
    """Test suite for SaasUserInfo model."""

    def test_saas_user_info_with_all_org_fields(self):
        """SaasUserInfo should accept all org-related fields."""
        from server.models.user_models import SaasUserInfo

        user_info = SaasUserInfo(
            id='user-123',
            org_id='org-456',
            org_name='Test Organization',
            role='admin',
            permissions=['read', 'write', 'delete'],
        )

        assert user_info.id == 'user-123'
        assert user_info.org_id == 'org-456'
        assert user_info.org_name == 'Test Organization'
        assert user_info.role == 'admin'
        assert user_info.permissions == ['read', 'write', 'delete']

    def test_saas_user_info_without_org_fields(self):
        """SaasUserInfo should work without org fields (fallback mode)."""
        from server.models.user_models import SaasUserInfo

        user_info = SaasUserInfo(id='user-123')

        assert user_info.id == 'user-123'
        assert user_info.org_id is None
        assert user_info.org_name is None
        assert user_info.role is None
        assert user_info.permissions is None

    def test_saas_user_info_with_partial_org_fields(self):
        """SaasUserInfo should handle partial org fields (e.g., role is None)."""
        from server.models.user_models import SaasUserInfo

        user_info = SaasUserInfo(
            id='user-123',
            org_id='org-456',
            org_name='Test Organization',
            role=None,
            permissions=[],
        )

        assert user_info.org_id == 'org-456'
        assert user_info.org_name == 'Test Organization'
        assert user_info.role is None
        assert user_info.permissions == []

    def test_saas_user_info_model_dump_includes_org_fields(self):
        """SaasUserInfo model_dump should include org fields."""
        from server.models.user_models import SaasUserInfo

        user_info = SaasUserInfo(
            id='user-123',
            org_id='org-456',
            org_name='Test Organization',
            role='member',
            permissions=['read'],
        )

        data = user_info.model_dump()
        assert data['org_id'] == 'org-456'
        assert data['org_name'] == 'Test Organization'
        assert data['role'] == 'member'
        assert data['permissions'] == ['read']

    def test_saas_user_info_extends_base_user_info(self):
        """SaasUserInfo should inherit from UserInfo base class."""
        from server.models.user_models import SaasUserInfo

        from openhands.app_server.user.user_models import UserInfo

        assert issubclass(SaasUserInfo, UserInfo)


class TestGetOrgInfoFromContext:
    """Test suite for _get_org_info_from_context helper function."""

    @pytest.mark.asyncio
    async def test_returns_org_info_from_saas_user_auth(self):
        """Should return org info when context has SaasUserAuth."""
        from server.auth.saas_user_auth import SaasUserAuth
        from server.routes.users_v1 import _get_org_info_from_context

        from openhands.app_server.user.auth_user_context import AuthUserContext

        # Create a SaasUserAuth with mocked get_org_info
        mock_user_auth = MagicMock(spec=SaasUserAuth)
        mock_user_auth.get_org_info = AsyncMock(
            return_value={
                'org_id': 'org-456',
                'org_name': 'Test Organization',
                'role': 'admin',
                'permissions': ['read', 'write'],
            }
        )

        # Create AuthUserContext with the mock
        context = MagicMock(spec=AuthUserContext)
        context.user_auth = mock_user_auth

        org_info = await _get_org_info_from_context(context)

        assert org_info is not None
        assert org_info['org_id'] == 'org-456'
        assert org_info['org_name'] == 'Test Organization'
        assert org_info['role'] == 'admin'
        mock_user_auth.get_org_info.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_none_for_non_auth_user_context(self):
        """Should return None when context is not AuthUserContext."""
        from server.routes.users_v1 import _get_org_info_from_context

        from openhands.app_server.user.user_context import UserContext

        # Create a non-AuthUserContext
        mock_context = MagicMock(spec=UserContext)

        org_info = await _get_org_info_from_context(mock_context)

        assert org_info is None

    @pytest.mark.asyncio
    async def test_returns_none_for_non_saas_user_auth(self):
        """Should return None when user_auth is not SaasUserAuth."""
        from server.routes.users_v1 import _get_org_info_from_context

        from openhands.app_server.user.auth_user_context import AuthUserContext
        from openhands.server.user_auth.user_auth import UserAuth

        # Create AuthUserContext with a non-SaasUserAuth
        mock_user_auth = MagicMock(spec=UserAuth)
        mock_context = MagicMock(spec=AuthUserContext)
        mock_context.user_auth = mock_user_auth

        org_info = await _get_org_info_from_context(mock_context)

        assert org_info is None


class TestGetCurrentUserSaasEndpoint:
    """Test suite for get_current_user_saas endpoint."""

    @pytest.fixture
    def mock_user_context(self):
        """Create a mock user context."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_endpoint_returns_saas_user_info_with_org_fields(
        self, mock_user_context
    ):
        """Endpoint should return SaasUserInfo with org fields."""
        from unittest.mock import patch

        from server.models.user_models import SaasUserInfo
        from server.routes.users_v1 import get_current_user_saas

        from openhands.app_server.user.user_models import UserInfo

        # Mock base user info
        base_user_info = UserInfo(id='user-123', llm_model='test-model')
        mock_user_context.get_user_info = AsyncMock(return_value=base_user_info)

        # Mock _get_org_info_from_context to return org info
        org_info = {
            'org_id': 'org-456',
            'org_name': 'Test Organization',
            'role': 'member',
            'permissions': ['read', 'write'],
        }

        with patch(
            'server.routes.users_v1._get_org_info_from_context',
            return_value=org_info,
        ):
            result = await get_current_user_saas(
                user_context=mock_user_context, expose_secrets=False
            )

        assert isinstance(result, SaasUserInfo)
        assert result.id == 'user-123'
        assert result.org_id == 'org-456'
        assert result.org_name == 'Test Organization'
        assert result.role == 'member'
        assert result.permissions == ['read', 'write']

    @pytest.mark.asyncio
    async def test_endpoint_returns_saas_user_info_without_org_fields(
        self, mock_user_context
    ):
        """Endpoint should work when org info is not available."""
        from unittest.mock import patch

        from server.models.user_models import SaasUserInfo
        from server.routes.users_v1 import get_current_user_saas

        from openhands.app_server.user.user_models import UserInfo

        # Mock base user info
        base_user_info = UserInfo(id='user-123', llm_model='test-model')
        mock_user_context.get_user_info = AsyncMock(return_value=base_user_info)

        # Mock _get_org_info_from_context to return None
        with patch(
            'server.routes.users_v1._get_org_info_from_context',
            return_value=None,
        ):
            result = await get_current_user_saas(
                user_context=mock_user_context, expose_secrets=False
            )

        assert isinstance(result, SaasUserInfo)
        assert result.id == 'user-123'
        assert result.org_id is None
        assert result.org_name is None
        assert result.role is None
        assert result.permissions is None

    @pytest.mark.asyncio
    async def test_endpoint_raises_401_when_user_info_is_none(self, mock_user_context):
        """Endpoint should raise 401 when user info is None."""
        from fastapi import HTTPException
        from server.routes.users_v1 import get_current_user_saas

        mock_user_context.get_user_info = AsyncMock(return_value=None)

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user_saas(
                user_context=mock_user_context, expose_secrets=False
            )

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == 'Not authenticated'


class TestOverrideUsersEndpoint:
    """Test suite for override_users_me_endpoint function."""

    def test_override_removes_oss_route_and_adds_saas_route(self):
        """override_users_me_endpoint should remove OSS route and add SAAS route."""
        from fastapi import FastAPI
        from server.routes.users_v1 import override_users_me_endpoint

        # Create a minimal app with a mock OSS route
        app = FastAPI()

        @app.get('/api/v1/users/me')
        def mock_oss_endpoint():
            return {'source': 'oss'}

        # Verify OSS route exists
        oss_routes = [
            r for r in app.routes if hasattr(r, 'path') and r.path == '/api/v1/users/me'
        ]
        assert len(oss_routes) == 1
        assert oss_routes[0].endpoint.__name__ == 'mock_oss_endpoint'

        # Apply the override
        override_users_me_endpoint(app)

        # Verify SAAS route exists and OSS route is gone
        saas_routes = [
            r for r in app.routes if hasattr(r, 'path') and r.path == '/api/v1/users/me'
        ]
        assert len(saas_routes) == 1
        assert saas_routes[0].endpoint.__name__ == 'get_current_user_saas'
