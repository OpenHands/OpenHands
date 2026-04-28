"""Tests for onboarding-related auth routes and functions.

Tests for:
- _should_redirect_to_onboarding() function
- _get_post_auth_redirect() function
- /complete_onboarding endpoint
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request, status
from fastapi.responses import JSONResponse
from server.auth.saas_user_auth import SaasUserAuth
from server.routes.auth import (
    _get_post_auth_redirect,
    _should_redirect_to_onboarding,
    complete_onboarding,
)
from storage.user import User

# --- Fixtures ---


@pytest.fixture
def mock_request():
    """Create a mock FastAPI Request."""
    request = MagicMock(spec=Request)
    request.url = MagicMock()
    request.url.hostname = 'localhost'
    request.url.netloc = 'localhost:8000'
    request.base_url = 'http://localhost:8000/'
    request.headers = {}
    request.cookies = {}
    return request


@pytest.fixture
def mock_user():
    """Create a mock User object."""
    user = MagicMock(spec=User)
    user.id = uuid.uuid4()
    user.current_org_id = uuid.uuid4()
    user.onboarding_completed = False
    return user


# --- Tests for _should_redirect_to_onboarding ---


class TestShouldRedirectToOnboarding:
    """Tests for the _should_redirect_to_onboarding function."""

    @pytest.mark.asyncio
    async def test_returns_false_when_onboarding_completed(self, mock_user):
        """Test that completed onboarding users are not redirected."""
        mock_user.onboarding_completed = True

        result = await _should_redirect_to_onboarding('user-123', mock_user)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_true_for_cloud_mode_new_user(self, mock_user):
        """Test that cloud mode users with incomplete onboarding are redirected."""
        mock_user.onboarding_completed = False

        with patch('server.routes.auth.DEPLOYMENT_MODE', 'cloud'):
            result = await _should_redirect_to_onboarding('user-123', mock_user)

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_true_for_self_hosted_super_admin(self, mock_user):
        """Test that the super admin (first owner to accept TOS) is redirected."""
        mock_user.onboarding_completed = False
        user_id = str(mock_user.id)

        # Mock this user as the first owner in the org (super admin)
        first_owner = MagicMock(spec=User)
        first_owner.id = mock_user.id

        with (
            patch('server.routes.auth.DEPLOYMENT_MODE', 'self_hosted'),
            patch(
                'server.routes.auth.UserStore.get_first_owner_in_org',
                new_callable=AsyncMock,
                return_value=first_owner,
            ),
        ):
            result = await _should_redirect_to_onboarding(user_id, mock_user)

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_for_self_hosted_non_super_admin_owner(self, mock_user):
        """Test that owners who aren't the super admin are NOT redirected."""
        mock_user.onboarding_completed = False
        user_id = str(mock_user.id)

        # Mock a different user as the first owner (super admin)
        first_owner = MagicMock(spec=User)
        first_owner.id = uuid.uuid4()  # Different user

        with (
            patch('server.routes.auth.DEPLOYMENT_MODE', 'self_hosted'),
            patch(
                'server.routes.auth.UserStore.get_first_owner_in_org',
                new_callable=AsyncMock,
                return_value=first_owner,
            ),
        ):
            result = await _should_redirect_to_onboarding(user_id, mock_user)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_for_self_hosted_when_no_owner_found(self, mock_user):
        """Test that users are not redirected when no owner is found."""
        mock_user.onboarding_completed = False
        user_id = str(mock_user.id)

        with (
            patch('server.routes.auth.DEPLOYMENT_MODE', 'self_hosted'),
            patch(
                'server.routes.auth.UserStore.get_first_owner_in_org',
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await _should_redirect_to_onboarding(user_id, mock_user)

        assert result is False

    @pytest.mark.asyncio
    async def test_passes_current_org_id_to_get_first_owner(self, mock_user):
        """Test that get_first_owner_in_org is called with user's current_org_id."""
        mock_user.onboarding_completed = False
        user_id = str(mock_user.id)
        mock_get_first_owner = AsyncMock(return_value=None)

        with (
            patch('server.routes.auth.DEPLOYMENT_MODE', 'self_hosted'),
            patch(
                'server.routes.auth.UserStore.get_first_owner_in_org',
                mock_get_first_owner,
            ),
        ):
            await _should_redirect_to_onboarding(user_id, mock_user)

        mock_get_first_owner.assert_called_once_with(mock_user.current_org_id)


# --- Tests for _get_post_auth_redirect ---


class TestGetPostAuthRedirect:
    """Tests for the _get_post_auth_redirect function."""

    @pytest.mark.asyncio
    async def test_returns_onboarding_url_when_onboarding_needed(self, mock_user):
        """Test that onboarding URL is returned when user needs onboarding."""
        mock_user.onboarding_completed = False
        user_id = str(mock_user.id)

        with (
            patch('server.routes.auth.DEPLOYMENT_MODE', 'cloud'),
            patch(
                'server.routes.auth.UserStore.get_user_by_id',
                new_callable=AsyncMock,
                return_value=mock_user,
            ),
        ):
            result = await _get_post_auth_redirect(
                user_id, 'https://example.com/', 'https://example.com'
            )

        assert result == 'https://example.com/onboarding'

    @pytest.mark.asyncio
    async def test_returns_default_url_when_onboarding_completed(self, mock_user):
        """Test that default URL is returned when user has completed onboarding."""
        mock_user.onboarding_completed = True
        user_id = str(mock_user.id)

        with patch(
            'server.routes.auth.UserStore.get_user_by_id',
            new_callable=AsyncMock,
            return_value=mock_user,
        ):
            result = await _get_post_auth_redirect(
                user_id, 'https://example.com/dashboard', 'https://example.com'
            )

        assert result == 'https://example.com/dashboard'

    @pytest.mark.asyncio
    async def test_returns_default_url_when_user_not_found(self):
        """Test that default URL is returned when user is not found."""
        with patch(
            'server.routes.auth.UserStore.get_user_by_id',
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await _get_post_auth_redirect(
                'nonexistent-user', 'https://example.com/', 'https://example.com'
            )

        assert result == 'https://example.com/'

    @pytest.mark.asyncio
    async def test_logs_when_redirecting_to_onboarding(self, mock_user):
        """Test that a log message is emitted when redirecting to onboarding."""
        mock_user.onboarding_completed = False
        user_id = str(mock_user.id)

        with (
            patch('server.routes.auth.DEPLOYMENT_MODE', 'cloud'),
            patch(
                'server.routes.auth.UserStore.get_user_by_id',
                new_callable=AsyncMock,
                return_value=mock_user,
            ),
            patch('server.routes.auth.logger') as mock_logger,
        ):
            await _get_post_auth_redirect(
                user_id, 'https://example.com/', 'https://example.com'
            )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert call_args[0][0] == 'Redirecting user to onboarding'
        assert call_args[1]['extra']['user_id'] == user_id


# --- Tests for /complete_onboarding endpoint ---


class TestCompleteOnboardingEndpoint:
    """Tests for the complete_onboarding API endpoint."""

    @pytest.mark.asyncio
    async def test_returns_401_when_not_authenticated(self, mock_request):
        """Test that unauthenticated requests return 401."""
        mock_user_auth = MagicMock(spec=SaasUserAuth)
        mock_user_auth.get_user_id = AsyncMock(return_value=None)

        with patch(
            'server.routes.auth.get_user_auth',
            new_callable=AsyncMock,
            return_value=mock_user_auth,
        ):
            result = await complete_onboarding(mock_request)

        assert isinstance(result, JSONResponse)
        assert result.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_returns_404_when_user_not_found(self, mock_request):
        """Test that request for non-existent user returns 404."""
        user_id = str(uuid.uuid4())
        mock_user_auth = MagicMock(spec=SaasUserAuth)
        mock_user_auth.get_user_id = AsyncMock(return_value=user_id)

        with (
            patch(
                'server.routes.auth.get_user_auth',
                new_callable=AsyncMock,
                return_value=mock_user_auth,
            ),
            patch(
                'server.routes.auth.UserStore.mark_onboarding_completed',
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await complete_onboarding(mock_request)

        assert isinstance(result, JSONResponse)
        assert result.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_returns_200_on_success(self, mock_request, mock_user):
        """Test successful onboarding completion returns 200."""
        user_id = str(uuid.uuid4())
        mock_user_auth = MagicMock(spec=SaasUserAuth)
        mock_user_auth.get_user_id = AsyncMock(return_value=user_id)

        with (
            patch(
                'server.routes.auth.get_user_auth',
                new_callable=AsyncMock,
                return_value=mock_user_auth,
            ),
            patch(
                'server.routes.auth.UserStore.mark_onboarding_completed',
                new_callable=AsyncMock,
                return_value=mock_user,
            ),
        ):
            result = await complete_onboarding(mock_request)

        assert isinstance(result, JSONResponse)
        assert result.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_calls_mark_onboarding_completed_with_user_id(
        self, mock_request, mock_user
    ):
        """Test that mark_onboarding_completed is called with the correct user_id."""
        user_id = str(uuid.uuid4())
        mock_user_auth = MagicMock(spec=SaasUserAuth)
        mock_user_auth.get_user_id = AsyncMock(return_value=user_id)
        mock_mark_completed = AsyncMock(return_value=mock_user)

        with (
            patch(
                'server.routes.auth.get_user_auth',
                new_callable=AsyncMock,
                return_value=mock_user_auth,
            ),
            patch(
                'server.routes.auth.UserStore.mark_onboarding_completed',
                mock_mark_completed,
            ),
        ):
            await complete_onboarding(mock_request)

        mock_mark_completed.assert_called_once_with(user_id)
