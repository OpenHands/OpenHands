"""Tests for onboarding-related auth routes and functions.

Tests for:
- _should_redirect_to_onboarding() function
- _get_post_auth_redirect() function
- /complete_onboarding endpoint
- /onboarding_status endpoint
"""

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request, status
from fastapi.responses import JSONResponse
from server.auth.saas_user_auth import SaasUserAuth
from server.routes.auth import (
    OnboardingSubmission,
    _build_cross_app_redirect_url,
    _build_onboarding_redirect,
    _get_post_auth_redirect,
    _should_redirect_to_onboarding,
    complete_onboarding,
    onboarding_status,
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
    async def test_unwraps_automation_login_redirect_when_onboarding_completed(
        self, mock_user
    ):
        """Automation redirects should bypass the main app login SPA."""
        mock_user.onboarding_completed = True
        user_id = str(mock_user.id)

        with patch(
            'server.routes.auth.UserStore.get_user_by_id',
            new_callable=AsyncMock,
            return_value=mock_user,
        ):
            result = await _get_post_auth_redirect(
                user_id,
                'https://example.com/login?redirect=%2Fautomations&login_method=github',
                'https://example.com',
            )

        assert result == 'https://example.com/automations?login_method=github'

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

    @pytest.mark.asyncio
    async def test_preserves_deep_link_path_as_returnTo(self, mock_user):
        """Regression: deep-link path is preserved as ``returnTo``.

        When an onboarding-needing user originally requested a deep
        link (e.g. ``/conversations/abc``), the backend must redirect
        to ``/onboarding?returnTo=...`` so the frontend
        ``OnboardingForm`` can restore that destination after the
        user finishes the form. Without this, deep links clicked
        while logged out are silently dropped at the onboarding
        interstitial.
        """
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
                user_id,
                'https://example.com/conversations/abc',
                'https://example.com',
            )

        assert result == (
            'https://example.com/onboarding?returnTo=%2Fconversations%2Fabc'
        )

    @pytest.mark.asyncio
    async def test_preserves_query_string_as_returnTo(self, mock_user):
        """Regression: query-string destinations are preserved fully.

        Destinations like ``/conversations/abc?foo=bar`` must be
        preserved in the encoded ``returnTo`` rather than truncated.
        """
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
                user_id,
                'https://example.com/conversations/abc?foo=bar',
                'https://example.com',
            )

        assert result == (
            'https://example.com/onboarding?returnTo=%2Fconversations%2Fabc%3Ffoo%3Dbar'
        )


class TestBuildCrossAppRedirectUrl:
    def test_unwraps_legacy_login_redirect_to_automations(self):
        result = _build_cross_app_redirect_url(
            'https://example.com/login?redirect=%2Fautomations&login_method=github',
            'https://example.com',
        )

        assert result == 'https://example.com/automations?login_method=github'

    def test_unwraps_login_return_to_automation_deep_link(self):
        result = _build_cross_app_redirect_url(
            'https://example.com/login'
            '?returnTo=%2Fautomations%2Fabc%3Ftab%3Druns'
            '&login_method=gitlab',
            'https://example.com',
        )

        assert result == (
            'https://example.com/automations/abc?tab=runs&login_method=gitlab'
        )

    def test_direct_automation_path_becomes_same_origin_absolute_url(self):
        result = _build_cross_app_redirect_url('/automations', 'https://example.com')

        assert result == 'https://example.com/automations'

    def test_non_automation_login_redirect_is_unchanged(self):
        result = _build_cross_app_redirect_url(
            'https://example.com/login?redirect=%2Fsettings&login_method=github',
            'https://example.com',
        )

        assert result == (
            'https://example.com/login?redirect=%2Fsettings&login_method=github'
        )

    def test_rejects_protocol_relative_inner_redirect(self):
        result = _build_cross_app_redirect_url(
            'https://example.com/login?redirect=%2F%2Fevil.example.com%2Fautomations',
            'https://example.com',
        )

        assert result == (
            'https://example.com/login?redirect=%2F%2Fevil.example.com%2Fautomations'
        )


# --- Tests for _build_onboarding_redirect ---


class TestBuildOnboardingRedirect:
    """Tests for the ``_build_onboarding_redirect`` helper.

    These exercise the unit directly — the integration coverage
    through ``_get_post_auth_redirect`` lives in
    ``TestGetPostAuthRedirect`` above.
    """

    def test_strips_web_url_prefix_to_relative_path(self):
        result = _build_onboarding_redirect(
            'https://example.com/foo/bar', 'https://example.com'
        )
        assert result == ('https://example.com/onboarding?returnTo=%2Ffoo%2Fbar')

    def test_preserves_query_string_in_returnTo(self):
        result = _build_onboarding_redirect(
            'https://example.com/foo?bar=baz&qux=1', 'https://example.com'
        )
        assert result == (
            'https://example.com/onboarding?returnTo=%2Ffoo%3Fbar%3Dbaz%26qux%3D1'
        )

    def test_skips_returnTo_for_bare_home_with_trailing_slash(self):
        # ``https://example.com/`` is the home page; appending a
        # ``returnTo=%2F`` would be noise since ``/`` is the default
        # post-onboarding landing page anyway.
        result = _build_onboarding_redirect(
            'https://example.com/', 'https://example.com'
        )
        assert result == 'https://example.com/onboarding'

    def test_skips_returnTo_for_bare_home_without_trailing_slash(self):
        result = _build_onboarding_redirect(
            'https://example.com', 'https://example.com'
        )
        assert result == 'https://example.com/onboarding'

    def test_skips_returnTo_when_original_url_is_empty(self):
        result = _build_onboarding_redirect('', 'https://example.com')
        assert result == 'https://example.com/onboarding'

    def test_passes_through_absolute_url_when_origin_does_not_match(self):
        # Defensive: if the original_url doesn't share the deployment
        # origin, the helper preserves it as-is so the frontend can
        # decide whether to navigate within the SPA or do a hard
        # window.location redirect.
        result = _build_onboarding_redirect(
            'https://other.example.com/foo', 'https://example.com'
        )
        assert result == (
            'https://example.com/onboarding'
            '?returnTo=https%3A%2F%2Fother.example.com%2Ffoo'
        )

    def test_unwraps_cross_origin_login_returnTo(self):
        """Cross-origin login URLs still have their inner ``returnTo`` extracted.

        ``_extract_login_inner_return_to`` matches on the path only
        (``parsed.path == '/login'``); it does not check the host.  A
        login URL at a different origin therefore has its inner
        destination unwrapped to a safe relative path rather than being
        preserved as a cross-origin absolute URL.

        This differs from ``test_passes_through_absolute_url_when_origin_does_not_match``
        (non-login cross-origin path → absolute URL preserved as-is) and
        the behaviour is intentionally *safer*: the attacker-controlled
        origin is discarded and only the relative ``returnTo`` value from
        the query string is used.
        """
        result = _build_onboarding_redirect(
            'https://other.example.com/login?returnTo=%2Ffoo', 'https://example.com'
        )
        assert result == 'https://example.com/onboarding?returnTo=%2Ffoo'

    def test_unwraps_legacy_login_redirect_to_inner_destination(self):
        result = _build_onboarding_redirect(
            'https://example.com/login?redirect=%2Fautomations&login_method=github',
            'https://example.com',
        )
        assert result == 'https://example.com/onboarding?returnTo=%2Fautomations'

    def test_unwraps_login_returnTo_to_inner_destination(self):
        """Regression: login-wrapped destinations are unwrapped.

        An unauthenticated deep-link visit creates an OAuth ``state``
        that wraps the user's real destination inside a
        ``/login?returnTo=...`` URL. The onboarding redirect must
        unwrap that so the user lands directly at their destination
        after finishing onboarding instead of bouncing through
        ``/login`` (which adds round-trips and is brittle when
        query-string layering goes wrong).
        """
        result = _build_onboarding_redirect(
            'https://example.com/login?returnTo=%2Fsettings%2Fuser&login_method=github',
            'https://example.com',
        )
        assert result == ('https://example.com/onboarding?returnTo=%2Fsettings%2Fuser')

    def test_unwraps_login_returnTo_with_inner_query_string(self):
        """Inner destinations with their own query string survive unwrap.

        Destinations like ``/conversations/abc?foo=bar`` must keep
        their query string when the outer login URL is unwrapped.
        """
        result = _build_onboarding_redirect(
            'https://example.com/login'
            '?returnTo=%2Fconversations%2Fabc%3Ffoo%3Dbar'
            '&login_method=github',
            'https://example.com',
        )
        assert result == (
            'https://example.com/onboarding?returnTo=%2Fconversations%2Fabc%3Ffoo%3Dbar'
        )

    def test_unwraps_login_returnTo_to_bare_home_skips_returnTo(self):
        """Home-page short-circuit applies after the unwrap step.

        If the unwrapped destination is the bare home page, the
        helper should still emit the clean ``/onboarding`` URL with
        no ``returnTo`` query parameter.
        """
        result = _build_onboarding_redirect(
            'https://example.com/login?returnTo=%2F&login_method=github',
            'https://example.com',
        )
        assert result == 'https://example.com/onboarding'

    def test_does_not_unwrap_non_login_path(self):
        """Non-login paths with a ``returnTo`` are preserved verbatim.

        Only ``/login`` URLs get the inner-returnTo unwrap; other
        paths that happen to carry a ``returnTo`` query parameter
        must be preserved as-is.
        """
        result = _build_onboarding_redirect(
            'https://example.com/foo?returnTo=%2Fbar', 'https://example.com'
        )
        assert result == (
            'https://example.com/onboarding?returnTo=%2Ffoo%3FreturnTo%3D%252Fbar'
        )


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

    @pytest.mark.asyncio
    async def test_fires_onboarding_analytics_with_selections(
        self, mock_request, mock_user
    ):
        """On success, the endpoint fires `track_onboarding_completed`
        with the selections from the request body and a group_identify on
        the user's current org."""
        user_id = str(uuid.uuid4())
        org_id = str(uuid.uuid4())
        mock_user.email = 'user@example.com'
        selections = {
            'role': 'software_engineer',
            'org_size': 'solo',
            'use_case': ['new_features', 'fixing_bugs'],
        }

        mock_user_auth = MagicMock(spec=SaasUserAuth)
        mock_user_auth.get_user_id = AsyncMock(return_value=user_id)

        mock_analytics = MagicMock()
        mock_ctx = MagicMock(org_id=org_id)

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
            patch(
                'server.routes.auth.get_analytics_service',
                return_value=mock_analytics,
            ),
            patch(
                'server.routes.auth.resolve_analytics_context',
                new_callable=AsyncMock,
                return_value=mock_ctx,
            ),
        ):
            result = await complete_onboarding(
                mock_request, body=OnboardingSubmission(selections=selections)
            )

        assert isinstance(result, JSONResponse)
        assert result.status_code == status.HTTP_200_OK
        mock_analytics.set_person_properties.assert_called_once_with(
            ctx=mock_ctx,
            properties={'email': 'user@example.com'},
        )
        mock_analytics.track_onboarding_completed.assert_called_once_with(
            ctx=mock_ctx,
            selections=selections,
        )
        mock_analytics.group_identify.assert_called_once()
        gi_kwargs = mock_analytics.group_identify.call_args.kwargs
        assert gi_kwargs['group_type'] == 'org'
        assert gi_kwargs['group_key'] == org_id
        assert 'onboarding_completed_at' in gi_kwargs['properties']

    @pytest.mark.asyncio
    async def test_skips_set_person_properties_when_no_email(
        self, mock_request, mock_user
    ):
        """set_person_properties must not fire when user.email is None."""
        user_id = str(uuid.uuid4())
        mock_user.email = None
        mock_user_auth = MagicMock(spec=SaasUserAuth)
        mock_user_auth.get_user_id = AsyncMock(return_value=user_id)

        mock_analytics = MagicMock()
        mock_ctx = MagicMock(org_id=None)

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
            patch(
                'server.routes.auth.get_analytics_service',
                return_value=mock_analytics,
            ),
            patch(
                'server.routes.auth.resolve_analytics_context',
                new_callable=AsyncMock,
                return_value=mock_ctx,
            ),
        ):
            result = await complete_onboarding(
                mock_request, body=OnboardingSubmission(selections={})
            )

        assert isinstance(result, JSONResponse)
        assert result.status_code == status.HTTP_200_OK
        mock_analytics.set_person_properties.assert_not_called()
        mock_analytics.track_onboarding_completed.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_group_identify_when_no_org_id(self, mock_request, mock_user):
        """group_identify must not fire when the user has no current org."""
        user_id = str(uuid.uuid4())
        mock_user_auth = MagicMock(spec=SaasUserAuth)
        mock_user_auth.get_user_id = AsyncMock(return_value=user_id)

        mock_analytics = MagicMock()
        mock_ctx = MagicMock(org_id=None)

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
            patch(
                'server.routes.auth.get_analytics_service',
                return_value=mock_analytics,
            ),
            patch(
                'server.routes.auth.resolve_analytics_context',
                new_callable=AsyncMock,
                return_value=mock_ctx,
            ),
        ):
            await complete_onboarding(
                mock_request, body=OnboardingSubmission(selections={})
            )

        mock_analytics.track_onboarding_completed.assert_called_once()
        mock_analytics.group_identify.assert_not_called()

    @pytest.mark.asyncio
    async def test_analytics_exception_does_not_break_endpoint(
        self, mock_request, mock_user
    ):
        """Telemetry failures must never block the user from finishing
        onboarding - the endpoint should still return 200."""
        user_id = str(uuid.uuid4())
        mock_user_auth = MagicMock(spec=SaasUserAuth)
        mock_user_auth.get_user_id = AsyncMock(return_value=user_id)

        mock_analytics = MagicMock()
        mock_analytics.track_onboarding_completed.side_effect = RuntimeError(
            'posthog down'
        )

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
            patch(
                'server.routes.auth.get_analytics_service',
                return_value=mock_analytics,
            ),
            patch(
                'server.routes.auth.resolve_analytics_context',
                new_callable=AsyncMock,
                return_value=MagicMock(org_id=None),
            ),
        ):
            result = await complete_onboarding(
                mock_request, body=OnboardingSubmission(selections={})
            )

        assert isinstance(result, JSONResponse)
        assert result.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_no_body_defaults_to_empty_selections(self, mock_request, mock_user):
        """When called with no body (backwards compat), analytics still
        fires with an empty selections dict."""
        user_id = str(uuid.uuid4())
        mock_user_auth = MagicMock(spec=SaasUserAuth)
        mock_user_auth.get_user_id = AsyncMock(return_value=user_id)

        mock_analytics = MagicMock()

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
            patch(
                'server.routes.auth.get_analytics_service',
                return_value=mock_analytics,
            ),
            patch(
                'server.routes.auth.resolve_analytics_context',
                new_callable=AsyncMock,
                return_value=MagicMock(org_id=None),
            ),
        ):
            await complete_onboarding(mock_request)

        mock_analytics.track_onboarding_completed.assert_called_once()
        kwargs = mock_analytics.track_onboarding_completed.call_args.kwargs
        assert kwargs['selections'] == {}


class TestOnboardingStatusEndpoint:
    """Tests for the /onboarding_status API endpoint."""

    @pytest.mark.asyncio
    async def test_returns_401_when_not_authenticated(self, mock_request):
        """Unauthenticated requests return 401."""
        mock_user_auth = MagicMock(spec=SaasUserAuth)
        mock_user_auth.get_user_id = AsyncMock(return_value=None)

        with patch(
            'server.routes.auth.get_user_auth',
            new_callable=AsyncMock,
            return_value=mock_user_auth,
        ):
            result = await onboarding_status(mock_request)

        assert isinstance(result, JSONResponse)
        assert result.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_returns_true_for_new_cloud_user(self, mock_request, mock_user):
        """A cloud user whose onboarding is incomplete should be told to complete it."""
        user_id = str(uuid.uuid4())
        mock_user.onboarding_completed = False
        mock_user_auth = MagicMock(spec=SaasUserAuth)
        mock_user_auth.get_user_id = AsyncMock(return_value=user_id)

        with (
            patch(
                'server.routes.auth.get_user_auth',
                new_callable=AsyncMock,
                return_value=mock_user_auth,
            ),
            patch(
                'server.routes.auth.UserStore.get_user_by_id',
                new_callable=AsyncMock,
                return_value=mock_user,
            ),
            patch('server.routes.auth.DEPLOYMENT_MODE', 'cloud'),
        ):
            result = await onboarding_status(mock_request)

        assert isinstance(result, JSONResponse)
        assert result.status_code == status.HTTP_200_OK
        body = json.loads(result.body)
        assert body == {'should_complete_onboarding': True}

    @pytest.mark.asyncio
    async def test_returns_false_for_completed_user(self, mock_request, mock_user):
        """A user who already completed onboarding should not be told to complete it."""
        user_id = str(uuid.uuid4())
        mock_user.onboarding_completed = True
        mock_user_auth = MagicMock(spec=SaasUserAuth)
        mock_user_auth.get_user_id = AsyncMock(return_value=user_id)

        with (
            patch(
                'server.routes.auth.get_user_auth',
                new_callable=AsyncMock,
                return_value=mock_user_auth,
            ),
            patch(
                'server.routes.auth.UserStore.get_user_by_id',
                new_callable=AsyncMock,
                return_value=mock_user,
            ),
        ):
            result = await onboarding_status(mock_request)

        assert isinstance(result, JSONResponse)
        assert result.status_code == status.HTTP_200_OK
        body = json.loads(result.body)
        assert body == {'should_complete_onboarding': False}
