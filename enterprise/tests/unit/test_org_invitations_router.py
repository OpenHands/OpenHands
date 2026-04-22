"""Tests for organization invitations API router."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from server.routes.org_invitation_models import (
    EmailMismatchError,
    InvitationExpiredError,
    InvitationInvalidError,
    UserAlreadyMemberError,
)
from server.routes.org_invitations import accept_router, invitation_router


@pytest.fixture
def app():
    """Create a FastAPI app with the invitation routers."""
    app = FastAPI()
    app.include_router(invitation_router)
    app.include_router(accept_router)
    return app


@pytest.fixture
def client(app):
    """Create a test client for the app."""
    return TestClient(app)


class TestRouterPrefixes:
    """Test that router prefixes are configured correctly."""

    def test_invitation_router_has_correct_prefix(self):
        """Test that invitation_router has /api/organizations/{org_id}/members prefix."""
        assert invitation_router.prefix == '/api/organizations/{org_id}/members'

    def test_accept_router_has_correct_prefix(self):
        """Test that accept_router has /api/organizations/members/invite prefix."""
        assert accept_router.prefix == '/api/organizations/members/invite'


class TestAcceptInvitationGetEndpoint:
    """Test cases for the GET accept invitation endpoint (redirect flow)."""

    def test_get_accept_redirects_to_home_with_token(self, client):
        """Test that GET request always redirects to home with invitation_token.

        The GET endpoint is accessed via the link in invitation emails.
        It always redirects to the home page with the token, allowing the
        frontend to handle acceptance via a modal with authenticated POST.
        """
        response = client.get(
            '/api/organizations/members/invite/accept?token=inv-test-token-123',
            follow_redirects=False,
        )

        assert response.status_code == 302
        location = response.headers.get('location', '')
        assert '/?invitation_token=inv-test-token-123' in location


class TestAcceptInvitationPostEndpoint:
    """Test cases for the POST accept invitation endpoint (authenticated flow)."""

    @pytest.fixture
    def auth_app(self):
        """Create a FastAPI app with dependency overrides for authenticated tests."""

        from openhands.server.user_auth import get_user_id

        app = FastAPI()
        app.include_router(accept_router)

        # Override the get_user_id dependency
        app.dependency_overrides[get_user_id] = (
            lambda: '87654321-4321-8765-4321-876543218765'
        )

        return app

    @pytest.fixture
    def auth_client(self, auth_app):
        """Create a test client with authentication dependency overrides."""
        return TestClient(auth_app)

    @pytest.mark.asyncio
    async def test_post_accept_success_returns_org_details(self, auth_client):
        """Test that successful POST acceptance returns organization details."""
        from uuid import UUID

        mock_invitation = MagicMock()
        mock_invitation.org_id = UUID('12345678-1234-5678-1234-567812345678')
        mock_invitation.role_id = 3

        mock_org = MagicMock()
        mock_org.name = 'Test Organization'

        mock_role = MagicMock()
        mock_role.name = 'member'

        with (
            patch(
                'server.routes.org_invitations.OrgInvitationService.accept_invitation',
                new_callable=AsyncMock,
                return_value=mock_invitation,
            ),
            patch(
                'server.routes.org_invitations.OrgStore.get_org_by_id',
                new_callable=AsyncMock,
                return_value=mock_org,
            ),
            patch(
                'server.routes.org_invitations.RoleStore.get_role_by_id',
                new_callable=AsyncMock,
                return_value=mock_role,
            ),
        ):
            response = auth_client.post(
                '/api/organizations/members/invite/accept',
                json={'token': 'inv-test-token-123'},
            )

            assert response.status_code == 200
            data = response.json()
            assert data['success'] is True
            assert data['org_id'] == '12345678-1234-5678-1234-567812345678'
            assert data['org_name'] == 'Test Organization'
            assert data['role'] == 'member'

    @pytest.mark.asyncio
    async def test_post_accept_expired_returns_400(self, auth_client):
        """Test that expired invitation returns 400 with detail."""
        with patch(
            'server.routes.org_invitations.OrgInvitationService.accept_invitation',
            new_callable=AsyncMock,
            side_effect=InvitationExpiredError(),
        ):
            response = auth_client.post(
                '/api/organizations/members/invite/accept',
                json={'token': 'inv-test-token-123'},
            )

            assert response.status_code == 400
            assert response.json()['detail'] == 'invitation_expired'

    @pytest.mark.asyncio
    async def test_post_accept_invalid_returns_400(self, auth_client):
        """Test that invalid invitation returns 400 with detail."""
        with patch(
            'server.routes.org_invitations.OrgInvitationService.accept_invitation',
            new_callable=AsyncMock,
            side_effect=InvitationInvalidError(),
        ):
            response = auth_client.post(
                '/api/organizations/members/invite/accept',
                json={'token': 'inv-test-token-123'},
            )

            assert response.status_code == 400
            assert response.json()['detail'] == 'invitation_invalid'

    @pytest.mark.asyncio
    async def test_post_accept_already_member_returns_409(self, auth_client):
        """Test that already member error returns 409 with detail."""
        with patch(
            'server.routes.org_invitations.OrgInvitationService.accept_invitation',
            new_callable=AsyncMock,
            side_effect=UserAlreadyMemberError(),
        ):
            response = auth_client.post(
                '/api/organizations/members/invite/accept',
                json={'token': 'inv-test-token-123'},
            )

            assert response.status_code == 409
            assert response.json()['detail'] == 'already_member'

    @pytest.mark.asyncio
    async def test_post_accept_email_mismatch_returns_403(self, auth_client):
        """Test that email mismatch error returns 403 with detail."""
        with patch(
            'server.routes.org_invitations.OrgInvitationService.accept_invitation',
            new_callable=AsyncMock,
            side_effect=EmailMismatchError(),
        ):
            response = auth_client.post(
                '/api/organizations/members/invite/accept',
                json={'token': 'inv-test-token-123'},
            )

            assert response.status_code == 403
            assert response.json()['detail'] == 'email_mismatch'


class TestCreateInvitationBatchEndpoint:
    """Test cases for the batch invitation creation endpoint."""

    @pytest.fixture
    def batch_app(self):
        """Create a FastAPI app with dependency overrides for batch tests."""
        from openhands.server.user_auth import get_user_id

        app = FastAPI()
        app.include_router(invitation_router)

        # Override the get_user_id dependency
        app.dependency_overrides[get_user_id] = (
            lambda: '87654321-4321-8765-4321-876543218765'
        )

        return app

    @pytest.fixture
    def batch_client(self, batch_app):
        """Create a test client with dependency overrides."""
        return TestClient(batch_app)

    @pytest.fixture
    def mock_invitation(self):
        """Create a mock invitation."""
        from datetime import datetime

        invitation = MagicMock()
        invitation.id = 1
        invitation.email = 'alice@example.com'
        invitation.role = MagicMock(name='member')
        invitation.role.name = 'member'
        invitation.role_id = 3
        invitation.status = 'pending'
        invitation.created_at = datetime(2026, 2, 17, 10, 0, 0)
        invitation.expires_at = datetime(2026, 2, 24, 10, 0, 0)
        return invitation

    @pytest.mark.asyncio
    async def test_batch_create_returns_successful_invitations(
        self, batch_client, mock_invitation
    ):
        """Test that batch creation returns successful invitations."""
        mock_invitation_2 = MagicMock()
        mock_invitation_2.id = 2
        mock_invitation_2.email = 'bob@example.com'
        mock_invitation_2.role = MagicMock()
        mock_invitation_2.role.name = 'member'
        mock_invitation_2.role_id = 3
        mock_invitation_2.status = 'pending'
        mock_invitation_2.created_at = mock_invitation.created_at
        mock_invitation_2.expires_at = mock_invitation.expires_at

        with (
            patch(
                'server.routes.org_invitations.check_rate_limit_by_user_id',
                new_callable=AsyncMock,
            ),
            patch(
                'server.routes.org_invitations.OrgInvitationService.create_invitations_batch',
                new_callable=AsyncMock,
                return_value=([mock_invitation, mock_invitation_2], []),
            ),
        ):
            response = batch_client.post(
                '/api/organizations/12345678-1234-5678-1234-567812345678/members/invite',
                json={
                    'emails': ['alice@example.com', 'bob@example.com'],
                    'role': 'member',
                },
            )

            assert response.status_code == 201
            data = response.json()
            assert len(data['successful']) == 2
            assert len(data['failed']) == 0

    @pytest.mark.asyncio
    async def test_batch_create_returns_partial_success(
        self, batch_client, mock_invitation
    ):
        """Test that batch creation returns both successful and failed invitations."""
        failed_emails = [('existing@example.com', 'User is already a member')]

        with (
            patch(
                'server.routes.org_invitations.check_rate_limit_by_user_id',
                new_callable=AsyncMock,
            ),
            patch(
                'server.routes.org_invitations.OrgInvitationService.create_invitations_batch',
                new_callable=AsyncMock,
                return_value=([mock_invitation], failed_emails),
            ),
        ):
            response = batch_client.post(
                '/api/organizations/12345678-1234-5678-1234-567812345678/members/invite',
                json={
                    'emails': ['alice@example.com', 'existing@example.com'],
                    'role': 'member',
                },
            )

            assert response.status_code == 201
            data = response.json()
            assert len(data['successful']) == 1
            assert len(data['failed']) == 1
            assert data['failed'][0]['email'] == 'existing@example.com'
            assert 'already a member' in data['failed'][0]['error']

    @pytest.mark.asyncio
    async def test_batch_create_permission_denied_returns_403(self, batch_client):
        """Test that permission denied returns 403 for entire batch."""
        from server.routes.org_invitation_models import InsufficientPermissionError

        with (
            patch(
                'server.routes.org_invitations.check_rate_limit_by_user_id',
                new_callable=AsyncMock,
            ),
            patch(
                'server.routes.org_invitations.OrgInvitationService.create_invitations_batch',
                new_callable=AsyncMock,
                side_effect=InsufficientPermissionError(
                    'Only owners and admins can invite'
                ),
            ),
        ):
            response = batch_client.post(
                '/api/organizations/12345678-1234-5678-1234-567812345678/members/invite',
                json={'emails': ['alice@example.com'], 'role': 'member'},
            )

            assert response.status_code == 403
            assert 'owners and admins' in response.json()['detail']

    @pytest.mark.asyncio
    async def test_batch_create_invalid_role_returns_400(self, batch_client):
        """Test that invalid role returns 400."""
        with (
            patch(
                'server.routes.org_invitations.check_rate_limit_by_user_id',
                new_callable=AsyncMock,
            ),
            patch(
                'server.routes.org_invitations.OrgInvitationService.create_invitations_batch',
                new_callable=AsyncMock,
                side_effect=ValueError('Invalid role: superuser'),
            ),
        ):
            response = batch_client.post(
                '/api/organizations/12345678-1234-5678-1234-567812345678/members/invite',
                json={'emails': ['alice@example.com'], 'role': 'superuser'},
            )

            assert response.status_code == 400
            assert 'Invalid role' in response.json()['detail']


# ---------------------------------------------------------------------------
# Tests for analytics tracking in create_invitation
# ---------------------------------------------------------------------------


class TestCreateInvitationAnalytics:
    """Test cases for analytics tracking in the create_invitation endpoint."""

    @pytest.fixture
    def batch_analytics_client(self, app):
        """Create a client with mocked auth dependencies for testing analytics."""

        # Override auth dependencies
        async def mock_get_user_id():
            return 'test-user-123'

        async def mock_verify_org_admin():
            return True

        from server.routes.org_invitations import (
            get_user_id,
            verify_current_user_is_org_admin,
        )

        app.dependency_overrides[get_user_id] = mock_get_user_id
        app.dependency_overrides[verify_current_user_is_org_admin] = (
            mock_verify_org_admin
        )

        return TestClient(app)

    def test_create_invitation_tracks_team_members_invited_analytics(
        self, batch_analytics_client
    ):
        """Test that create_invitation calls track_team_members_invited on analytics service."""
        mock_invitation = MagicMock()
        mock_invitation.id = 'inv-123'
        mock_invitation.org_id = '12345678-1234-5678-1234-567812345678'
        mock_invitation.email = 'alice@example.com'
        mock_invitation.role_name = 'member'
        mock_invitation.expires_at = None
        mock_invitation.status = 'pending'

        mock_analytics = MagicMock()
        mock_analytics.track_team_members_invited = MagicMock()

        mock_user = MagicMock()
        mock_user.user_consents_to_analytics = True

        with (
            patch(
                'server.routes.org_invitations.check_rate_limit_by_user_id',
                new_callable=AsyncMock,
            ),
            patch(
                'server.routes.org_invitations.OrgInvitationService.create_invitations_batch',
                new_callable=AsyncMock,
                return_value=([mock_invitation], []),
            ),
            patch(
                'server.routes.org_invitations.get_analytics_service',
                return_value=mock_analytics,
            ),
            patch(
                'storage.user_store.UserStore.get_user_by_id',
                new_callable=AsyncMock,
                return_value=mock_user,
            ),
        ):
            response = batch_analytics_client.post(
                '/api/organizations/12345678-1234-5678-1234-567812345678/members/invite',
                json={'emails': ['alice@example.com'], 'role': 'member'},
            )

            assert response.status_code == 200

        mock_analytics.track_team_members_invited.assert_called_once()
        call_kwargs = mock_analytics.track_team_members_invited.call_args.kwargs
        assert call_kwargs['distinct_id'] == 'test-user-123'
        assert call_kwargs['org_id'] == '12345678-1234-5678-1234-567812345678'
        assert call_kwargs['invited_count'] == 1
        assert call_kwargs['successful_count'] == 1
        assert call_kwargs['failed_count'] == 0
        assert call_kwargs['role'] == 'member'
        assert call_kwargs['consented'] is True

    def test_create_invitation_analytics_tracks_mixed_results(
        self, batch_analytics_client
    ):
        """Test that create_invitation tracks correct counts for mixed success/failure."""
        mock_invitation = MagicMock()
        mock_invitation.id = 'inv-123'
        mock_invitation.org_id = '12345678-1234-5678-1234-567812345678'
        mock_invitation.email = 'alice@example.com'
        mock_invitation.role_name = 'member'
        mock_invitation.expires_at = None
        mock_invitation.status = 'pending'

        mock_failed = MagicMock()
        mock_failed.email = 'bob@example.com'
        mock_failed.reason = 'User already member'

        mock_analytics = MagicMock()
        mock_analytics.track_team_members_invited = MagicMock()

        mock_user = MagicMock()
        mock_user.user_consents_to_analytics = True

        with (
            patch(
                'server.routes.org_invitations.check_rate_limit_by_user_id',
                new_callable=AsyncMock,
            ),
            patch(
                'server.routes.org_invitations.OrgInvitationService.create_invitations_batch',
                new_callable=AsyncMock,
                return_value=([mock_invitation], [mock_failed]),
            ),
            patch(
                'server.routes.org_invitations.get_analytics_service',
                return_value=mock_analytics,
            ),
            patch(
                'storage.user_store.UserStore.get_user_by_id',
                new_callable=AsyncMock,
                return_value=mock_user,
            ),
        ):
            response = batch_analytics_client.post(
                '/api/organizations/12345678-1234-5678-1234-567812345678/members/invite',
                json={
                    'emails': ['alice@example.com', 'bob@example.com'],
                    'role': 'member',
                },
            )

            assert response.status_code == 200

        call_kwargs = mock_analytics.track_team_members_invited.call_args.kwargs
        assert call_kwargs['invited_count'] == 2
        assert call_kwargs['successful_count'] == 1
        assert call_kwargs['failed_count'] == 1

    def test_create_invitation_skips_analytics_when_service_is_none(
        self, batch_analytics_client
    ):
        """Test that create_invitation doesn't fail when analytics service is None."""
        mock_invitation = MagicMock()
        mock_invitation.id = 'inv-123'
        mock_invitation.org_id = '12345678-1234-5678-1234-567812345678'
        mock_invitation.email = 'alice@example.com'
        mock_invitation.role_name = 'member'
        mock_invitation.expires_at = None
        mock_invitation.status = 'pending'

        with (
            patch(
                'server.routes.org_invitations.check_rate_limit_by_user_id',
                new_callable=AsyncMock,
            ),
            patch(
                'server.routes.org_invitations.OrgInvitationService.create_invitations_batch',
                new_callable=AsyncMock,
                return_value=([mock_invitation], []),
            ),
            patch(
                'server.routes.org_invitations.get_analytics_service',
                return_value=None,
            ),
        ):
            # Should not raise even without analytics service
            response = batch_analytics_client.post(
                '/api/organizations/12345678-1234-5678-1234-567812345678/members/invite',
                json={'emails': ['alice@example.com'], 'role': 'member'},
            )

            assert response.status_code == 200

    def test_create_invitation_analytics_respects_consent_false(
        self, batch_analytics_client
    ):
        """Test that create_invitation passes consented=False when user has not consented."""
        mock_invitation = MagicMock()
        mock_invitation.id = 'inv-123'
        mock_invitation.org_id = '12345678-1234-5678-1234-567812345678'
        mock_invitation.email = 'alice@example.com'
        mock_invitation.role_name = 'member'
        mock_invitation.expires_at = None
        mock_invitation.status = 'pending'

        mock_analytics = MagicMock()
        mock_analytics.track_team_members_invited = MagicMock()

        mock_user = MagicMock()
        mock_user.user_consents_to_analytics = False  # User has NOT consented

        with (
            patch(
                'server.routes.org_invitations.check_rate_limit_by_user_id',
                new_callable=AsyncMock,
            ),
            patch(
                'server.routes.org_invitations.OrgInvitationService.create_invitations_batch',
                new_callable=AsyncMock,
                return_value=([mock_invitation], []),
            ),
            patch(
                'server.routes.org_invitations.get_analytics_service',
                return_value=mock_analytics,
            ),
            patch(
                'storage.user_store.UserStore.get_user_by_id',
                new_callable=AsyncMock,
                return_value=mock_user,
            ),
        ):
            response = batch_analytics_client.post(
                '/api/organizations/12345678-1234-5678-1234-567812345678/members/invite',
                json={'emails': ['alice@example.com'], 'role': 'member'},
            )

            assert response.status_code == 200

        call_kwargs = mock_analytics.track_team_members_invited.call_args.kwargs
        assert call_kwargs['consented'] is False

    def test_create_invitation_analytics_exception_does_not_fail_invitation(
        self, batch_analytics_client
    ):
        """Test that analytics exception doesn't prevent successful invitation creation."""
        mock_invitation = MagicMock()
        mock_invitation.id = 'inv-123'
        mock_invitation.org_id = '12345678-1234-5678-1234-567812345678'
        mock_invitation.email = 'alice@example.com'
        mock_invitation.role_name = 'member'
        mock_invitation.expires_at = None
        mock_invitation.status = 'pending'

        mock_analytics = MagicMock()
        mock_analytics.track_team_members_invited.side_effect = RuntimeError(
            'PostHog error'
        )

        mock_user = MagicMock()
        mock_user.user_consents_to_analytics = True

        with (
            patch(
                'server.routes.org_invitations.check_rate_limit_by_user_id',
                new_callable=AsyncMock,
            ),
            patch(
                'server.routes.org_invitations.OrgInvitationService.create_invitations_batch',
                new_callable=AsyncMock,
                return_value=([mock_invitation], []),
            ),
            patch(
                'server.routes.org_invitations.get_analytics_service',
                return_value=mock_analytics,
            ),
            patch(
                'storage.user_store.UserStore.get_user_by_id',
                new_callable=AsyncMock,
                return_value=mock_user,
            ),
        ):
            # Should not raise even when analytics fails
            response = batch_analytics_client.post(
                '/api/organizations/12345678-1234-5678-1234-567812345678/members/invite',
                json={'emails': ['alice@example.com'], 'role': 'member'},
            )

            # The invitation should still be created successfully
            assert response.status_code == 200
