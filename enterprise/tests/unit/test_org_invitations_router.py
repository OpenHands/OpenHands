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


class TestAcceptInvitationEndpoint:
    """Test cases for the accept invitation endpoint."""

    @pytest.fixture
    def mock_user_auth(self):
        """Create a mock user auth."""
        user_auth = MagicMock()
        user_auth.get_user_id = AsyncMock(
            return_value='87654321-4321-8765-4321-876543218765'
        )
        return user_auth

    @pytest.mark.asyncio
    async def test_accept_unauthenticated_redirects_to_login(self, client):
        """Test that unauthenticated users are redirected to login with invitation token."""
        with patch(
            'server.routes.org_invitations.get_user_auth',
            new_callable=AsyncMock,
            return_value=None,
        ):
            response = client.get(
                '/api/organizations/members/invite/accept?token=inv-test-token-123',
                follow_redirects=False,
            )

            assert response.status_code == 302
            assert '/login?invitation_token=inv-test-token-123' in response.headers.get(
                'location', ''
            )

    @pytest.mark.asyncio
    async def test_accept_authenticated_success_redirects_home(
        self, client, mock_user_auth
    ):
        """Test that successful acceptance redirects to home page."""
        mock_invitation = MagicMock()

        with (
            patch(
                'server.routes.org_invitations.get_user_auth',
                new_callable=AsyncMock,
                return_value=mock_user_auth,
            ),
            patch(
                'server.routes.org_invitations.OrgInvitationService.accept_invitation',
                new_callable=AsyncMock,
                return_value=mock_invitation,
            ),
        ):
            response = client.get(
                '/api/organizations/members/invite/accept?token=inv-test-token-123',
                follow_redirects=False,
            )

            assert response.status_code == 302
            location = response.headers.get('location', '')
            assert location.endswith('/')
            assert 'invitation_expired' not in location
            assert 'invitation_invalid' not in location
            assert 'email_mismatch' not in location

    @pytest.mark.asyncio
    async def test_accept_expired_invitation_redirects_with_flag(
        self, client, mock_user_auth
    ):
        """Test that expired invitation redirects with invitation_expired=true."""
        with (
            patch(
                'server.routes.org_invitations.get_user_auth',
                new_callable=AsyncMock,
                return_value=mock_user_auth,
            ),
            patch(
                'server.routes.org_invitations.OrgInvitationService.accept_invitation',
                new_callable=AsyncMock,
                side_effect=InvitationExpiredError(),
            ),
        ):
            response = client.get(
                '/api/organizations/members/invite/accept?token=inv-test-token-123',
                follow_redirects=False,
            )

            assert response.status_code == 302
            assert 'invitation_expired=true' in response.headers.get('location', '')

    @pytest.mark.asyncio
    async def test_accept_invalid_invitation_redirects_with_flag(
        self, client, mock_user_auth
    ):
        """Test that invalid invitation redirects with invitation_invalid=true."""
        with (
            patch(
                'server.routes.org_invitations.get_user_auth',
                new_callable=AsyncMock,
                return_value=mock_user_auth,
            ),
            patch(
                'server.routes.org_invitations.OrgInvitationService.accept_invitation',
                new_callable=AsyncMock,
                side_effect=InvitationInvalidError(),
            ),
        ):
            response = client.get(
                '/api/organizations/members/invite/accept?token=inv-test-token-123',
                follow_redirects=False,
            )

            assert response.status_code == 302
            assert 'invitation_invalid=true' in response.headers.get('location', '')

    @pytest.mark.asyncio
    async def test_accept_already_member_redirects_with_flag(
        self, client, mock_user_auth
    ):
        """Test that already member error redirects with already_member=true."""
        with (
            patch(
                'server.routes.org_invitations.get_user_auth',
                new_callable=AsyncMock,
                return_value=mock_user_auth,
            ),
            patch(
                'server.routes.org_invitations.OrgInvitationService.accept_invitation',
                new_callable=AsyncMock,
                side_effect=UserAlreadyMemberError(),
            ),
        ):
            response = client.get(
                '/api/organizations/members/invite/accept?token=inv-test-token-123',
                follow_redirects=False,
            )

            assert response.status_code == 302
            assert 'already_member=true' in response.headers.get('location', '')

    @pytest.mark.asyncio
    async def test_accept_email_mismatch_redirects_with_flag(
        self, client, mock_user_auth
    ):
        """Test that email mismatch error redirects with email_mismatch=true."""
        with (
            patch(
                'server.routes.org_invitations.get_user_auth',
                new_callable=AsyncMock,
                return_value=mock_user_auth,
            ),
            patch(
                'server.routes.org_invitations.OrgInvitationService.accept_invitation',
                new_callable=AsyncMock,
                side_effect=EmailMismatchError(),
            ),
        ):
            response = client.get(
                '/api/organizations/members/invite/accept?token=inv-test-token-123',
                follow_redirects=False,
            )

            assert response.status_code == 302
            assert 'email_mismatch=true' in response.headers.get('location', '')

    @pytest.mark.asyncio
    async def test_accept_unexpected_error_redirects_with_flag(
        self, client, mock_user_auth
    ):
        """Test that unexpected errors redirect with invitation_error=true."""
        with (
            patch(
                'server.routes.org_invitations.get_user_auth',
                new_callable=AsyncMock,
                return_value=mock_user_auth,
            ),
            patch(
                'server.routes.org_invitations.OrgInvitationService.accept_invitation',
                new_callable=AsyncMock,
                side_effect=Exception('Unexpected error'),
            ),
        ):
            response = client.get(
                '/api/organizations/members/invite/accept?token=inv-test-token-123',
                follow_redirects=False,
            )

            assert response.status_code == 302
            assert 'invitation_error=true' in response.headers.get('location', '')


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
