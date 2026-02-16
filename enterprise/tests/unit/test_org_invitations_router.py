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
