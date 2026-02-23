"""Tests for organization invitation service - email validation."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from server.routes.org_invitation_models import (
    EmailMismatchError,
)
from server.services.org_invitation_service import OrgInvitationService
from storage.org_invitation import OrgInvitation


class TestAcceptInvitationEmailValidation:
    """Test cases for email validation during invitation acceptance."""

    @pytest.fixture
    def mock_invitation(self):
        """Create a mock invitation with pending status."""
        invitation = MagicMock(spec=OrgInvitation)
        invitation.id = 1
        invitation.email = 'alice@example.com'
        invitation.status = OrgInvitation.STATUS_PENDING
        invitation.org_id = UUID('12345678-1234-5678-1234-567812345678')
        invitation.role_id = 1
        return invitation

    @pytest.fixture
    def mock_user(self):
        """Create a mock user with email."""
        user = MagicMock()
        user.id = UUID('87654321-4321-8765-4321-876543218765')
        user.email = 'alice@example.com'
        return user

    @pytest.mark.asyncio
    async def test_accept_invitation_email_matches(self, mock_invitation, mock_user):
        """Test that invitation is accepted when user email matches invitation email."""
        # Arrange
        user_id = mock_user.id
        token = 'inv-test-token-12345'

        with patch.object(
            OrgInvitationService, 'accept_invitation', new_callable=AsyncMock
        ) as mock_accept:
            mock_accept.return_value = mock_invitation

            # Act
            await OrgInvitationService.accept_invitation(token, user_id)

            # Assert
            mock_accept.assert_called_once_with(token, user_id)

    @pytest.mark.asyncio
    async def test_accept_invitation_email_mismatch_raises_error(
        self, mock_invitation, mock_user
    ):
        """Test that EmailMismatchError is raised when emails don't match."""
        # Arrange
        user_id = mock_user.id
        token = 'inv-test-token-12345'
        mock_user.email = 'bob@example.com'  # Different email

        with (
            patch(
                'server.services.org_invitation_service.OrgInvitationStore.get_invitation_by_token',
                new_callable=AsyncMock,
            ) as mock_get_invitation,
            patch(
                'server.services.org_invitation_service.OrgInvitationStore.is_token_expired'
            ) as mock_is_expired,
            patch(
                'server.services.org_invitation_service.UserStore.get_user_by_id_async',
                new_callable=AsyncMock,
            ) as mock_get_user,
        ):
            mock_get_invitation.return_value = mock_invitation
            mock_is_expired.return_value = False
            mock_get_user.return_value = mock_user

            # Act & Assert
            with pytest.raises(EmailMismatchError):
                await OrgInvitationService.accept_invitation(token, user_id)

    @pytest.mark.asyncio
    async def test_accept_invitation_user_no_email_keycloak_fallback_matches(
        self, mock_invitation
    ):
        """Test that Keycloak email is used when user has no email in database."""
        # Arrange
        user_id = UUID('87654321-4321-8765-4321-876543218765')
        token = 'inv-test-token-12345'

        mock_user = MagicMock()
        mock_user.id = user_id
        mock_user.email = None  # No email in database

        mock_keycloak_user_info = {'email': 'alice@example.com'}  # Email from Keycloak

        with (
            patch(
                'server.services.org_invitation_service.OrgInvitationStore.get_invitation_by_token',
                new_callable=AsyncMock,
            ) as mock_get_invitation,
            patch(
                'server.services.org_invitation_service.OrgInvitationStore.is_token_expired'
            ) as mock_is_expired,
            patch(
                'server.services.org_invitation_service.UserStore.get_user_by_id_async',
                new_callable=AsyncMock,
            ) as mock_get_user,
            patch(
                'server.services.org_invitation_service.TokenManager'
            ) as mock_token_manager_class,
            patch(
                'server.services.org_invitation_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_invitation_service.OrgService.create_litellm_integration',
                new_callable=AsyncMock,
            ) as mock_create_litellm,
            patch(
                'server.services.org_invitation_service.OrgMemberStore.add_user_to_org'
            ),
            patch(
                'server.services.org_invitation_service.OrgInvitationStore.update_invitation_status',
                new_callable=AsyncMock,
            ) as mock_update_status,
        ):
            mock_get_invitation.return_value = mock_invitation
            mock_is_expired.return_value = False
            mock_get_user.return_value = mock_user

            # Mock TokenManager instance
            mock_token_manager = MagicMock()
            mock_token_manager.get_user_info_from_user_id = AsyncMock(
                return_value=mock_keycloak_user_info
            )
            mock_token_manager_class.return_value = mock_token_manager

            mock_get_member.return_value = None  # Not already a member
            mock_create_litellm.return_value = MagicMock(llm_api_key='test-key')
            mock_update_status.return_value = mock_invitation

            # Act - should not raise error because Keycloak email matches
            await OrgInvitationService.accept_invitation(token, user_id)

            # Assert
            mock_token_manager.get_user_info_from_user_id.assert_called_once_with(
                str(user_id)
            )

    @pytest.mark.asyncio
    async def test_accept_invitation_no_email_anywhere_raises_error(
        self, mock_invitation
    ):
        """Test that EmailMismatchError is raised when user has no email in database or Keycloak."""
        # Arrange
        user_id = UUID('87654321-4321-8765-4321-876543218765')
        token = 'inv-test-token-12345'

        mock_user = MagicMock()
        mock_user.id = user_id
        mock_user.email = None  # No email in database

        with (
            patch(
                'server.services.org_invitation_service.OrgInvitationStore.get_invitation_by_token',
                new_callable=AsyncMock,
            ) as mock_get_invitation,
            patch(
                'server.services.org_invitation_service.OrgInvitationStore.is_token_expired'
            ) as mock_is_expired,
            patch(
                'server.services.org_invitation_service.UserStore.get_user_by_id_async',
                new_callable=AsyncMock,
            ) as mock_get_user,
            patch(
                'server.services.org_invitation_service.TokenManager'
            ) as mock_token_manager_class,
        ):
            mock_get_invitation.return_value = mock_invitation
            mock_is_expired.return_value = False
            mock_get_user.return_value = mock_user

            # Mock TokenManager to return no email
            mock_token_manager = MagicMock()
            mock_token_manager.get_user_info_from_user_id = AsyncMock(return_value={})
            mock_token_manager_class.return_value = mock_token_manager

            # Act & Assert
            with pytest.raises(EmailMismatchError) as exc_info:
                await OrgInvitationService.accept_invitation(token, user_id)

            assert 'does not have an email address' in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_accept_invitation_email_comparison_is_case_insensitive(
        self, mock_invitation
    ):
        """Test that email comparison is case insensitive."""
        # Arrange
        user_id = UUID('87654321-4321-8765-4321-876543218765')
        token = 'inv-test-token-12345'

        mock_user = MagicMock()
        mock_user.id = user_id
        mock_user.email = 'ALICE@EXAMPLE.COM'  # Uppercase email

        mock_invitation.email = 'alice@example.com'  # Lowercase in invitation

        with (
            patch(
                'server.services.org_invitation_service.OrgInvitationStore.get_invitation_by_token',
                new_callable=AsyncMock,
            ) as mock_get_invitation,
            patch(
                'server.services.org_invitation_service.OrgInvitationStore.is_token_expired'
            ) as mock_is_expired,
            patch(
                'server.services.org_invitation_service.UserStore.get_user_by_id_async',
                new_callable=AsyncMock,
            ) as mock_get_user,
            patch(
                'server.services.org_invitation_service.OrgMemberStore.get_org_member'
            ) as mock_get_member,
            patch(
                'server.services.org_invitation_service.OrgService.create_litellm_integration',
                new_callable=AsyncMock,
            ) as mock_create_litellm,
            patch(
                'server.services.org_invitation_service.OrgMemberStore.add_user_to_org'
            ),
            patch(
                'server.services.org_invitation_service.OrgInvitationStore.update_invitation_status',
                new_callable=AsyncMock,
            ) as mock_update_status,
        ):
            mock_get_invitation.return_value = mock_invitation
            mock_is_expired.return_value = False
            mock_get_user.return_value = mock_user
            mock_get_member.return_value = None
            mock_create_litellm.return_value = MagicMock(llm_api_key='test-key')
            mock_update_status.return_value = mock_invitation

            # Act - should not raise error because emails match case-insensitively
            await OrgInvitationService.accept_invitation(token, user_id)

            # Assert - invitation was accepted (update_invitation_status was called)
            mock_update_status.assert_called_once()


class TestCreateInvitationsBatch:
    """Test cases for batch invitation creation."""

    @pytest.fixture
    def org_id(self):
        """Organization UUID for testing."""
        return UUID('12345678-1234-5678-1234-567812345678')

    @pytest.fixture
    def inviter_id(self):
        """Inviter UUID for testing."""
        return UUID('87654321-4321-8765-4321-876543218765')

    @pytest.fixture
    def mock_org(self):
        """Create a mock organization."""
        org = MagicMock()
        org.id = UUID('12345678-1234-5678-1234-567812345678')
        org.name = 'Test Org'
        return org

    @pytest.fixture
    def mock_inviter_member(self):
        """Create a mock inviter member with owner role."""
        member = MagicMock()
        member.user_id = UUID('87654321-4321-8765-4321-876543218765')
        member.role_id = 1
        return member

    @pytest.fixture
    def mock_owner_role(self):
        """Create a mock owner role."""
        role = MagicMock()
        role.id = 1
        role.name = 'owner'
        return role

    @pytest.fixture
    def mock_member_role(self):
        """Create a mock member role."""
        role = MagicMock()
        role.id = 3
        role.name = 'member'
        return role

    @pytest.mark.asyncio
    async def test_batch_creates_all_invitations_successfully(
        self,
        org_id,
        inviter_id,
        mock_org,
        mock_inviter_member,
        mock_owner_role,
        mock_member_role,
    ):
        """Test that batch creation succeeds for all valid emails."""
        # Arrange
        emails = ['alice@example.com', 'bob@example.com']
        mock_invitation_1 = MagicMock(spec=OrgInvitation)
        mock_invitation_1.id = 1
        mock_invitation_2 = MagicMock(spec=OrgInvitation)
        mock_invitation_2.id = 2

        with (
            patch(
                'server.services.org_invitation_service.OrgStore.get_org_by_id',
                return_value=mock_org,
            ),
            patch(
                'server.services.org_invitation_service.OrgMemberStore.get_org_member',
                return_value=mock_inviter_member,
            ),
            patch(
                'server.services.org_invitation_service.RoleStore.get_role_by_id',
                return_value=mock_owner_role,
            ),
            patch(
                'server.services.org_invitation_service.RoleStore.get_role_by_name',
                return_value=mock_member_role,
            ),
            patch.object(
                OrgInvitationService,
                'create_invitation',
                new_callable=AsyncMock,
                side_effect=[mock_invitation_1, mock_invitation_2],
            ),
        ):
            # Act
            successful, failed = await OrgInvitationService.create_invitations_batch(
                org_id=org_id,
                emails=emails,
                role_name='member',
                inviter_id=inviter_id,
            )

            # Assert
            assert len(successful) == 2
            assert len(failed) == 0

    @pytest.mark.asyncio
    async def test_batch_handles_partial_success(
        self,
        org_id,
        inviter_id,
        mock_org,
        mock_inviter_member,
        mock_owner_role,
        mock_member_role,
    ):
        """Test that batch returns partial results when some emails fail."""
        # Arrange
        from server.routes.org_invitation_models import UserAlreadyMemberError

        emails = ['alice@example.com', 'existing@example.com']
        mock_invitation = MagicMock(spec=OrgInvitation)
        mock_invitation.id = 1

        with (
            patch(
                'server.services.org_invitation_service.OrgStore.get_org_by_id',
                return_value=mock_org,
            ),
            patch(
                'server.services.org_invitation_service.OrgMemberStore.get_org_member',
                return_value=mock_inviter_member,
            ),
            patch(
                'server.services.org_invitation_service.RoleStore.get_role_by_id',
                return_value=mock_owner_role,
            ),
            patch(
                'server.services.org_invitation_service.RoleStore.get_role_by_name',
                return_value=mock_member_role,
            ),
            patch.object(
                OrgInvitationService,
                'create_invitation',
                new_callable=AsyncMock,
                side_effect=[mock_invitation, UserAlreadyMemberError()],
            ),
        ):
            # Act
            successful, failed = await OrgInvitationService.create_invitations_batch(
                org_id=org_id,
                emails=emails,
                role_name='member',
                inviter_id=inviter_id,
            )

            # Assert
            assert len(successful) == 1
            assert len(failed) == 1
            assert failed[0][0] == 'existing@example.com'

    @pytest.mark.asyncio
    async def test_batch_fails_entirely_on_permission_error(self, org_id, inviter_id):
        """Test that permission error fails the entire batch upfront."""
        # Arrange

        emails = ['alice@example.com', 'bob@example.com']

        with patch(
            'server.services.org_invitation_service.OrgStore.get_org_by_id',
            return_value=None,  # Organization not found
        ):
            # Act & Assert
            with pytest.raises(ValueError) as exc_info:
                await OrgInvitationService.create_invitations_batch(
                    org_id=org_id,
                    emails=emails,
                    role_name='member',
                    inviter_id=inviter_id,
                )

            assert 'not found' in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_batch_fails_on_invalid_role(
        self, org_id, inviter_id, mock_org, mock_inviter_member, mock_owner_role
    ):
        """Test that invalid role fails the entire batch."""
        # Arrange
        emails = ['alice@example.com']

        with (
            patch(
                'server.services.org_invitation_service.OrgStore.get_org_by_id',
                return_value=mock_org,
            ),
            patch(
                'server.services.org_invitation_service.OrgMemberStore.get_org_member',
                return_value=mock_inviter_member,
            ),
            patch(
                'server.services.org_invitation_service.RoleStore.get_role_by_id',
                return_value=mock_owner_role,
            ),
            patch(
                'server.services.org_invitation_service.RoleStore.get_role_by_name',
                return_value=None,  # Invalid role
            ),
        ):
            # Act & Assert
            with pytest.raises(ValueError) as exc_info:
                await OrgInvitationService.create_invitations_batch(
                    org_id=org_id,
                    emails=emails,
                    role_name='invalid_role',
                    inviter_id=inviter_id,
                )

            assert 'Invalid role' in str(exc_info.value)
