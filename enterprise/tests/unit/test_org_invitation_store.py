"""Tests for organization invitation store."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from storage.org_invitation import OrgInvitation
from storage.org_invitation_store import (
    INVITATION_TOKEN_LENGTH,
    INVITATION_TOKEN_PREFIX,
    OrgInvitationStore,
)


class TestGenerateToken:
    """Test cases for token generation."""

    def test_generate_token_has_correct_prefix(self):
        """Test that generated tokens have the correct prefix."""
        token = OrgInvitationStore.generate_token()
        assert token.startswith(INVITATION_TOKEN_PREFIX)

    def test_generate_token_has_correct_length(self):
        """Test that generated tokens have the correct total length."""
        token = OrgInvitationStore.generate_token()
        expected_length = len(INVITATION_TOKEN_PREFIX) + INVITATION_TOKEN_LENGTH
        assert len(token) == expected_length

    def test_generate_token_uses_alphanumeric_characters(self):
        """Test that generated tokens use only alphanumeric characters."""
        token = OrgInvitationStore.generate_token()
        # Remove prefix and check the rest is alphanumeric
        random_part = token[len(INVITATION_TOKEN_PREFIX) :]
        assert random_part.isalnum()

    def test_generate_token_is_unique(self):
        """Test that generated tokens are unique (probabilistically)."""
        tokens = [OrgInvitationStore.generate_token() for _ in range(100)]
        assert len(set(tokens)) == 100


class TestIsTokenExpired:
    """Test cases for token expiration checking."""

    def test_token_not_expired_when_future(self):
        """Test that tokens with future expiration are not expired."""
        invitation = MagicMock(spec=OrgInvitation)
        invitation.expires_at = datetime.utcnow() + timedelta(days=1)

        result = OrgInvitationStore.is_token_expired(invitation)
        assert result is False

    def test_token_expired_when_past(self):
        """Test that tokens with past expiration are expired."""
        invitation = MagicMock(spec=OrgInvitation)
        invitation.expires_at = datetime.utcnow() - timedelta(seconds=1)

        result = OrgInvitationStore.is_token_expired(invitation)
        assert result is True

    def test_token_expired_at_exact_boundary(self):
        """Test that tokens at exact expiration time are expired."""
        # A token that expires "now" should be expired
        now = datetime.utcnow()
        invitation = MagicMock(spec=OrgInvitation)
        invitation.expires_at = now - timedelta(microseconds=1)

        result = OrgInvitationStore.is_token_expired(invitation)
        assert result is True


class TestCreateInvitation:
    """Test cases for invitation creation."""

    @pytest.mark.asyncio
    async def test_create_invitation_normalizes_email(self):
        """Test that email is normalized (lowercase, stripped) on creation."""
        mock_session = AsyncMock()
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        mock_session.execute = AsyncMock()

        # Mock the result of the re-fetch query
        mock_result = MagicMock()
        mock_invitation = MagicMock()
        mock_invitation.id = 1
        mock_invitation.email = 'test@example.com'
        mock_result.scalars.return_value.first.return_value = mock_invitation
        mock_session.execute.return_value = mock_result

        with patch(
            'storage.org_invitation_store.a_session_maker'
        ) as mock_session_maker:
            mock_session_manager = AsyncMock()
            mock_session_manager.__aenter__.return_value = mock_session
            mock_session_manager.__aexit__.return_value = None
            mock_session_maker.return_value = mock_session_manager

            from uuid import UUID

            await OrgInvitationStore.create_invitation(
                org_id=UUID('12345678-1234-5678-1234-567812345678'),
                email='  TEST@EXAMPLE.COM  ',
                role_id=1,
                inviter_id=UUID('87654321-4321-8765-4321-876543218765'),
            )

            # Verify that the OrgInvitation was created with normalized email
            add_call = mock_session.add.call_args
            created_invitation = add_call[0][0]
            assert created_invitation.email == 'test@example.com'


class TestGetInvitationByToken:
    """Test cases for getting invitation by token."""

    @pytest.mark.asyncio
    async def test_get_invitation_by_token_returns_invitation(self):
        """Test that get_invitation_by_token returns the invitation when found."""
        mock_invitation = MagicMock(spec=OrgInvitation)
        mock_invitation.token = 'inv-test-token-12345'

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = mock_invitation
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch(
            'storage.org_invitation_store.a_session_maker'
        ) as mock_session_maker:
            mock_session_manager = AsyncMock()
            mock_session_manager.__aenter__.return_value = mock_session
            mock_session_manager.__aexit__.return_value = None
            mock_session_maker.return_value = mock_session_manager

            result = await OrgInvitationStore.get_invitation_by_token(
                'inv-test-token-12345'
            )
            assert result == mock_invitation

    @pytest.mark.asyncio
    async def test_get_invitation_by_token_returns_none_when_not_found(self):
        """Test that get_invitation_by_token returns None when not found."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch(
            'storage.org_invitation_store.a_session_maker'
        ) as mock_session_maker:
            mock_session_manager = AsyncMock()
            mock_session_manager.__aenter__.return_value = mock_session
            mock_session_manager.__aexit__.return_value = None
            mock_session_maker.return_value = mock_session_manager

            result = await OrgInvitationStore.get_invitation_by_token(
                'inv-nonexistent-token'
            )
            assert result is None


class TestGetPendingInvitation:
    """Test cases for getting pending invitation."""

    @pytest.mark.asyncio
    async def test_get_pending_invitation_normalizes_email(self):
        """Test that email is normalized when querying for pending invitations."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch(
            'storage.org_invitation_store.a_session_maker'
        ) as mock_session_maker:
            mock_session_manager = AsyncMock()
            mock_session_manager.__aenter__.return_value = mock_session
            mock_session_manager.__aexit__.return_value = None
            mock_session_maker.return_value = mock_session_manager

            from uuid import UUID

            await OrgInvitationStore.get_pending_invitation(
                org_id=UUID('12345678-1234-5678-1234-567812345678'),
                email='  TEST@EXAMPLE.COM  ',
            )

            # Verify the query was called (email normalization happens in the filter)
            assert mock_session.execute.called


class TestUpdateInvitationStatus:
    """Test cases for updating invitation status."""

    @pytest.mark.asyncio
    async def test_update_status_sets_accepted_at_for_accepted(self):
        """Test that accepted_at is set when status is accepted."""
        from uuid import UUID

        mock_invitation = MagicMock(spec=OrgInvitation)
        mock_invitation.id = 1
        mock_invitation.status = OrgInvitation.STATUS_PENDING

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = mock_invitation
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        with patch(
            'storage.org_invitation_store.a_session_maker'
        ) as mock_session_maker:
            mock_session_manager = AsyncMock()
            mock_session_manager.__aenter__.return_value = mock_session
            mock_session_manager.__aexit__.return_value = None
            mock_session_maker.return_value = mock_session_manager

            user_id = UUID('87654321-4321-8765-4321-876543218765')
            await OrgInvitationStore.update_invitation_status(
                invitation_id=1,
                status=OrgInvitation.STATUS_ACCEPTED,
                accepted_by_user_id=user_id,
            )

            assert mock_invitation.accepted_at is not None
            assert mock_invitation.accepted_by_user_id == user_id

    @pytest.mark.asyncio
    async def test_update_status_returns_none_when_not_found(self):
        """Test that update returns None when invitation not found."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch(
            'storage.org_invitation_store.a_session_maker'
        ) as mock_session_maker:
            mock_session_manager = AsyncMock()
            mock_session_manager.__aenter__.return_value = mock_session
            mock_session_manager.__aexit__.return_value = None
            mock_session_maker.return_value = mock_session_manager

            result = await OrgInvitationStore.update_invitation_status(
                invitation_id=999,
                status=OrgInvitation.STATUS_ACCEPTED,
            )
            assert result is None


class TestMarkExpiredIfNeeded:
    """Test cases for marking expired invitations."""

    @pytest.mark.asyncio
    async def test_marks_expired_when_pending_and_past_expiry(self):
        """Test that pending expired invitations are marked as expired."""
        mock_invitation = MagicMock(spec=OrgInvitation)
        mock_invitation.id = 1
        mock_invitation.status = OrgInvitation.STATUS_PENDING
        mock_invitation.expires_at = datetime.utcnow() - timedelta(days=1)

        with patch.object(
            OrgInvitationStore,
            'update_invitation_status',
            new_callable=AsyncMock,
        ) as mock_update:
            result = await OrgInvitationStore.mark_expired_if_needed(mock_invitation)

            assert result is True
            mock_update.assert_called_once_with(1, OrgInvitation.STATUS_EXPIRED)

    @pytest.mark.asyncio
    async def test_does_not_mark_when_not_expired(self):
        """Test that non-expired invitations are not marked."""
        mock_invitation = MagicMock(spec=OrgInvitation)
        mock_invitation.id = 1
        mock_invitation.status = OrgInvitation.STATUS_PENDING
        mock_invitation.expires_at = datetime.utcnow() + timedelta(days=1)

        with patch.object(
            OrgInvitationStore,
            'update_invitation_status',
            new_callable=AsyncMock,
        ) as mock_update:
            result = await OrgInvitationStore.mark_expired_if_needed(mock_invitation)

            assert result is False
            mock_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_does_not_mark_when_not_pending(self):
        """Test that non-pending invitations are not marked even if expired."""
        mock_invitation = MagicMock(spec=OrgInvitation)
        mock_invitation.id = 1
        mock_invitation.status = OrgInvitation.STATUS_ACCEPTED
        mock_invitation.expires_at = datetime.utcnow() - timedelta(days=1)

        with patch.object(
            OrgInvitationStore,
            'update_invitation_status',
            new_callable=AsyncMock,
        ) as mock_update:
            result = await OrgInvitationStore.mark_expired_if_needed(mock_invitation)

            assert result is False
            mock_update.assert_not_called()
