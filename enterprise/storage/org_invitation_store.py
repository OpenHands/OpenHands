"""
Store class for managing organization invitations.
"""

import secrets
import string
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.orm import joinedload
from storage.database import a_session_maker
from storage.org_invitation import OrgInvitation

from openhands.core.logger import openhands_logger as logger

# Invitation token configuration
INVITATION_TOKEN_PREFIX = 'inv-'
INVITATION_TOKEN_LENGTH = 48  # Total length will be 52 with prefix
DEFAULT_EXPIRATION_DAYS = 7


class OrgInvitationStore:
    """Store for managing organization invitations."""

    @staticmethod
    def generate_token(length: int = INVITATION_TOKEN_LENGTH) -> str:
        """Generate a secure invitation token.

        Uses cryptographically secure random generation for tokens.
        Pattern from api_key_store.py.

        Args:
            length: Length of the random part of the token

        Returns:
            str: Token with prefix (e.g., 'inv-aBcDeF123...')
        """
        alphabet = string.ascii_letters + string.digits
        random_part = ''.join(secrets.choice(alphabet) for _ in range(length))
        return f'{INVITATION_TOKEN_PREFIX}{random_part}'

    @staticmethod
    async def create_invitation(
        org_id: UUID,
        email: str,
        role_id: int,
        inviter_id: UUID,
        expiration_days: int = DEFAULT_EXPIRATION_DAYS,
    ) -> OrgInvitation:
        """Create a new organization invitation.

        Args:
            org_id: Organization UUID
            email: Invitee's email address
            role_id: Role ID to assign on acceptance
            inviter_id: User ID of the person creating the invitation
            expiration_days: Days until the invitation expires

        Returns:
            OrgInvitation: The created invitation record
        """
        async with a_session_maker() as session:
            token = OrgInvitationStore.generate_token()
            # Use timezone-naive datetime for database compatibility
            expires_at = datetime.utcnow() + timedelta(days=expiration_days)

            invitation = OrgInvitation(
                token=token,
                org_id=org_id,
                email=email.lower().strip(),
                role_id=role_id,
                inviter_id=inviter_id,
                status=OrgInvitation.STATUS_PENDING,
                expires_at=expires_at,
            )
            session.add(invitation)
            await session.commit()

            # Re-fetch with eagerly loaded relationships to avoid DetachedInstanceError
            result = await session.execute(
                select(OrgInvitation)
                .options(joinedload(OrgInvitation.role))
                .filter(OrgInvitation.id == invitation.id)
            )
            invitation = result.scalars().first()

            logger.info(
                'Created organization invitation',
                extra={
                    'invitation_id': invitation.id,
                    'org_id': str(org_id),
                    'email': email,
                    'inviter_id': str(inviter_id),
                    'expires_at': expires_at.isoformat(),
                },
            )

            return invitation

    @staticmethod
    async def get_invitation_by_token(token: str) -> Optional[OrgInvitation]:
        """Get an invitation by its token.

        Args:
            token: The invitation token

        Returns:
            OrgInvitation or None if not found
        """
        async with a_session_maker() as session:
            result = await session.execute(
                select(OrgInvitation)
                .options(joinedload(OrgInvitation.org), joinedload(OrgInvitation.role))
                .filter(OrgInvitation.token == token)
            )
            return result.scalars().first()

    @staticmethod
    async def get_pending_invitation(
        org_id: UUID, email: str
    ) -> Optional[OrgInvitation]:
        """Get a pending invitation for an email in an organization.

        Args:
            org_id: Organization UUID
            email: Email address to check

        Returns:
            OrgInvitation or None if no pending invitation exists
        """
        async with a_session_maker() as session:
            result = await session.execute(
                select(OrgInvitation).filter(
                    and_(
                        OrgInvitation.org_id == org_id,
                        OrgInvitation.email == email.lower().strip(),
                        OrgInvitation.status == OrgInvitation.STATUS_PENDING,
                    )
                )
            )
            return result.scalars().first()

    @staticmethod
    async def update_invitation_status(
        invitation_id: int,
        status: str,
        accepted_by_user_id: Optional[UUID] = None,
    ) -> Optional[OrgInvitation]:
        """Update an invitation's status.

        Args:
            invitation_id: The invitation ID
            status: New status (pending, accepted, revoked, expired)
            accepted_by_user_id: User ID who accepted (only for 'accepted' status)

        Returns:
            Updated OrgInvitation or None if not found
        """
        async with a_session_maker() as session:
            result = await session.execute(
                select(OrgInvitation).filter(OrgInvitation.id == invitation_id)
            )
            invitation = result.scalars().first()

            if not invitation:
                return None

            old_status = invitation.status
            invitation.status = status

            if status == OrgInvitation.STATUS_ACCEPTED and accepted_by_user_id:
                # Use timezone-naive datetime for database compatibility
                invitation.accepted_at = datetime.utcnow()
                invitation.accepted_by_user_id = accepted_by_user_id

            await session.commit()
            await session.refresh(invitation)

            logger.info(
                'Updated invitation status',
                extra={
                    'invitation_id': invitation_id,
                    'old_status': old_status,
                    'new_status': status,
                    'accepted_by_user_id': (
                        str(accepted_by_user_id) if accepted_by_user_id else None
                    ),
                },
            )

            return invitation

    @staticmethod
    def is_token_expired(invitation: OrgInvitation) -> bool:
        """Check if an invitation token has expired.

        Args:
            invitation: The invitation to check

        Returns:
            bool: True if expired, False otherwise
        """
        # Use timezone-naive datetime for comparison (database stores without timezone)
        now = datetime.utcnow()
        return invitation.expires_at < now

    @staticmethod
    async def mark_expired_if_needed(invitation: OrgInvitation) -> bool:
        """Check if invitation is expired and update status if needed.

        Args:
            invitation: The invitation to check

        Returns:
            bool: True if invitation was marked as expired, False otherwise
        """
        if (
            invitation.status == OrgInvitation.STATUS_PENDING
            and OrgInvitationStore.is_token_expired(invitation)
        ):
            await OrgInvitationStore.update_invitation_status(
                invitation.id, OrgInvitation.STATUS_EXPIRED
            )
            return True
        return False
