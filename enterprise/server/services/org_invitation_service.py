"""Service for managing organization invitations."""

import asyncio
from uuid import UUID

from server.auth.token_manager import TokenManager
from server.constants import ROLE_ADMIN, ROLE_OWNER
from server.routes.org_invitation_models import (
    EmailMismatchError,
    InsufficientPermissionError,
    InvitationExpiredError,
    InvitationInvalidError,
    UserAlreadyMemberError,
)
from server.services.email_service import EmailService
from storage.org_invitation import OrgInvitation
from storage.org_invitation_store import OrgInvitationStore
from storage.org_member_store import OrgMemberStore
from storage.org_service import OrgService
from storage.org_store import OrgStore
from storage.role_store import RoleStore
from storage.user_store import UserStore

from openhands.core.logger import openhands_logger as logger


class OrgInvitationService:
    """Service for organization invitation operations."""

    @staticmethod
    async def create_invitation(
        org_id: UUID,
        email: str,
        role_name: str,
        inviter_id: UUID,
    ) -> OrgInvitation:
        """Create a new organization invitation.

        This method:
        1. Validates the organization exists
        2. Validates this is not a personal workspace
        3. Checks inviter has owner/admin role
        4. Validates role assignment permissions
        5. Checks if user is already a member
        6. Creates the invitation
        7. Sends the invitation email

        Args:
            org_id: Organization UUID
            email: Invitee's email address
            role_name: Role to assign on acceptance (owner, admin, member)
            inviter_id: User ID of the person creating the invitation

        Returns:
            OrgInvitation: The created invitation

        Raises:
            ValueError: If organization or role not found
            InsufficientPermissionError: If inviter lacks permission
            UserAlreadyMemberError: If email is already a member
            InvitationAlreadyExistsError: If pending invitation exists
        """
        email = email.lower().strip()

        logger.info(
            'Creating organization invitation',
            extra={
                'org_id': str(org_id),
                'email': email,
                'role_name': role_name,
                'inviter_id': str(inviter_id),
            },
        )

        # Step 1: Validate organization exists
        org = OrgStore.get_org_by_id(org_id)
        if not org:
            raise ValueError(f'Organization {org_id} not found')

        # Step 2: Check this is not a personal workspace
        # A personal workspace has org_id matching the user's id
        if str(org_id) == str(inviter_id):
            raise InsufficientPermissionError(
                'Cannot invite users to a personal workspace'
            )

        # Step 3: Check inviter is a member and has permission
        inviter_member = OrgMemberStore.get_org_member(org_id, inviter_id)
        if not inviter_member:
            raise InsufficientPermissionError(
                'You are not a member of this organization'
            )

        inviter_role = RoleStore.get_role_by_id(inviter_member.role_id)
        if not inviter_role or inviter_role.name not in [ROLE_OWNER, ROLE_ADMIN]:
            raise InsufficientPermissionError('Only owners and admins can invite users')

        # Step 4: Validate role assignment permissions
        role_name_lower = role_name.lower()
        if role_name_lower == ROLE_OWNER and inviter_role.name != ROLE_OWNER:
            raise InsufficientPermissionError('Only owners can invite with owner role')

        # Get the target role
        target_role = RoleStore.get_role_by_name(role_name_lower)
        if not target_role:
            raise ValueError(f'Invalid role: {role_name}')

        # Step 5: Check if user is already a member (by email)
        existing_user = await UserStore.get_user_by_email_async(email)
        if existing_user:
            existing_member = OrgMemberStore.get_org_member(org_id, existing_user.id)
            if existing_member:
                raise UserAlreadyMemberError(
                    'User is already a member of this organization'
                )

        # Step 6: Create the invitation
        invitation = await OrgInvitationStore.create_invitation(
            org_id=org_id,
            email=email,
            role_id=target_role.id,
            inviter_id=inviter_id,
        )

        # Step 7: Send invitation email
        try:
            # Get inviter info for the email
            inviter_user = UserStore.get_user_by_id(str(inviter_member.user_id))
            inviter_name = 'A team member'
            if inviter_user and inviter_user.email:
                inviter_name = inviter_user.email.split('@')[0]

            EmailService.send_invitation_email(
                to_email=email,
                org_name=org.name,
                inviter_name=inviter_name,
                role_name=target_role.name,
                invitation_token=invitation.token,
                invitation_id=invitation.id,
            )
        except Exception as e:
            logger.error(
                'Failed to send invitation email',
                extra={
                    'invitation_id': invitation.id,
                    'email': email,
                    'error': str(e),
                },
            )
            # Don't fail the invitation creation if email fails
            # The user can still access via direct link

        return invitation

    @staticmethod
    async def create_invitations_batch(
        org_id: UUID,
        emails: list[str],
        role_name: str,
        inviter_id: UUID,
    ) -> tuple[list[OrgInvitation], list[tuple[str, str]]]:
        """Create multiple organization invitations concurrently.

        Validates permissions once upfront, then creates invitations in parallel.

        Args:
            org_id: Organization UUID
            emails: List of invitee email addresses
            role_name: Role to assign on acceptance (owner, admin, member)
            inviter_id: User ID of the person creating the invitations

        Returns:
            Tuple of (successful_invitations, failed_emails_with_errors)

        Raises:
            ValueError: If organization or role not found
            InsufficientPermissionError: If inviter lacks permission
        """
        logger.info(
            'Creating batch organization invitations',
            extra={
                'org_id': str(org_id),
                'email_count': len(emails),
                'role_name': role_name,
                'inviter_id': str(inviter_id),
            },
        )

        # Step 1: Validate permissions upfront (shared for all emails)
        org = OrgStore.get_org_by_id(org_id)
        if not org:
            raise ValueError(f'Organization {org_id} not found')

        if str(org_id) == str(inviter_id):
            raise InsufficientPermissionError(
                'Cannot invite users to a personal workspace'
            )

        inviter_member = OrgMemberStore.get_org_member(org_id, inviter_id)
        if not inviter_member:
            raise InsufficientPermissionError(
                'You are not a member of this organization'
            )

        inviter_role = RoleStore.get_role_by_id(inviter_member.role_id)
        if not inviter_role or inviter_role.name not in [ROLE_OWNER, ROLE_ADMIN]:
            raise InsufficientPermissionError('Only owners and admins can invite users')

        role_name_lower = role_name.lower()
        if role_name_lower == ROLE_OWNER and inviter_role.name != ROLE_OWNER:
            raise InsufficientPermissionError('Only owners can invite with owner role')

        target_role = RoleStore.get_role_by_name(role_name_lower)
        if not target_role:
            raise ValueError(f'Invalid role: {role_name}')

        # Step 2: Create invitations concurrently
        async def create_single(
            email: str,
        ) -> tuple[str, OrgInvitation | None, str | None]:
            """Create single invitation, return (email, invitation, error)."""
            try:
                invitation = await OrgInvitationService.create_invitation(
                    org_id=org_id,
                    email=email,
                    role_name=role_name,
                    inviter_id=inviter_id,
                )
                return (email, invitation, None)
            except (UserAlreadyMemberError, ValueError) as e:
                return (email, None, str(e))

        results = await asyncio.gather(*[create_single(email) for email in emails])

        # Step 3: Separate successes and failures
        successful: list[OrgInvitation] = []
        failed: list[tuple[str, str]] = []
        for email, invitation, error in results:
            if invitation:
                successful.append(invitation)
            elif error:
                failed.append((email, error))

        logger.info(
            'Batch invitation creation completed',
            extra={
                'org_id': str(org_id),
                'successful': len(successful),
                'failed': len(failed),
            },
        )

        return successful, failed

    @staticmethod
    async def accept_invitation(token: str, user_id: UUID) -> OrgInvitation:
        """Accept an organization invitation.

        This method:
        1. Validates the token and invitation status
        2. Checks expiration
        3. Verifies user is not already a member
        4. Creates LiteLLM integration
        5. Adds user to the organization
        6. Marks invitation as accepted

        Args:
            token: The invitation token
            user_id: The user accepting the invitation

        Returns:
            OrgInvitation: The accepted invitation

        Raises:
            InvitationInvalidError: If token is invalid or invitation not pending
            InvitationExpiredError: If invitation has expired
            UserAlreadyMemberError: If user is already a member
        """
        logger.info(
            'Accepting organization invitation',
            extra={
                'token_prefix': token[:10] + '...' if len(token) > 10 else token,
                'user_id': str(user_id),
            },
        )

        # Step 1: Get and validate invitation
        invitation = await OrgInvitationStore.get_invitation_by_token(token)

        if not invitation:
            raise InvitationInvalidError('Invalid invitation token')

        if invitation.status != OrgInvitation.STATUS_PENDING:
            if invitation.status == OrgInvitation.STATUS_ACCEPTED:
                raise InvitationInvalidError('Invitation has already been accepted')
            elif invitation.status == OrgInvitation.STATUS_REVOKED:
                raise InvitationInvalidError('Invitation has been revoked')
            else:
                raise InvitationInvalidError('Invitation is no longer valid')

        # Step 2: Check expiration
        if OrgInvitationStore.is_token_expired(invitation):
            await OrgInvitationStore.update_invitation_status(
                invitation.id, OrgInvitation.STATUS_EXPIRED
            )
            raise InvitationExpiredError('Invitation has expired')

        # Step 2.5: Verify user email matches invitation email
        user = await UserStore.get_user_by_id_async(str(user_id))
        if not user:
            raise InvitationInvalidError('User not found')

        user_email = user.email
        # Fallback: fetch email from Keycloak if not in database (for existing users)
        if not user_email:
            token_manager = TokenManager()
            user_info = await token_manager.get_user_info_from_user_id(str(user_id))
            user_email = user_info.get('email') if user_info else None

        if not user_email:
            raise EmailMismatchError('Your account does not have an email address')

        user_email = user_email.lower().strip()
        invitation_email = invitation.email.lower().strip()

        if user_email != invitation_email:
            logger.warning(
                'Email mismatch during invitation acceptance',
                extra={
                    'user_id': str(user_id),
                    'user_email': user_email,
                    'invitation_email': invitation_email,
                    'invitation_id': invitation.id,
                },
            )
            raise EmailMismatchError()

        # Step 3: Check if user is already a member
        existing_member = OrgMemberStore.get_org_member(invitation.org_id, user_id)
        if existing_member:
            raise UserAlreadyMemberError(
                'You are already a member of this organization'
            )

        # Step 4: Create LiteLLM integration for the user in the new org
        try:
            settings = await OrgService.create_litellm_integration(
                invitation.org_id, str(user_id)
            )
        except Exception as e:
            logger.error(
                'Failed to create LiteLLM integration for invitation acceptance',
                extra={
                    'invitation_id': invitation.id,
                    'user_id': str(user_id),
                    'org_id': str(invitation.org_id),
                    'error': str(e),
                },
            )
            raise InvitationInvalidError(
                'Failed to set up organization access. Please try again.'
            )

        # Step 5: Add user to organization
        from storage.org_member_store import OrgMemberStore as OMS

        org_member_kwargs = OMS.get_kwargs_from_settings(settings)
        # Don't override with org defaults - use invitation-specified role
        org_member_kwargs.pop('llm_model', None)
        org_member_kwargs.pop('llm_base_url', None)

        OrgMemberStore.add_user_to_org(
            org_id=invitation.org_id,
            user_id=user_id,
            role_id=invitation.role_id,
            llm_api_key=settings.llm_api_key,
            status='active',
        )

        # Step 6: Mark invitation as accepted
        updated_invitation = await OrgInvitationStore.update_invitation_status(
            invitation.id,
            OrgInvitation.STATUS_ACCEPTED,
            accepted_by_user_id=user_id,
        )

        logger.info(
            'Organization invitation accepted',
            extra={
                'invitation_id': invitation.id,
                'user_id': str(user_id),
                'org_id': str(invitation.org_id),
                'role_id': invitation.role_id,
            },
        )

        return updated_invitation
