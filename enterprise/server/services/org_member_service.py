"""Service for managing organization members."""

from uuid import UUID

from server.constants import ROLE_ADMIN, ROLE_MEMBER, ROLE_OWNER
from server.routes.org_models import (
    CannotModifySelfError,
    InsufficientPermissionError,
    InvalidRoleError,
    LastOwnerError,
    MemberUpdateError,
    MeResponse,
    OrgMemberNotFoundError,
    OrgMemberPage,
    OrgMemberResponse,
    OrgMemberUpdate,
    RoleNotFoundError,
)
from storage.org_member_store import OrgMemberStore
from storage.role_store import RoleStore
from storage.user_store import UserStore

from openhands.utils.async_utils import call_sync_from_async


class OrgMemberService:
    """Service for organization member operations."""

    @staticmethod
    def get_me(org_id: UUID, user_id: UUID) -> MeResponse:
        """Get the current user's membership record for an organization.

        Retrieves the authenticated user's role, status, email, and LLM override
        fields (with masked API keys) within the specified organization.

        Args:
            org_id: Organization ID (UUID)
            user_id: User ID (UUID)

        Returns:
            MeResponse: The user's membership data with masked API keys

        Raises:
            OrgMemberNotFoundError: If user is not a member of the organization
            RoleNotFoundError: If the role associated with the member is not found
        """
        # Look up the user's membership in this org
        org_member = OrgMemberStore.get_org_member(org_id, user_id)
        if org_member is None:
            raise OrgMemberNotFoundError(str(org_id), str(user_id))

        # Resolve role name from role_id
        role = RoleStore.get_role_by_id(org_member.role_id)
        if role is None:
            raise RoleNotFoundError(org_member.role_id)

        # Get user email
        user = UserStore.get_user_by_id(str(user_id))
        email = user.email if user and user.email else ''

        return MeResponse.from_org_member(org_member, role, email)

    @staticmethod
    async def get_org_members(
        org_id: UUID,
        current_user_id: UUID,
        page_id: str | None = None,
        limit: int = 100,
    ) -> tuple[bool, str | None, OrgMemberPage | None]:
        """Get organization members with authorization check.

        Returns:
            Tuple of (success, error_code, data). If success is True, error_code is None.
        """
        # Verify current user is a member of the organization
        requester_membership = OrgMemberStore.get_org_member(org_id, current_user_id)
        if not requester_membership:
            return False, 'not_a_member', None

        # Parse page_id to get offset (page_id is offset encoded as string)
        offset = 0
        if page_id is not None:
            try:
                offset = int(page_id)
                if offset < 0:
                    return False, 'invalid_page_id', None
            except ValueError:
                return False, 'invalid_page_id', None

        # Call store to get paginated members
        members, has_more = await OrgMemberStore.get_org_members_paginated(
            org_id=org_id, offset=offset, limit=limit
        )

        # Transform data to response format
        items = []
        for member in members:
            # Access user and role relationships (eagerly loaded)
            user = member.user
            role = member.role

            items.append(
                OrgMemberResponse(
                    user_id=str(member.user_id),
                    email=user.email if user else None,
                    role_id=member.role_id,
                    role=role.name if role else '',
                    role_rank=role.rank if role else 0,
                    status=member.status,
                )
            )

        # Calculate next_page_id
        next_page_id = None
        if has_more:
            next_page_id = str(offset + limit)

        return True, None, OrgMemberPage(items=items, next_page_id=next_page_id)

    @staticmethod
    async def remove_org_member(
        org_id: UUID,
        target_user_id: UUID,
        current_user_id: UUID,
    ) -> tuple[bool, str | None]:
        """Remove a member from an organization.

        Returns:
            Tuple of (success, error_message). If success is True, error_message is None.
        """

        def _remove_member():
            # Get current user's membership in the org
            requester_membership = OrgMemberStore.get_org_member(
                org_id, current_user_id
            )
            if not requester_membership:
                return False, 'not_a_member'

            # Check if trying to remove self
            if str(current_user_id) == str(target_user_id):
                return False, 'cannot_remove_self'

            # Get target user's membership
            target_membership = OrgMemberStore.get_org_member(org_id, target_user_id)
            if not target_membership:
                return False, 'member_not_found'

            requester_role = RoleStore.get_role_by_id(requester_membership.role_id)
            target_role = RoleStore.get_role_by_id(target_membership.role_id)

            if not requester_role or not target_role:
                return False, 'role_not_found'

            # Check permission based on roles
            if not OrgMemberService._can_remove_member(
                requester_role.name, target_role.name
            ):
                return False, 'insufficient_permission'

            # Check if removing the last owner
            if target_role.name == ROLE_OWNER:
                if OrgMemberService._is_last_owner(org_id, target_user_id):
                    return False, 'cannot_remove_last_owner'

            # Perform the removal
            success = OrgMemberStore.remove_user_from_org(org_id, target_user_id)
            if not success:
                return False, 'removal_failed'

            # Update user's current_org_id if it points to the org they were removed from
            user = UserStore.get_user_by_id(str(target_user_id))
            if user and user.current_org_id == org_id:
                # Set current_org_id to personal workspace (org.id == user.id)
                UserStore.update_current_org(str(target_user_id), target_user_id)

            return True, None

        return await call_sync_from_async(_remove_member)

    @staticmethod
    async def update_org_member(
        org_id: UUID,
        target_user_id: UUID,
        current_user_id: UUID,
        update_data: OrgMemberUpdate,
    ) -> OrgMemberResponse:
        """Update a member's role in an organization.

        Permission rules:
        - Admins can change roles of users (rank > ADMIN_RANK) to Admin or User
        - Admins cannot modify other Admins or Owners
        - Owners can change roles of non-owners (rank > OWNER_RANK) to any role
        - Owners cannot modify other Owners

        Args:
            org_id: Organization ID
            target_user_id: User ID of the member to update
            current_user_id: User ID of the requester
            update_data: Update data containing fields to modify

        Returns:
            OrgMemberResponse: The updated member data

        Raises:
            OrgMemberNotFoundError: If requester or target is not a member
            CannotModifySelfError: If trying to modify self
            RoleNotFoundError: If role configuration is invalid
            InvalidRoleError: If new_role_name is not a valid role
            InsufficientPermissionError: If requester lacks permission
            LastOwnerError: If trying to demote the last owner
            MemberUpdateError: If update operation fails
        """
        new_role_name = update_data.role

        def _update_member():
            # Get current user's membership in the org
            requester_membership = OrgMemberStore.get_org_member(
                org_id, current_user_id
            )
            if not requester_membership:
                raise OrgMemberNotFoundError(str(org_id), str(current_user_id))

            # Check if trying to modify self
            if str(current_user_id) == str(target_user_id):
                raise CannotModifySelfError('modify')

            # Get target user's membership
            target_membership = OrgMemberStore.get_org_member(org_id, target_user_id)
            if not target_membership:
                raise OrgMemberNotFoundError(str(org_id), str(target_user_id))

            # Get roles
            requester_role = RoleStore.get_role_by_id(requester_membership.role_id)
            target_role = RoleStore.get_role_by_id(target_membership.role_id)

            if not requester_role:
                raise RoleNotFoundError(requester_membership.role_id)
            if not target_role:
                raise RoleNotFoundError(target_membership.role_id)

            # If no role change requested, return current state
            if new_role_name is None:
                user = UserStore.get_user_by_id(str(target_user_id))
                return OrgMemberResponse(
                    user_id=str(target_membership.user_id),
                    email=user.email if user else None,
                    role_id=target_membership.role_id,
                    role=target_role.name,
                    role_rank=target_role.rank,
                    status=target_membership.status,
                )

            # Validate new role exists
            new_role = RoleStore.get_role_by_name(new_role_name.lower())
            if not new_role:
                raise InvalidRoleError(new_role_name)

            # Check permission to modify target
            if not OrgMemberService._can_update_member_role(
                requester_role.name, target_role.name, new_role.name
            ):
                raise InsufficientPermissionError(
                    'You do not have permission to modify this member'
                )

            # Check if demoting the last owner
            if (
                target_role.name == ROLE_OWNER
                and new_role.name != ROLE_OWNER
                and OrgMemberService._is_last_owner(org_id, target_user_id)
            ):
                raise LastOwnerError('demote')

            # Perform the update
            updated_member = OrgMemberStore.update_user_role_in_org(
                org_id, target_user_id, new_role.id
            )
            if not updated_member:
                raise MemberUpdateError('Failed to update member')

            # Get user email for response
            user = UserStore.get_user_by_id(str(target_user_id))

            return OrgMemberResponse(
                user_id=str(updated_member.user_id),
                email=user.email if user else None,
                role_id=updated_member.role_id,
                role=new_role.name,
                role_rank=new_role.rank,
                status=updated_member.status,
            )

        return await call_sync_from_async(_update_member)

    @staticmethod
    def _can_update_member_role(
        requester_role_name: str, target_role_name: str, new_role_name: str
    ) -> bool:
        """Check if requester can change target's role to new_role.

        Permission rules:
        - Owners can modify admins and users, can set any role
        - Owners cannot modify other owners
        - Admins can only modify users
        - Admins can only set admin or user roles (not owner)
        """
        is_requester_owner = requester_role_name == ROLE_OWNER
        is_requester_admin = requester_role_name == ROLE_ADMIN
        is_target_owner = target_role_name == ROLE_OWNER
        is_target_admin = target_role_name == ROLE_ADMIN
        is_new_role_owner = new_role_name == ROLE_OWNER

        if is_requester_owner:
            # Owners cannot modify other owners
            if is_target_owner:
                return False
            # Owners can set any role (owner, admin, user)
            return True
        elif is_requester_admin:
            # Admins cannot modify owners or other admins
            if is_target_owner or is_target_admin:
                return False
            # Admins can only set admin or user roles (not owner)
            return not is_new_role_owner
        return False

    @staticmethod
    def _can_remove_member(requester_role_name: str, target_role_name: str) -> bool:
        """Check if requester can remove target based on roles."""
        if requester_role_name == ROLE_OWNER:
            return True
        elif requester_role_name == ROLE_ADMIN:
            # Admins can only remove members (not owners or other admins)
            return target_role_name == ROLE_MEMBER
        return False

    @staticmethod
    def _is_last_owner(org_id: UUID, user_id: UUID) -> bool:
        """Check if user is the last owner of the organization."""
        members = OrgMemberStore.get_org_members(org_id)
        owners = []
        for m in members:
            # Use role_id (column) instead of role (relationship) to avoid DetachedInstanceError
            role = RoleStore.get_role_by_id(m.role_id)
            if role and role.name == ROLE_OWNER:
                owners.append(m)
        return len(owners) == 1 and str(owners[0].user_id) == str(user_id)
