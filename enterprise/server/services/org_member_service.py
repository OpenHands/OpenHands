"""Service for managing organization members."""

from uuid import UUID

from server.routes.org_models import (
    MeResponse,
    OrgMemberNotFoundError,
    OrgMemberPage,
    OrgMemberResponse,
    RoleNotFoundError,
)
from storage.org_member_store import OrgMemberStore
from storage.role_store import RoleStore
from storage.user_store import UserStore

from openhands.utils.async_utils import call_sync_from_async

# Role rank constants
OWNER_RANK = 10
ADMIN_RANK = 20


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
                    role_name=role.name if role else '',
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

            # Check permission based on role ranks
            if not OrgMemberService._can_remove_member(
                requester_role.rank, target_role.rank
            ):
                return False, 'insufficient_permission'

            # Check if removing the last owner
            if target_role.rank == OWNER_RANK:
                if OrgMemberService._is_last_owner(org_id, target_user_id):
                    return False, 'cannot_remove_last_owner'

            # Perform the removal
            success = OrgMemberStore.remove_user_from_org(org_id, target_user_id)
            if not success:
                return False, 'removal_failed'

            return True, None

        return await call_sync_from_async(_remove_member)

    @staticmethod
    def _can_remove_member(requester_rank: int, target_rank: int) -> bool:
        """Check if requester can remove target based on role ranks."""
        if requester_rank == OWNER_RANK:
            return True
        elif requester_rank == ADMIN_RANK:
            return target_rank > ADMIN_RANK
        return False

    @staticmethod
    def _is_last_owner(org_id: UUID, user_id: UUID) -> bool:
        """Check if user is the last owner of the organization."""
        members = OrgMemberStore.get_org_members(org_id)
        owners = []
        for m in members:
            # Use role_id (column) instead of role (relationship) to avoid DetachedInstanceError
            role = RoleStore.get_role_by_id(m.role_id)
            if role and role.rank == OWNER_RANK:
                owners.append(m)
        return len(owners) == 1 and str(owners[0].user_id) == str(user_id)
