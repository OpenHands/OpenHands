"""Service for managing organization members."""

from uuid import UUID

from server.routes.org_models import OrgMemberPage, OrgMemberResponse
from storage.org_member_store import OrgMemberStore


class OrgMemberService:
    """Service for organization member operations."""

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
