"""
Permission-based authorization dependencies for API endpoints.

This module provides FastAPI dependencies for checking user permissions
within organizations. It uses a permission-based authorization model where
roles (owner, admin, member) are mapped to specific permissions.

Permissions are defined in the Permission enum and mapped to roles via
ROLE_PERMISSIONS. This allows fine-grained access control while maintaining
the familiar role-based hierarchy.

Usage:
    from server.auth.authorization import (
        Permission,
        require_permission,
    )

    @router.get('/{org_id}/settings')
    async def get_settings(
        org_id: UUID,
        user_id: str = Depends(require_permission(Permission.VIEW_LLM_SETTINGS)),
    ):
        # Only users with VIEW_LLM_SETTINGS permission can access
        ...

    @router.patch('/{org_id}/settings')
    async def update_settings(
        org_id: UUID,
        user_id: str = Depends(require_permission(Permission.EDIT_LLM_SETTINGS)),
    ):
        # Only users with EDIT_LLM_SETTINGS permission can access
        ...
"""

from enum import Enum
from uuid import UUID

from fastapi import Depends, HTTPException, status
from storage.org_member_store import OrgMemberStore
from storage.role import Role
from storage.role_store import RoleStore

from openhands.core.logger import openhands_logger as logger
from openhands.server.user_auth import get_user_id


class Permission(str, Enum):
    """Permissions that can be assigned to roles."""

    # Secrets
    MANAGE_SECRETS = 'manage_secrets'

    # MCP
    MANAGE_MCP = 'manage_mcp'

    # Integrations
    MANAGE_INTEGRATIONS = 'manage_integrations'

    # Application Settings
    MANAGE_APPLICATION_SETTINGS = 'manage_application_settings'

    # API Keys
    MANAGE_API_KEYS = 'manage_api_keys'

    # LLM Settings
    VIEW_LLM_SETTINGS = 'view_llm_settings'
    EDIT_LLM_SETTINGS = 'edit_llm_settings'

    # Billing
    VIEW_BILLING = 'view_billing'
    ADD_CREDITS = 'add_credits'

    # Organization Members
    INVITE_USER_TO_ORGANIZATION = 'invite_user_to_organization'
    CHANGE_USER_ROLE_MEMBER = 'change_user_role:member'
    CHANGE_USER_ROLE_ADMIN = 'change_user_role:admin'
    CHANGE_USER_ROLE_OWNER = 'change_user_role:owner'

    # Organization Management
    VIEW_ORG_SETTINGS = 'view_org_settings'
    CHANGE_ORGANIZATION_NAME = 'change_organization_name'
    DELETE_ORGANIZATION = 'delete_organization'

    # Temporary permissions until we finish the API updates.
    EDIT_ORG_SETTINGS = 'edit_org_settings'


class RoleName(str, Enum):
    """Role names used in the system."""

    OWNER = 'owner'
    ADMIN = 'admin'
    MEMBER = 'member'


# Permission mappings for each role
ROLE_PERMISSIONS: dict[RoleName, frozenset[Permission]] = {
    RoleName.OWNER: frozenset(
        [
            # Settings (Full access)
            Permission.MANAGE_SECRETS,
            Permission.MANAGE_MCP,
            Permission.MANAGE_INTEGRATIONS,
            Permission.MANAGE_APPLICATION_SETTINGS,
            Permission.MANAGE_API_KEYS,
            Permission.VIEW_LLM_SETTINGS,
            Permission.EDIT_LLM_SETTINGS,
            Permission.VIEW_BILLING,
            Permission.ADD_CREDITS,
            # Organization Members
            Permission.INVITE_USER_TO_ORGANIZATION,
            Permission.CHANGE_USER_ROLE_MEMBER,
            Permission.CHANGE_USER_ROLE_ADMIN,
            Permission.CHANGE_USER_ROLE_OWNER,
            # Organization Management
            Permission.VIEW_ORG_SETTINGS,
            Permission.EDIT_ORG_SETTINGS,
            # Organization Management (Owner only)
            Permission.CHANGE_ORGANIZATION_NAME,
            Permission.DELETE_ORGANIZATION,
        ]
    ),
    RoleName.ADMIN: frozenset(
        [
            # Settings (Full access)
            Permission.MANAGE_SECRETS,
            Permission.MANAGE_MCP,
            Permission.MANAGE_INTEGRATIONS,
            Permission.MANAGE_APPLICATION_SETTINGS,
            Permission.MANAGE_API_KEYS,
            Permission.VIEW_LLM_SETTINGS,
            Permission.EDIT_LLM_SETTINGS,
            Permission.VIEW_BILLING,
            Permission.ADD_CREDITS,
            # Organization Members
            Permission.INVITE_USER_TO_ORGANIZATION,
            Permission.CHANGE_USER_ROLE_MEMBER,
            Permission.CHANGE_USER_ROLE_ADMIN,
            # Organization Management
            Permission.VIEW_ORG_SETTINGS,
            Permission.EDIT_ORG_SETTINGS,
        ]
    ),
    RoleName.MEMBER: frozenset(
        [
            # Settings (Full access)
            Permission.MANAGE_SECRETS,
            Permission.MANAGE_MCP,
            Permission.MANAGE_INTEGRATIONS,
            Permission.MANAGE_APPLICATION_SETTINGS,
            Permission.MANAGE_API_KEYS,
            # Settings (View only)
            Permission.VIEW_ORG_SETTINGS,
            Permission.VIEW_LLM_SETTINGS,
        ]
    ),
}


def get_user_org_role(user_id: str, org_id: UUID | None) -> Role | None:
    """
    Get the user's role in an organization (synchronous version).

    Args:
        user_id: User ID (string that will be converted to UUID)
        org_id: Organization ID, or None to use the user's current organization

    Returns:
        Role object if user is a member, None otherwise
    """
    from uuid import UUID as parse_uuid

    if org_id is None:
        org_member = OrgMemberStore.get_org_member_for_current_org(parse_uuid(user_id))
    else:
        org_member = OrgMemberStore.get_org_member(org_id, parse_uuid(user_id))
    if not org_member:
        return None

    return RoleStore.get_role_by_id(org_member.role_id)


async def get_user_org_role_async(user_id: str, org_id: UUID | None) -> Role | None:
    """
    Get the user's role in an organization (async version).

    Args:
        user_id: User ID (string that will be converted to UUID)
        org_id: Organization ID, or None to use the user's current organization

    Returns:
        Role object if user is a member, None otherwise
    """
    from uuid import UUID as parse_uuid

    if org_id is None:
        org_member = await OrgMemberStore.get_org_member_for_current_org_async(
            parse_uuid(user_id)
        )
    else:
        org_member = await OrgMemberStore.get_org_member_async(
            org_id, parse_uuid(user_id)
        )
    if not org_member:
        return None

    return await RoleStore.get_role_by_id_async(org_member.role_id)


def get_role_permissions(role_name: str) -> frozenset[Permission]:
    """
    Get the permissions for a role.

    Args:
        role_name: Name of the role

    Returns:
        Set of permissions for the role
    """
    try:
        role_enum = RoleName(role_name)
        return ROLE_PERMISSIONS.get(role_enum, frozenset())
    except ValueError:
        return frozenset()


def has_permission(user_role: Role, permission: Permission) -> bool:
    """
    Check if a role has a specific permission.

    Args:
        user_role: User's Role object
        permission: Permission to check

    Returns:
        True if the role has the permission
    """
    permissions = get_role_permissions(user_role.name)
    return permission in permissions


def require_permission(permission: Permission):
    """
    Factory function that creates a dependency to require a specific permission.

    This creates a FastAPI dependency that:
    1. Extracts org_id from the path parameter
    2. Gets the authenticated user_id
    3. Checks if the user has the required permission in the organization
    4. Returns the user_id if authorized, raises HTTPException otherwise

    Usage:
        @router.get('/{org_id}/settings')
        async def get_settings(
            org_id: UUID,
            user_id: str = Depends(require_permission(Permission.VIEW_LLM_SETTINGS)),
        ):
            ...

    Args:
        permission: The permission required to access the endpoint

    Returns:
        Dependency function that validates permission and returns user_id
    """

    async def permission_checker(
        org_id: UUID | None = None,
        user_id: str | None = Depends(get_user_id),
    ) -> str:
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail='User not authenticated',
            )

        user_role = await get_user_org_role_async(user_id, org_id)

        if not user_role:
            logger.warning(
                'User not a member of organization',
                extra={'user_id': user_id, 'org_id': str(org_id)},
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail='User is not a member of this organization',
            )

        if not has_permission(user_role, permission):
            logger.warning(
                'Insufficient permissions',
                extra={
                    'user_id': user_id,
                    'org_id': str(org_id),
                    'user_role': user_role.name,
                    'required_permission': permission.value,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f'Requires {permission.value} permission',
            )

        return user_id

    return permission_checker
