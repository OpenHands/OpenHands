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

from fastapi import Depends, HTTPException, Request, status
from storage.org_member_store import OrgMemberStore
from storage.role import Role
from storage.role_store import RoleStore
from storage.user_store import UserStore

from openhands.app_server.user_auth import get_user_auth, get_user_id
from openhands.app_server.utils.logger import openhands_logger as logger


class Permission(str, Enum):
    """Permissions that can be assigned to roles."""

    # Secrets
    MANAGE_SECRETS = 'manage_secrets'

    # MCP
    MANAGE_MCP = 'manage_mcp'

    # Integrations
    MANAGE_INTEGRATIONS = 'manage_integrations'
    # Setting up an integration connection (admin/owner) vs the member-level
    # MANAGE_INTEGRATIONS (linking/using an already-configured integration).
    MANAGE_INTEGRATION_PROVIDERS = 'manage_integration_providers'

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
    CREATE_ORGANIZATION = 'create_organization'

    # Temporary permissions until we finish the API updates.
    EDIT_ORG_SETTINGS = 'edit_org_settings'

    # Git organization claims
    MANAGE_ORG_CLAIMS = 'manage_org_claims'

    # Manage Automations
    MANAGE_AUTOMATIONS = 'manage_automations'

    # User provisioning (create new Keycloak/OpenHands users directly in an org)
    PROVISION_USER = 'provision_user'

    # Organization Conversations (Admin/Owner only)
    VIEW_ORG_CONVERSATIONS = 'view_org_conversations'


class RoleName(str, Enum):
    """Role names used in the system.

    The same role rows (``owner``, ``admin``, ``member``) can be
    assigned in two distinct places:

    * **Org-scoped role** -- stored on ``org_member.role_id``. Applies
      only to the user's membership in that organization. Permissions
      are looked up via :data:`ROLE_PERMISSIONS`.
    * **Super role** -- stored on ``user.role_id``. Applies at the
      instance level across organizations. The only functional super
      role currently defined is ``superadmin`` (``role_id`` pointing to
      the ``admin`` role row). Super permissions are defined explicitly
      in :data:`SUPER_ROLE_PERMISSIONS`; they do not inherit
      org-scoped role permissions.
    """

    OWNER = 'owner'
    ADMIN = 'admin'
    MEMBER = 'member'


def super_role_name(role_name: str) -> str:
    """Return the conceptual super-role label for a regular role name.

    For example, ``super_role_name('admin') == 'superadmin'``. Used only
    for logging / API responses -- super roles share the same database
    rows as their parallel regular roles.
    """
    return f'super{role_name}'


# Base permission mappings for the regular (org-scoped) roles.
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
            # Git organization claims
            Permission.MANAGE_ORG_CLAIMS,
            # Manage Automations
            Permission.MANAGE_AUTOMATIONS,
            # User provisioning
            Permission.PROVISION_USER,
            # Organization Conversations (Admin/Owner only)
            Permission.VIEW_ORG_CONVERSATIONS,
            # Integration provider setup (Admin/Owner only)
            Permission.MANAGE_INTEGRATION_PROVIDERS,
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
            # Git organization claims
            Permission.MANAGE_ORG_CLAIMS,
            # Manage Automations
            Permission.MANAGE_AUTOMATIONS,
            # User provisioning
            Permission.PROVISION_USER,
            # Organization Conversations (Admin/Owner only)
            Permission.VIEW_ORG_CONVERSATIONS,
            # Integration provider setup (Admin/Owner only)
            Permission.MANAGE_INTEGRATION_PROVIDERS,
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
            # Manage Automations
            Permission.MANAGE_AUTOMATIONS,
        ]
    ),
}


# Instance-level permissions for super roles. These are intentionally
# explicit and do not inherit the parallel org-scoped role permissions.
# This keeps instance administration separate from organization-scoped
# product/configuration administration.
SUPER_ROLE_PERMISSIONS: dict[RoleName, frozenset[Permission]] = {
    # Only superadmin is functional for now. It can create organizations
    # and provision users into a selected organization without becoming
    # an org member itself. Additional instance-admin capabilities should
    # be added here explicitly as the corresponding routes are wired to
    # permission checks.
    RoleName.OWNER: frozenset(),
    RoleName.ADMIN: frozenset(
        [Permission.CREATE_ORGANIZATION, Permission.PROVISION_USER]
    ),
    RoleName.MEMBER: frozenset(),
}


async def get_user_org_role(user_id: str, org_id: UUID | None) -> Role | None:
    """
    Get the user's role in an organization.

    Args:
        user_id: User ID (string that will be converted to UUID)
        org_id: Organization ID, or None to use the user's current organization

    Returns:
        Role object if user is a member, None otherwise
    """
    from uuid import UUID as parse_uuid

    if org_id is None:
        org_member = await OrgMemberStore.get_org_member_for_current_org(
            parse_uuid(user_id)
        )
    else:
        org_member = await OrgMemberStore.get_org_member(org_id, parse_uuid(user_id))
    if not org_member:
        return None

    return await RoleStore.get_role_by_id(org_member.role_id)


async def get_user_super_role(user_id: str) -> Role | None:
    """
    Get the user's cross-organization ("super") role.

    Super roles live on the ``user`` table (``user.role_id``) and apply to
    the user across **every** organization, in contrast to the
    organization-scoped role stored on ``org_member.role_id``.

    Args:
        user_id: User ID (string that will be converted to UUID)

    Returns:
        The :class:`Role` referenced by ``user.role_id`` if one is set,
        otherwise ``None``.
    """
    user = await UserStore.get_user_by_id(user_id)
    if not user or user.role_id is None:
        return None

    return await RoleStore.get_role_by_id(user.role_id)


def get_role_permissions(role_name: str) -> frozenset[Permission]:
    """
    Get the org-scoped permissions for a role.

    Args:
        role_name: Name of the role

    Returns:
        Set of permissions for the org-scoped role
    """
    try:
        role_enum = RoleName(role_name)
        return ROLE_PERMISSIONS.get(role_enum, frozenset())
    except ValueError:
        return frozenset()


def get_super_role_permissions(role_name: str) -> frozenset[Permission]:
    """
    Get the permissions for a "super" role.

    A super role is the same role row (``owner`` / ``admin`` / ``member``)
    referenced via ``user.role_id`` rather than ``org_member.role_id``;
    its effective permissions are defined explicitly and do not inherit
    org-scoped permissions.

    Args:
        role_name: Name of the role (e.g. ``'admin'`` for ``superadmin``)

    Returns:
        Set of permissions for the super role
    """
    try:
        role_enum = RoleName(role_name)
        return SUPER_ROLE_PERMISSIONS.get(role_enum, frozenset())
    except ValueError:
        return frozenset()


def has_permission(
    user_role: Role, permission: Permission, *, is_super: bool = False
) -> bool:
    """
    Check if a role has a specific permission.

    Args:
        user_role: User's Role object
        permission: Permission to check
        is_super: If ``True`` the role was looked up from ``user.role_id``
            and should be treated as a "super" role -- i.e. permissions
            are read from the explicit :data:`SUPER_ROLE_PERMISSIONS` mapping
            instead of :data:`ROLE_PERMISSIONS`.

    Returns:
        True if the role has the permission
    """
    if is_super:
        return permission in get_super_role_permissions(user_role.name)
    return permission in get_role_permissions(user_role.name)


async def get_api_key_org_id_from_request(request: Request) -> UUID | None:
    """Get the org_id bound to the API key used for authentication.

    Returns None if:
    - Not authenticated via API key (cookie auth), or
    - The API key is *unbound* -- the request's effective org is then
      resolved per-request via the ``X-Org-Id`` header or the caller's
      ``user.current_org_id`` (see ``SaasUserAuth._resolve_org_id``).
    """
    user_auth = getattr(request.state, 'user_auth', None)
    if user_auth and hasattr(user_auth, 'get_api_key_org_id'):
        return user_auth.get_api_key_org_id()
    return None


def require_permission(permission: Permission):
    """
    Factory function that creates a dependency to require a specific permission.

    This creates a FastAPI dependency that:
    1. Extracts org_id from the path parameter
    2. Gets the authenticated user_id
    3. Validates API key org binding (if using API key auth)
    4. Checks if the user has the required permission in the organization
    5. Returns the user_id if authorized, raises HTTPException otherwise

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
        request: Request,
        org_id: UUID | None = None,
        user_id: str | None = Depends(get_user_id),
    ) -> str:
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail='User not authenticated',
            )

        # Validate API key organization binding
        api_key_org_id = await get_api_key_org_id_from_request(request)
        if api_key_org_id is not None and org_id is not None:
            if api_key_org_id != org_id:
                logger.warning(
                    'API key organization mismatch',
                    extra={
                        'user_id': user_id,
                        'api_key_org_id': str(api_key_org_id),
                        'target_org_id': str(org_id),
                    },
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail='API key is not authorized for this organization',
                )

        # If the route does not carry an ``{org_id}`` path parameter,
        # resolve the target org for the request -- honoring any
        # ``X-Org-Id`` header override and the API-key binding, but
        # *without* requiring the user to already be a member of that
        # org. Membership is checked below via ``get_user_org_role``;
        # if the user is not a member but has a super role with the
        # required permission, they are still granted access. Routes
        # that *do* carry ``{org_id}`` use the path value directly and
        # skip this resolver, so their semantics are unchanged.
        if org_id is None:
            # Local import to avoid circular import via saas_user_auth.
            from server.auth.org_context import (
                resolve_target_org_id_for_permission_check,
            )

            org_id = await resolve_target_org_id_for_permission_check(request)

        user_role = await get_user_org_role(user_id, org_id)

        # 1. Org-scoped role check (existing behavior).
        if user_role and has_permission(user_role, permission):
            return user_id

        # 2. Fall back to the user's "super" role. The role row is the
        #    same as a regular role, but because it is attached via
        #    ``user.role_id`` it grants the super-role permission set
        #    (parallel role perms + super-only extras) across every
        #    organization the user touches.
        super_role = await get_user_super_role(user_id)
        if super_role and has_permission(super_role, permission, is_super=True):
            logger.debug(
                'Permission granted via super role',
                extra={
                    'user_id': user_id,
                    'org_id': str(org_id),
                    'super_role': super_role_name(super_role.name),
                    'required_permission': permission.value,
                },
            )
            return user_id

        # 3. Neither path granted access -- deny.
        if not user_role:
            logger.warning(
                'User not a member of organization and lacks super-role permission',
                extra={
                    'user_id': user_id,
                    'org_id': str(org_id),
                    'super_role': (
                        super_role_name(super_role.name) if super_role else None
                    ),
                    'required_permission': permission.value,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail='User is not a member of this organization',
            )

        logger.warning(
            'Insufficient permissions',
            extra={
                'user_id': user_id,
                'org_id': str(org_id),
                'user_role': user_role.name,
                'super_role': (
                    super_role_name(super_role.name) if super_role else None
                ),
                'required_permission': permission.value,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f'Requires {permission.value} permission',
        )

    return permission_checker


async def require_financial_data_access(
    request: Request,
    org_id: UUID,
    user_id: str | None = Depends(get_user_id),
) -> str:
    """
    Authorization dependency for accessing organization financial data.

    Allows access if ANY of these conditions are met:
    1. User has Admin or Owner role in the organization
    2. User has @openhands.dev email domain

    This is used for the organization members financial data endpoint.

    Args:
        request: FastAPI request object
        org_id: Organization UUID from path parameter
        user_id: User ID from authentication

    Returns:
        str: User ID if authorized

    Raises:
        HTTPException: 401 if not authenticated, 403 if not authorized
    """
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='User not authenticated',
        )

    # Validate API key organization binding
    api_key_org_id = await get_api_key_org_id_from_request(request)
    if api_key_org_id is not None:
        if api_key_org_id != org_id:
            logger.warning(
                'API key organization mismatch for financial data access',
                extra={
                    'user_id': user_id,
                    'api_key_org_id': str(api_key_org_id),
                    'target_org_id': str(org_id),
                },
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail='API key is not authorized for this organization',
            )

    # Check if user has @openhands.dev email
    user_auth = await get_user_auth(request)
    user_email = await user_auth.get_user_email()

    if user_email and user_email.endswith('@openhands.dev'):
        logger.debug(
            'Financial data access granted via @openhands.dev email',
            extra={'user_id': user_id, 'org_id': str(org_id)},
        )
        return user_id

    # Check if user has Admin or Owner role in the organization
    user_role = await get_user_org_role(user_id, org_id)

    if not user_role:
        logger.warning(
            'Financial data access denied - user not a member of organization',
            extra={'user_id': user_id, 'org_id': str(org_id)},
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='User is not a member of this organization',
        )

    if user_role.name not in (RoleName.OWNER.value, RoleName.ADMIN.value):
        logger.warning(
            'Financial data access denied - insufficient role',
            extra={
                'user_id': user_id,
                'org_id': str(org_id),
                'user_role': user_role.name,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='Access restricted to organization admins, owners, or OpenHands members',
        )

    logger.debug(
        'Financial data access granted via admin/owner role',
        extra={'user_id': user_id, 'org_id': str(org_id), 'role': user_role.name},
    )
    return user_id
