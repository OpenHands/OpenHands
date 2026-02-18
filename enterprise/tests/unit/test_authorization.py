"""
Unit tests for permission-based authorization (authorization.py).

Tests the FastAPI dependencies that validate user permissions within organizations.
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException
from server.auth.authorization import (
    ROLE_PERMISSIONS,
    Permission,
    RoleName,
    get_role_permissions,
    get_user_org_role,
    has_permission,
    require_permission,
)

# =============================================================================
# Tests for Permission enum
# =============================================================================


class TestPermission:
    """Tests for Permission enum."""

    def test_permission_values(self):
        """
        GIVEN: Permission enum
        WHEN: Accessing permission values
        THEN: All expected permissions exist with correct string values
        """
        assert Permission.MANAGE_SECRETS.value == 'manage_secrets'
        assert Permission.MANAGE_MCP.value == 'manage_mcp'
        assert Permission.MANAGE_INTEGRATIONS.value == 'manage_integrations'
        assert (
            Permission.MANAGE_APPLICATION_SETTINGS.value
            == 'manage_application_settings'
        )
        assert Permission.MANAGE_API_KEYS.value == 'manage_api_keys'
        assert Permission.VIEW_LLM_SETTINGS.value == 'view_llm_settings'
        assert Permission.EDIT_LLM_SETTINGS.value == 'edit_llm_settings'
        assert Permission.VIEW_BILLING.value == 'view_billing'
        assert Permission.ADD_CREDITS.value == 'add_credits'
        assert (
            Permission.INVITE_USER_TO_ORGANIZATION.value
            == 'invite_user_to_organization'
        )
        assert Permission.CHANGE_USER_ROLE_MEMBER.value == 'change_user_role:member'
        assert Permission.CHANGE_USER_ROLE_ADMIN.value == 'change_user_role:admin'
        assert Permission.CHANGE_USER_ROLE_OWNER.value == 'change_user_role:owner'
        assert Permission.VIEW_ORG_SETTINGS.value == 'view_org_settings'
        assert Permission.CHANGE_ORGANIZATION_NAME.value == 'change_organization_name'
        assert Permission.DELETE_ORGANIZATION.value == 'delete_organization'

    def test_permission_from_string(self):
        """
        GIVEN: Valid permission string
        WHEN: Creating Permission from string
        THEN: Correct enum value is returned
        """
        assert Permission('manage_secrets') == Permission.MANAGE_SECRETS
        assert Permission('view_llm_settings') == Permission.VIEW_LLM_SETTINGS
        assert Permission('delete_organization') == Permission.DELETE_ORGANIZATION

    def test_permission_invalid_string(self):
        """
        GIVEN: Invalid permission string
        WHEN: Creating Permission from string
        THEN: ValueError is raised
        """
        with pytest.raises(ValueError):
            Permission('invalid_permission')


# =============================================================================
# Tests for RoleName enum
# =============================================================================


class TestRoleName:
    """Tests for RoleName enum."""

    def test_role_name_values(self):
        """
        GIVEN: RoleName enum
        WHEN: Accessing role name values
        THEN: All expected roles exist with correct string values
        """
        assert RoleName.OWNER.value == 'owner'
        assert RoleName.ADMIN.value == 'admin'
        assert RoleName.MEMBER.value == 'member'

    def test_role_name_from_string(self):
        """
        GIVEN: Valid role name string
        WHEN: Creating RoleName from string
        THEN: Correct enum value is returned
        """
        assert RoleName('owner') == RoleName.OWNER
        assert RoleName('admin') == RoleName.ADMIN
        assert RoleName('member') == RoleName.MEMBER

    def test_role_name_invalid_string(self):
        """
        GIVEN: Invalid role name string
        WHEN: Creating RoleName from string
        THEN: ValueError is raised
        """
        with pytest.raises(ValueError):
            RoleName('invalid_role')


# =============================================================================
# Tests for ROLE_PERMISSIONS mapping
# =============================================================================


class TestRolePermissions:
    """Tests for role permission mappings."""

    def test_owner_has_all_permissions(self):
        """
        GIVEN: ROLE_PERMISSIONS mapping
        WHEN: Checking owner permissions
        THEN: Owner has all permissions including owner-only permissions
        """
        owner_perms = ROLE_PERMISSIONS[RoleName.OWNER]
        assert Permission.MANAGE_SECRETS in owner_perms
        assert Permission.MANAGE_MCP in owner_perms
        assert Permission.VIEW_LLM_SETTINGS in owner_perms
        assert Permission.EDIT_LLM_SETTINGS in owner_perms
        assert Permission.VIEW_BILLING in owner_perms
        assert Permission.ADD_CREDITS in owner_perms
        assert Permission.INVITE_USER_TO_ORGANIZATION in owner_perms
        assert Permission.CHANGE_USER_ROLE_MEMBER in owner_perms
        assert Permission.CHANGE_USER_ROLE_ADMIN in owner_perms
        assert Permission.CHANGE_USER_ROLE_OWNER in owner_perms
        assert Permission.CHANGE_ORGANIZATION_NAME in owner_perms
        assert Permission.DELETE_ORGANIZATION in owner_perms

    def test_admin_has_admin_permissions(self):
        """
        GIVEN: ROLE_PERMISSIONS mapping
        WHEN: Checking admin permissions
        THEN: Admin has admin permissions but not owner-only permissions
        """
        admin_perms = ROLE_PERMISSIONS[RoleName.ADMIN]
        assert Permission.MANAGE_SECRETS in admin_perms
        assert Permission.MANAGE_MCP in admin_perms
        assert Permission.VIEW_LLM_SETTINGS in admin_perms
        assert Permission.EDIT_LLM_SETTINGS in admin_perms
        assert Permission.VIEW_BILLING in admin_perms
        assert Permission.ADD_CREDITS in admin_perms
        assert Permission.INVITE_USER_TO_ORGANIZATION in admin_perms
        assert Permission.CHANGE_USER_ROLE_MEMBER in admin_perms
        assert Permission.CHANGE_USER_ROLE_ADMIN in admin_perms
        # Admin should NOT have owner-only permissions
        assert Permission.CHANGE_USER_ROLE_OWNER not in admin_perms
        assert Permission.CHANGE_ORGANIZATION_NAME not in admin_perms
        assert Permission.DELETE_ORGANIZATION not in admin_perms

    def test_member_has_limited_permissions(self):
        """
        GIVEN: ROLE_PERMISSIONS mapping
        WHEN: Checking member permissions
        THEN: Member has limited permissions
        """
        member_perms = ROLE_PERMISSIONS[RoleName.MEMBER]
        # Member has basic settings permissions
        assert Permission.MANAGE_SECRETS in member_perms
        assert Permission.MANAGE_MCP in member_perms
        assert Permission.MANAGE_INTEGRATIONS in member_perms
        assert Permission.MANAGE_APPLICATION_SETTINGS in member_perms
        assert Permission.MANAGE_API_KEYS in member_perms
        assert Permission.VIEW_LLM_SETTINGS in member_perms
        assert Permission.VIEW_ORG_SETTINGS in member_perms
        # Member should NOT have admin/owner permissions
        assert Permission.EDIT_LLM_SETTINGS not in member_perms
        assert Permission.VIEW_BILLING not in member_perms
        assert Permission.ADD_CREDITS not in member_perms
        assert Permission.INVITE_USER_TO_ORGANIZATION not in member_perms
        assert Permission.CHANGE_USER_ROLE_MEMBER not in member_perms
        assert Permission.CHANGE_USER_ROLE_ADMIN not in member_perms
        assert Permission.CHANGE_USER_ROLE_OWNER not in member_perms
        assert Permission.CHANGE_ORGANIZATION_NAME not in member_perms
        assert Permission.DELETE_ORGANIZATION not in member_perms


# =============================================================================
# Tests for get_role_permissions function
# =============================================================================


class TestGetRolePermissions:
    """Tests for get_role_permissions function."""

    def test_get_owner_permissions(self):
        """
        GIVEN: Role name 'owner'
        WHEN: get_role_permissions is called
        THEN: Owner permissions are returned
        """
        perms = get_role_permissions('owner')
        assert Permission.DELETE_ORGANIZATION in perms
        assert Permission.CHANGE_ORGANIZATION_NAME in perms

    def test_get_admin_permissions(self):
        """
        GIVEN: Role name 'admin'
        WHEN: get_role_permissions is called
        THEN: Admin permissions are returned
        """
        perms = get_role_permissions('admin')
        assert Permission.EDIT_LLM_SETTINGS in perms
        assert Permission.DELETE_ORGANIZATION not in perms

    def test_get_member_permissions(self):
        """
        GIVEN: Role name 'member'
        WHEN: get_role_permissions is called
        THEN: Member permissions are returned
        """
        perms = get_role_permissions('member')
        assert Permission.VIEW_LLM_SETTINGS in perms
        assert Permission.EDIT_LLM_SETTINGS not in perms

    def test_get_invalid_role_permissions(self):
        """
        GIVEN: Invalid role name
        WHEN: get_role_permissions is called
        THEN: Empty frozenset is returned
        """
        perms = get_role_permissions('invalid_role')
        assert perms == frozenset()


# =============================================================================
# Tests for has_permission function
# =============================================================================


class TestHasPermission:
    """Tests for has_permission function."""

    def test_owner_has_delete_organization_permission(self):
        """
        GIVEN: User with owner role
        WHEN: Checking for DELETE_ORGANIZATION permission
        THEN: Returns True
        """
        mock_role = MagicMock()
        mock_role.name = 'owner'
        assert has_permission(mock_role, Permission.DELETE_ORGANIZATION) is True

    def test_owner_has_view_llm_settings_permission(self):
        """
        GIVEN: User with owner role
        WHEN: Checking for VIEW_LLM_SETTINGS permission
        THEN: Returns True
        """
        mock_role = MagicMock()
        mock_role.name = 'owner'
        assert has_permission(mock_role, Permission.VIEW_LLM_SETTINGS) is True

    def test_admin_has_edit_llm_settings_permission(self):
        """
        GIVEN: User with admin role
        WHEN: Checking for EDIT_LLM_SETTINGS permission
        THEN: Returns True
        """
        mock_role = MagicMock()
        mock_role.name = 'admin'
        assert has_permission(mock_role, Permission.EDIT_LLM_SETTINGS) is True

    def test_admin_lacks_delete_organization_permission(self):
        """
        GIVEN: User with admin role
        WHEN: Checking for DELETE_ORGANIZATION permission
        THEN: Returns False
        """
        mock_role = MagicMock()
        mock_role.name = 'admin'
        assert has_permission(mock_role, Permission.DELETE_ORGANIZATION) is False

    def test_member_has_view_llm_settings_permission(self):
        """
        GIVEN: User with member role
        WHEN: Checking for VIEW_LLM_SETTINGS permission
        THEN: Returns True
        """
        mock_role = MagicMock()
        mock_role.name = 'member'
        assert has_permission(mock_role, Permission.VIEW_LLM_SETTINGS) is True

    def test_member_lacks_edit_llm_settings_permission(self):
        """
        GIVEN: User with member role
        WHEN: Checking for EDIT_LLM_SETTINGS permission
        THEN: Returns False
        """
        mock_role = MagicMock()
        mock_role.name = 'member'
        assert has_permission(mock_role, Permission.EDIT_LLM_SETTINGS) is False

    def test_member_lacks_delete_organization_permission(self):
        """
        GIVEN: User with member role
        WHEN: Checking for DELETE_ORGANIZATION permission
        THEN: Returns False
        """
        mock_role = MagicMock()
        mock_role.name = 'member'
        assert has_permission(mock_role, Permission.DELETE_ORGANIZATION) is False

    def test_invalid_role_has_no_permissions(self):
        """
        GIVEN: User with invalid role
        WHEN: Checking for any permission
        THEN: Returns False
        """
        mock_role = MagicMock()
        mock_role.name = 'invalid_role'
        assert has_permission(mock_role, Permission.VIEW_LLM_SETTINGS) is False
        assert has_permission(mock_role, Permission.DELETE_ORGANIZATION) is False


# =============================================================================
# Tests for get_user_org_role function
# =============================================================================


class TestGetUserOrgRole:
    """Tests for get_user_org_role function."""

    def test_returns_role_when_member_exists(self):
        """
        GIVEN: User is a member of organization with role
        WHEN: get_user_org_role is called
        THEN: Role object is returned
        """
        user_id = str(uuid4())
        org_id = uuid4()

        mock_org_member = MagicMock()
        mock_org_member.role_id = 1

        mock_role = MagicMock()
        mock_role.name = 'admin'

        with (
            patch(
                'server.auth.authorization.OrgMemberStore.get_org_member',
                return_value=mock_org_member,
            ),
            patch(
                'server.auth.authorization.RoleStore.get_role_by_id',
                return_value=mock_role,
            ),
        ):
            result = get_user_org_role(user_id, org_id)
            assert result == mock_role

    def test_returns_none_when_not_member(self):
        """
        GIVEN: User is not a member of organization
        WHEN: get_user_org_role is called
        THEN: None is returned
        """
        user_id = str(uuid4())
        org_id = uuid4()

        with patch(
            'server.auth.authorization.OrgMemberStore.get_org_member',
            return_value=None,
        ):
            result = get_user_org_role(user_id, org_id)
            assert result is None

    def test_returns_role_when_org_id_is_none(self):
        """
        GIVEN: User with a current organization
        WHEN: get_user_org_role is called with org_id=None
        THEN: Role object is returned using get_org_member_for_current_org
        """
        user_id = str(uuid4())

        mock_org_member = MagicMock()
        mock_org_member.role_id = 1

        mock_role = MagicMock()
        mock_role.name = 'admin'

        with (
            patch(
                'server.auth.authorization.OrgMemberStore.get_org_member_for_current_org',
                return_value=mock_org_member,
            ) as mock_get_current,
            patch(
                'server.auth.authorization.OrgMemberStore.get_org_member',
            ) as mock_get_org_member,
            patch(
                'server.auth.authorization.RoleStore.get_role_by_id',
                return_value=mock_role,
            ),
        ):
            result = get_user_org_role(user_id, None)
            assert result == mock_role
            mock_get_current.assert_called_once()
            mock_get_org_member.assert_not_called()

    def test_returns_none_when_org_id_is_none_and_no_current_org(self):
        """
        GIVEN: User with no current organization membership
        WHEN: get_user_org_role is called with org_id=None
        THEN: None is returned
        """
        user_id = str(uuid4())

        with patch(
            'server.auth.authorization.OrgMemberStore.get_org_member_for_current_org',
            return_value=None,
        ):
            result = get_user_org_role(user_id, None)
            assert result is None


# =============================================================================
# Tests for require_permission dependency
# =============================================================================


class TestRequirePermission:
    """Tests for require_permission dependency factory."""

    @pytest.mark.asyncio
    async def test_returns_user_id_when_authorized(self):
        """
        GIVEN: User with required permission
        WHEN: Permission checker is called
        THEN: User ID is returned
        """
        user_id = str(uuid4())
        org_id = uuid4()

        mock_role = MagicMock()
        mock_role.name = 'admin'

        with patch(
            'server.auth.authorization.get_user_org_role_async',
            AsyncMock(return_value=mock_role),
        ):
            permission_checker = require_permission(Permission.VIEW_LLM_SETTINGS)
            result = await permission_checker(org_id=org_id, user_id=user_id)
            assert result == user_id

    @pytest.mark.asyncio
    async def test_raises_401_when_not_authenticated(self):
        """
        GIVEN: No user ID (not authenticated)
        WHEN: Permission checker is called
        THEN: 401 Unauthorized is raised
        """
        org_id = uuid4()

        permission_checker = require_permission(Permission.VIEW_LLM_SETTINGS)
        with pytest.raises(HTTPException) as exc_info:
            await permission_checker(org_id=org_id, user_id=None)

        assert exc_info.value.status_code == 401
        assert 'not authenticated' in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_raises_403_when_not_member(self):
        """
        GIVEN: User is not a member of organization
        WHEN: Permission checker is called
        THEN: 403 Forbidden is raised
        """
        user_id = str(uuid4())
        org_id = uuid4()

        with patch(
            'server.auth.authorization.get_user_org_role_async',
            AsyncMock(return_value=None),
        ):
            permission_checker = require_permission(Permission.VIEW_LLM_SETTINGS)
            with pytest.raises(HTTPException) as exc_info:
                await permission_checker(org_id=org_id, user_id=user_id)

            assert exc_info.value.status_code == 403
            assert 'not a member' in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_raises_403_when_insufficient_permission(self):
        """
        GIVEN: User without required permission
        WHEN: Permission checker is called
        THEN: 403 Forbidden is raised
        """
        user_id = str(uuid4())
        org_id = uuid4()

        mock_role = MagicMock()
        mock_role.name = 'member'

        with patch(
            'server.auth.authorization.get_user_org_role_async',
            AsyncMock(return_value=mock_role),
        ):
            permission_checker = require_permission(Permission.DELETE_ORGANIZATION)
            with pytest.raises(HTTPException) as exc_info:
                await permission_checker(org_id=org_id, user_id=user_id)

            assert exc_info.value.status_code == 403
            assert 'delete_organization' in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_owner_can_delete_organization(self):
        """
        GIVEN: User with owner role
        WHEN: DELETE_ORGANIZATION permission is required
        THEN: User ID is returned
        """
        user_id = str(uuid4())
        org_id = uuid4()

        mock_role = MagicMock()
        mock_role.name = 'owner'

        with patch(
            'server.auth.authorization.get_user_org_role_async',
            AsyncMock(return_value=mock_role),
        ):
            permission_checker = require_permission(Permission.DELETE_ORGANIZATION)
            result = await permission_checker(org_id=org_id, user_id=user_id)
            assert result == user_id

    @pytest.mark.asyncio
    async def test_admin_cannot_delete_organization(self):
        """
        GIVEN: User with admin role
        WHEN: DELETE_ORGANIZATION permission is required
        THEN: 403 Forbidden is raised
        """
        user_id = str(uuid4())
        org_id = uuid4()

        mock_role = MagicMock()
        mock_role.name = 'admin'

        with patch(
            'server.auth.authorization.get_user_org_role_async',
            AsyncMock(return_value=mock_role),
        ):
            permission_checker = require_permission(Permission.DELETE_ORGANIZATION)
            with pytest.raises(HTTPException) as exc_info:
                await permission_checker(org_id=org_id, user_id=user_id)

            assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_logs_warning_on_insufficient_permission(self):
        """
        GIVEN: User without required permission
        WHEN: Permission checker is called
        THEN: Warning is logged with details
        """
        user_id = str(uuid4())
        org_id = uuid4()

        mock_role = MagicMock()
        mock_role.name = 'member'

        with (
            patch(
                'server.auth.authorization.get_user_org_role_async',
                AsyncMock(return_value=mock_role),
            ),
            patch('server.auth.authorization.logger') as mock_logger,
        ):
            permission_checker = require_permission(Permission.DELETE_ORGANIZATION)
            with pytest.raises(HTTPException):
                await permission_checker(org_id=org_id, user_id=user_id)

            mock_logger.warning.assert_called()
            call_args = mock_logger.warning.call_args
            assert call_args[1]['extra']['user_id'] == user_id
            assert call_args[1]['extra']['user_role'] == 'member'
            assert call_args[1]['extra']['required_permission'] == 'delete_organization'

    @pytest.mark.asyncio
    async def test_returns_user_id_when_org_id_is_none(self):
        """
        GIVEN: User with required permission in their current org
        WHEN: Permission checker is called with org_id=None
        THEN: User ID is returned
        """
        user_id = str(uuid4())

        mock_role = MagicMock()
        mock_role.name = 'admin'

        with patch(
            'server.auth.authorization.get_user_org_role_async',
            AsyncMock(return_value=mock_role),
        ) as mock_get_role:
            permission_checker = require_permission(Permission.VIEW_LLM_SETTINGS)
            result = await permission_checker(org_id=None, user_id=user_id)
            assert result == user_id
            mock_get_role.assert_called_once_with(user_id, None)

    @pytest.mark.asyncio
    async def test_raises_403_when_org_id_is_none_and_not_member(self):
        """
        GIVEN: User not a member of their current organization
        WHEN: Permission checker is called with org_id=None
        THEN: HTTPException with 403 status is raised
        """
        user_id = str(uuid4())

        with patch(
            'server.auth.authorization.get_user_org_role_async',
            AsyncMock(return_value=None),
        ):
            permission_checker = require_permission(Permission.VIEW_LLM_SETTINGS)
            with pytest.raises(HTTPException) as exc_info:
                await permission_checker(org_id=None, user_id=user_id)

            assert exc_info.value.status_code == 403
            assert 'not a member' in exc_info.value.detail


# =============================================================================
# Tests for permission-based access control scenarios
# =============================================================================


class TestPermissionScenarios:
    """Tests for real-world permission scenarios."""

    @pytest.mark.asyncio
    async def test_member_can_manage_secrets(self):
        """
        GIVEN: User with member role
        WHEN: MANAGE_SECRETS permission is required
        THEN: User ID is returned
        """
        user_id = str(uuid4())
        org_id = uuid4()

        mock_role = MagicMock()
        mock_role.name = 'member'

        with patch(
            'server.auth.authorization.get_user_org_role_async',
            AsyncMock(return_value=mock_role),
        ):
            permission_checker = require_permission(Permission.MANAGE_SECRETS)
            result = await permission_checker(org_id=org_id, user_id=user_id)
            assert result == user_id

    @pytest.mark.asyncio
    async def test_member_cannot_invite_users(self):
        """
        GIVEN: User with member role
        WHEN: INVITE_USER_TO_ORGANIZATION permission is required
        THEN: 403 Forbidden is raised
        """
        user_id = str(uuid4())
        org_id = uuid4()

        mock_role = MagicMock()
        mock_role.name = 'member'

        with patch(
            'server.auth.authorization.get_user_org_role_async',
            AsyncMock(return_value=mock_role),
        ):
            permission_checker = require_permission(
                Permission.INVITE_USER_TO_ORGANIZATION
            )
            with pytest.raises(HTTPException) as exc_info:
                await permission_checker(org_id=org_id, user_id=user_id)

            assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_admin_can_invite_users(self):
        """
        GIVEN: User with admin role
        WHEN: INVITE_USER_TO_ORGANIZATION permission is required
        THEN: User ID is returned
        """
        user_id = str(uuid4())
        org_id = uuid4()

        mock_role = MagicMock()
        mock_role.name = 'admin'

        with patch(
            'server.auth.authorization.get_user_org_role_async',
            AsyncMock(return_value=mock_role),
        ):
            permission_checker = require_permission(
                Permission.INVITE_USER_TO_ORGANIZATION
            )
            result = await permission_checker(org_id=org_id, user_id=user_id)
            assert result == user_id

    @pytest.mark.asyncio
    async def test_admin_cannot_change_owner_role(self):
        """
        GIVEN: User with admin role
        WHEN: CHANGE_USER_ROLE_OWNER permission is required
        THEN: 403 Forbidden is raised
        """
        user_id = str(uuid4())
        org_id = uuid4()

        mock_role = MagicMock()
        mock_role.name = 'admin'

        with patch(
            'server.auth.authorization.get_user_org_role_async',
            AsyncMock(return_value=mock_role),
        ):
            permission_checker = require_permission(Permission.CHANGE_USER_ROLE_OWNER)
            with pytest.raises(HTTPException) as exc_info:
                await permission_checker(org_id=org_id, user_id=user_id)

            assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_owner_can_change_owner_role(self):
        """
        GIVEN: User with owner role
        WHEN: CHANGE_USER_ROLE_OWNER permission is required
        THEN: User ID is returned
        """
        user_id = str(uuid4())
        org_id = uuid4()

        mock_role = MagicMock()
        mock_role.name = 'owner'

        with patch(
            'server.auth.authorization.get_user_org_role_async',
            AsyncMock(return_value=mock_role),
        ):
            permission_checker = require_permission(Permission.CHANGE_USER_ROLE_OWNER)
            result = await permission_checker(org_id=org_id, user_id=user_id)
            assert result == user_id
