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
    SUPER_ROLE_PERMISSIONS,
    Permission,
    RoleName,
    get_api_key_org_id_from_request,
    get_role_permissions,
    get_super_role_permissions,
    get_user_org_role,
    get_user_super_role,
    has_permission,
    require_permission,
    super_role_name,
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
            Permission.MANAGE_INTEGRATION_PROVIDERS.value
            == 'manage_integration_providers'
        )
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
        assert Permission.CREATE_ORGANIZATION.value == 'create_organization'
        assert Permission.MANAGE_AUTOMATIONS.value == 'manage_automations'
        assert Permission.VIEW_ORG_CONVERSATIONS.value == 'view_org_conversations'

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
        assert Permission.VIEW_ORG_SETTINGS in owner_perms
        assert Permission.EDIT_ORG_SETTINGS in owner_perms
        assert Permission.VIEW_BILLING in owner_perms
        assert Permission.ADD_CREDITS in owner_perms
        assert Permission.INVITE_USER_TO_ORGANIZATION in owner_perms
        assert Permission.CHANGE_USER_ROLE_MEMBER in owner_perms
        assert Permission.CHANGE_USER_ROLE_ADMIN in owner_perms
        assert Permission.CHANGE_USER_ROLE_OWNER in owner_perms
        assert Permission.CHANGE_ORGANIZATION_NAME in owner_perms
        assert Permission.DELETE_ORGANIZATION in owner_perms
        assert Permission.MANAGE_AUTOMATIONS in owner_perms
        assert Permission.VIEW_ORG_CONVERSATIONS in owner_perms
        assert Permission.MANAGE_INTEGRATION_PROVIDERS in owner_perms

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
        assert Permission.VIEW_ORG_SETTINGS in admin_perms
        assert Permission.EDIT_ORG_SETTINGS in admin_perms
        assert Permission.VIEW_BILLING in admin_perms
        assert Permission.ADD_CREDITS in admin_perms
        assert Permission.INVITE_USER_TO_ORGANIZATION in admin_perms
        assert Permission.CHANGE_USER_ROLE_MEMBER in admin_perms
        assert Permission.CHANGE_USER_ROLE_ADMIN in admin_perms
        assert Permission.MANAGE_AUTOMATIONS in admin_perms
        assert Permission.VIEW_ORG_CONVERSATIONS in admin_perms
        assert Permission.MANAGE_INTEGRATION_PROVIDERS in admin_perms
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
        assert Permission.MANAGE_AUTOMATIONS in member_perms
        assert Permission.VIEW_LLM_SETTINGS in member_perms
        assert Permission.VIEW_ORG_SETTINGS in member_perms
        # Member should NOT have admin/owner permissions
        assert Permission.EDIT_LLM_SETTINGS not in member_perms
        assert Permission.EDIT_ORG_SETTINGS not in member_perms
        assert Permission.MANAGE_INTEGRATION_PROVIDERS not in member_perms
        assert Permission.VIEW_BILLING not in member_perms
        assert Permission.ADD_CREDITS not in member_perms
        assert Permission.INVITE_USER_TO_ORGANIZATION not in member_perms
        assert Permission.CHANGE_USER_ROLE_MEMBER not in member_perms
        assert Permission.CHANGE_USER_ROLE_ADMIN not in member_perms
        assert Permission.CHANGE_USER_ROLE_OWNER not in member_perms
        assert Permission.CHANGE_ORGANIZATION_NAME not in member_perms
        assert Permission.DELETE_ORGANIZATION not in member_perms
        assert Permission.VIEW_ORG_CONVERSATIONS not in member_perms

    def test_create_organization_is_not_org_scoped_for_any_role(self):
        """
        GIVEN: ROLE_PERMISSIONS mapping
        WHEN: Checking CREATE_ORGANIZATION across regular roles
        THEN: No regular org-scoped role grants CREATE_ORGANIZATION
              (it is a super-only permission).
        """
        for role_name, perms in ROLE_PERMISSIONS.items():
            assert Permission.CREATE_ORGANIZATION not in perms, (
                f'{role_name.value} unexpectedly grants CREATE_ORGANIZATION '
                'at the org-scoped level'
            )


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

    @pytest.mark.asyncio
    async def test_returns_role_when_member_exists(self):
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
                new_callable=AsyncMock,
                return_value=mock_org_member,
            ),
            patch(
                'server.auth.authorization.RoleStore.get_role_by_id',
                new_callable=AsyncMock,
                return_value=mock_role,
            ),
        ):
            result = await get_user_org_role(user_id, org_id)
            assert result == mock_role

    @pytest.mark.asyncio
    async def test_returns_none_when_not_member(self):
        """
        GIVEN: User is not a member of organization
        WHEN: get_user_org_role is called
        THEN: None is returned
        """
        user_id = str(uuid4())
        org_id = uuid4()

        with patch(
            'server.auth.authorization.OrgMemberStore.get_org_member',
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await get_user_org_role(user_id, org_id)
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_role_when_org_id_is_none(self):
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
                new_callable=AsyncMock,
                return_value=mock_org_member,
            ) as mock_get_current,
            patch(
                'server.auth.authorization.OrgMemberStore.get_org_member',
                new_callable=AsyncMock,
            ) as mock_get_org_member,
            patch(
                'server.auth.authorization.RoleStore.get_role_by_id',
                new_callable=AsyncMock,
                return_value=mock_role,
            ),
        ):
            result = await get_user_org_role(user_id, None)
            assert result == mock_role
            mock_get_current.assert_called_once()
            mock_get_org_member.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_none_when_org_id_is_none_and_no_current_org(self):
        """
        GIVEN: User with no current organization membership
        WHEN: get_user_org_role is called with org_id=None
        THEN: None is returned
        """
        user_id = str(uuid4())

        with patch(
            'server.auth.authorization.OrgMemberStore.get_org_member_for_current_org',
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await get_user_org_role(user_id, None)
            assert result is None


# =============================================================================
# Tests for require_permission dependency
# =============================================================================


def _create_mock_request(api_key_org_id=None):
    """Helper to create a mock request with optional api_key_org_id."""
    mock_request = MagicMock()
    mock_user_auth = MagicMock()
    mock_user_auth.get_api_key_org_id.return_value = api_key_org_id
    mock_request.state.user_auth = mock_user_auth
    return mock_request


@pytest.fixture
def _no_super_role():
    """Default ``get_user_super_role`` to ``None`` for tests that don't
    care about the super-role fallback. Individual tests can still
    re-patch ``server.auth.authorization.get_user_super_role`` inside
    their own ``with patch(...)`` block to override this default.
    """
    with patch(
        'server.auth.authorization.get_user_super_role',
        AsyncMock(return_value=None),
    ) as mocked:
        yield mocked


@pytest.mark.usefixtures('_no_super_role')
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
        mock_request = _create_mock_request()

        mock_role = MagicMock()
        mock_role.name = 'admin'

        with patch(
            'server.auth.authorization.get_user_org_role',
            AsyncMock(return_value=mock_role),
        ):
            permission_checker = require_permission(Permission.VIEW_LLM_SETTINGS)
            result = await permission_checker(
                request=mock_request, org_id=org_id, user_id=user_id
            )
            assert result == user_id

    @pytest.mark.asyncio
    async def test_raises_401_when_not_authenticated(self):
        """
        GIVEN: No user ID (not authenticated)
        WHEN: Permission checker is called
        THEN: 401 Unauthorized is raised
        """
        org_id = uuid4()
        mock_request = _create_mock_request()

        permission_checker = require_permission(Permission.VIEW_LLM_SETTINGS)
        with pytest.raises(HTTPException) as exc_info:
            await permission_checker(request=mock_request, org_id=org_id, user_id=None)

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
        mock_request = _create_mock_request()

        with patch(
            'server.auth.authorization.get_user_org_role',
            AsyncMock(return_value=None),
        ):
            permission_checker = require_permission(Permission.VIEW_LLM_SETTINGS)
            with pytest.raises(HTTPException) as exc_info:
                await permission_checker(
                    request=mock_request, org_id=org_id, user_id=user_id
                )

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
        mock_request = _create_mock_request()

        mock_role = MagicMock()
        mock_role.name = 'member'

        with patch(
            'server.auth.authorization.get_user_org_role',
            AsyncMock(return_value=mock_role),
        ):
            permission_checker = require_permission(Permission.DELETE_ORGANIZATION)
            with pytest.raises(HTTPException) as exc_info:
                await permission_checker(
                    request=mock_request, org_id=org_id, user_id=user_id
                )

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
        mock_request = _create_mock_request()

        mock_role = MagicMock()
        mock_role.name = 'owner'

        with patch(
            'server.auth.authorization.get_user_org_role',
            AsyncMock(return_value=mock_role),
        ):
            permission_checker = require_permission(Permission.DELETE_ORGANIZATION)
            result = await permission_checker(
                request=mock_request, org_id=org_id, user_id=user_id
            )
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
        mock_request = _create_mock_request()

        mock_role = MagicMock()
        mock_role.name = 'admin'

        with patch(
            'server.auth.authorization.get_user_org_role',
            AsyncMock(return_value=mock_role),
        ):
            permission_checker = require_permission(Permission.DELETE_ORGANIZATION)
            with pytest.raises(HTTPException) as exc_info:
                await permission_checker(
                    request=mock_request, org_id=org_id, user_id=user_id
                )

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
        mock_request = _create_mock_request()

        mock_role = MagicMock()
        mock_role.name = 'member'

        with (
            patch(
                'server.auth.authorization.get_user_org_role',
                AsyncMock(return_value=mock_role),
            ),
            patch('server.auth.authorization.logger') as mock_logger,
        ):
            permission_checker = require_permission(Permission.DELETE_ORGANIZATION)
            with pytest.raises(HTTPException):
                await permission_checker(
                    request=mock_request, org_id=org_id, user_id=user_id
                )

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
        mock_request = _create_mock_request()

        mock_role = MagicMock()
        mock_role.name = 'admin'

        with patch(
            'server.auth.authorization.get_user_org_role',
            AsyncMock(return_value=mock_role),
        ) as mock_get_role:
            permission_checker = require_permission(Permission.VIEW_LLM_SETTINGS)
            result = await permission_checker(
                request=mock_request, org_id=None, user_id=user_id
            )
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
        mock_request = _create_mock_request()

        with patch(
            'server.auth.authorization.get_user_org_role',
            AsyncMock(return_value=None),
        ):
            permission_checker = require_permission(Permission.VIEW_LLM_SETTINGS)
            with pytest.raises(HTTPException) as exc_info:
                await permission_checker(
                    request=mock_request, org_id=None, user_id=user_id
                )

            assert exc_info.value.status_code == 403
            assert 'not a member' in exc_info.value.detail


@pytest.mark.usefixtures('_no_super_role')
class TestRequirePermissionQueryStringOrgIdBypass:
    """Regression tests for a cross-org privilege-escalation bypass.

    On routes with no ``{org_id}`` path segment (e.g. the flat
    ``EFFECTIVE_ORG_ID``-scoped routes), FastAPI has no path value to bind
    ``permission_checker``'s own ``org_id`` parameter to, so it silently
    falls back to binding it from an ordinary, client-controlled query
    string. Previously that query-string value was trusted directly,
    letting a caller supply ``?org_id=<org they administer>`` to pass the
    permission check while the route handler itself operates on a
    *different* org (resolved separately from ``X-Org-Id`` via
    ``EFFECTIVE_ORG_ID``). These tests pin the fix: such a route must
    always ignore query-string ``org_id`` and re-resolve the target org via
    ``resolve_target_org_id_for_permission_check`` -- the same resolver
    ``EFFECTIVE_ORG_ID`` uses -- so the two can never diverge.
    """

    @pytest.mark.asyncio
    async def test_query_string_org_id_is_ignored_without_path_org_id(self):
        """
        GIVEN: A route with no {org_id} path segment; the caller is only a
               MEMBER of org_a (the request's real effective org) but is
               OWNER of org_b
        WHEN: The caller supplies ?org_id=<org_b> as a query-string override
        THEN: org_b is discarded -- the permission check queries org_a (via
              resolve_target_org_id_for_permission_check) and denies the
              member-role caller EDIT_ORG_SETTINGS
        """
        user_id = str(uuid4())
        org_a = uuid4()  # the request's real effective org; caller is a mere member
        org_b = uuid4()  # an org the caller owns -- smuggled in via ?org_id=

        mock_request = _create_mock_request()
        mock_request.path_params = {}

        member_role = MagicMock()
        member_role.name = 'member'

        async def fake_get_user_org_role(_user_id, queried_org_id):
            assert queried_org_id == org_a, (
                'permission check queried the attacker-supplied org_id '
                "instead of the request's real effective org"
            )
            return member_role

        with (
            patch(
                'server.auth.authorization.get_user_org_role',
                AsyncMock(side_effect=fake_get_user_org_role),
            ),
            patch(
                'server.auth.org_context.resolve_target_org_id_for_permission_check',
                AsyncMock(return_value=org_a),
            ),
        ):
            permission_checker = require_permission(Permission.EDIT_ORG_SETTINGS)
            with pytest.raises(HTTPException) as exc_info:
                await permission_checker(
                    request=mock_request, org_id=org_b, user_id=user_id
                )

        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_legitimate_no_query_org_id_still_resolves_effective_org(self):
        """
        GIVEN: A route with no {org_id} path segment and no query-string
               org_id supplied (the only legitimate shape before this fix)
        WHEN: Permission checker runs
        THEN: It still falls through to
              resolve_target_org_id_for_permission_check -- unchanged
              pre-existing behavior for the honest case
        """
        user_id = str(uuid4())
        org_a = uuid4()

        mock_request = _create_mock_request()
        mock_request.path_params = {}

        admin_role = MagicMock()
        admin_role.name = 'admin'

        with (
            patch(
                'server.auth.authorization.get_user_org_role',
                AsyncMock(return_value=admin_role),
            ) as mock_get_role,
            patch(
                'server.auth.org_context.resolve_target_org_id_for_permission_check',
                AsyncMock(return_value=org_a),
            ),
        ):
            permission_checker = require_permission(Permission.EDIT_ORG_SETTINGS)
            result = await permission_checker(
                request=mock_request, org_id=None, user_id=user_id
            )

        assert result == user_id
        mock_get_role.assert_awaited_once_with(user_id, org_a)

    @pytest.mark.asyncio
    async def test_path_scoped_org_id_is_still_trusted_directly(self):
        """
        GIVEN: A route WITH an {org_id} path segment (path_params carries it)
        WHEN: Permission checker runs
        THEN: The path-bound org_id is used directly with no fallback
              resolver call -- preserving org_profiles.py-style semantics
        """
        user_id = str(uuid4())
        org_id = uuid4()

        mock_request = _create_mock_request()
        mock_request.path_params = {'org_id': str(org_id)}

        admin_role = MagicMock()
        admin_role.name = 'admin'

        with (
            patch(
                'server.auth.authorization.get_user_org_role',
                AsyncMock(return_value=admin_role),
            ) as mock_get_role,
            patch(
                'server.auth.org_context.resolve_target_org_id_for_permission_check',
                AsyncMock(side_effect=AssertionError('should not be called')),
            ),
        ):
            permission_checker = require_permission(Permission.EDIT_ORG_SETTINGS)
            result = await permission_checker(
                request=mock_request, org_id=org_id, user_id=user_id
            )

        assert result == user_id
        mock_get_role.assert_awaited_once_with(user_id, org_id)


# =============================================================================
# Tests for permission-based access control scenarios
# =============================================================================


@pytest.mark.usefixtures('_no_super_role')
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
        mock_request = _create_mock_request()

        mock_role = MagicMock()
        mock_role.name = 'member'

        with patch(
            'server.auth.authorization.get_user_org_role',
            AsyncMock(return_value=mock_role),
        ):
            permission_checker = require_permission(Permission.MANAGE_SECRETS)
            result = await permission_checker(
                request=mock_request, org_id=org_id, user_id=user_id
            )
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
        mock_request = _create_mock_request()

        mock_role = MagicMock()
        mock_role.name = 'member'

        with patch(
            'server.auth.authorization.get_user_org_role',
            AsyncMock(return_value=mock_role),
        ):
            permission_checker = require_permission(
                Permission.INVITE_USER_TO_ORGANIZATION
            )
            with pytest.raises(HTTPException) as exc_info:
                await permission_checker(
                    request=mock_request, org_id=org_id, user_id=user_id
                )

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
        mock_request = _create_mock_request()

        mock_role = MagicMock()
        mock_role.name = 'admin'

        with patch(
            'server.auth.authorization.get_user_org_role',
            AsyncMock(return_value=mock_role),
        ):
            permission_checker = require_permission(
                Permission.INVITE_USER_TO_ORGANIZATION
            )
            result = await permission_checker(
                request=mock_request, org_id=org_id, user_id=user_id
            )
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
        mock_request = _create_mock_request()

        mock_role = MagicMock()
        mock_role.name = 'admin'

        with patch(
            'server.auth.authorization.get_user_org_role',
            AsyncMock(return_value=mock_role),
        ):
            permission_checker = require_permission(Permission.CHANGE_USER_ROLE_OWNER)
            with pytest.raises(HTTPException) as exc_info:
                await permission_checker(
                    request=mock_request, org_id=org_id, user_id=user_id
                )

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
        mock_request = _create_mock_request()

        mock_role = MagicMock()
        mock_role.name = 'owner'

        with patch(
            'server.auth.authorization.get_user_org_role',
            AsyncMock(return_value=mock_role),
        ):
            permission_checker = require_permission(Permission.CHANGE_USER_ROLE_OWNER)
            result = await permission_checker(
                request=mock_request, org_id=org_id, user_id=user_id
            )
            assert result == user_id


# =============================================================================
# Tests for API key organization validation
# =============================================================================


@pytest.mark.usefixtures('_no_super_role')
class TestApiKeyOrgValidation:
    """Tests for API key organization binding validation in require_permission."""

    @pytest.mark.asyncio
    async def test_allows_access_when_api_key_org_matches_target_org(self):
        """
        GIVEN: API key with org_id that matches the target org_id in the request
        WHEN: Permission checker is called
        THEN: User ID is returned (access allowed)
        """
        # Arrange
        user_id = str(uuid4())
        org_id = uuid4()
        mock_request = _create_mock_request(api_key_org_id=org_id)

        mock_role = MagicMock()
        mock_role.name = 'admin'

        # Act & Assert
        with patch(
            'server.auth.authorization.get_user_org_role',
            AsyncMock(return_value=mock_role),
        ):
            permission_checker = require_permission(Permission.VIEW_LLM_SETTINGS)
            result = await permission_checker(
                request=mock_request, org_id=org_id, user_id=user_id
            )
            assert result == user_id

    @pytest.mark.asyncio
    async def test_denies_access_when_api_key_org_mismatches_target_org(self):
        """
        GIVEN: API key created for Org A, but user tries to access Org B
        WHEN: Permission checker is called
        THEN: 403 Forbidden is raised with org mismatch message
        """
        # Arrange
        user_id = str(uuid4())
        api_key_org_id = uuid4()  # Org A - where API key was created
        target_org_id = uuid4()  # Org B - where user is trying to access
        mock_request = _create_mock_request(api_key_org_id=api_key_org_id)

        # Act & Assert
        permission_checker = require_permission(Permission.VIEW_LLM_SETTINGS)
        with pytest.raises(HTTPException) as exc_info:
            await permission_checker(
                request=mock_request, org_id=target_org_id, user_id=user_id
            )

        assert exc_info.value.status_code == 403
        assert (
            'API key is not authorized for this organization' in exc_info.value.detail
        )

    @pytest.mark.asyncio
    async def test_allows_access_for_unbound_api_key_without_org_binding(self):
        """
        GIVEN: Unbound API key (api_key_org_id is None)
        WHEN: Permission checker is called
        THEN: Falls through to the normal permission check.

        Unbound keys are explicitly supported: the effective org is
        resolved per-request via ``X-Org-Id`` or
        ``user.current_org_id``; here the explicit path ``{org_id}``
        parameter is what the route uses, so the unbound key does not
        impose any additional check beyond the regular role lookup.
        """
        # Arrange
        user_id = str(uuid4())
        org_id = uuid4()
        mock_request = _create_mock_request(api_key_org_id=None)

        mock_role = MagicMock()
        mock_role.name = 'admin'

        # Act & Assert
        with patch(
            'server.auth.authorization.get_user_org_role',
            AsyncMock(return_value=mock_role),
        ):
            permission_checker = require_permission(Permission.VIEW_LLM_SETTINGS)
            result = await permission_checker(
                request=mock_request, org_id=org_id, user_id=user_id
            )
            assert result == user_id

    @pytest.mark.asyncio
    async def test_allows_access_for_cookie_auth_without_api_key_org_id(self):
        """
        GIVEN: Cookie-based authentication (no api_key_org_id in user_auth)
        WHEN: Permission checker is called
        THEN: Falls through to normal permission check
        """
        # Arrange
        user_id = str(uuid4())
        org_id = uuid4()
        mock_request = _create_mock_request(api_key_org_id=None)

        mock_role = MagicMock()
        mock_role.name = 'admin'

        # Act & Assert
        with patch(
            'server.auth.authorization.get_user_org_role',
            AsyncMock(return_value=mock_role),
        ):
            permission_checker = require_permission(Permission.VIEW_LLM_SETTINGS)
            result = await permission_checker(
                request=mock_request, org_id=org_id, user_id=user_id
            )
            assert result == user_id

    @pytest.mark.asyncio
    async def test_logs_warning_on_api_key_org_mismatch(self):
        """
        GIVEN: API key org_id doesn't match target org_id
        WHEN: Permission checker is called
        THEN: Warning is logged with org mismatch details
        """
        # Arrange
        user_id = str(uuid4())
        api_key_org_id = uuid4()
        target_org_id = uuid4()
        mock_request = _create_mock_request(api_key_org_id=api_key_org_id)

        # Act & Assert
        with patch('server.auth.authorization.logger') as mock_logger:
            permission_checker = require_permission(Permission.VIEW_LLM_SETTINGS)
            with pytest.raises(HTTPException):
                await permission_checker(
                    request=mock_request, org_id=target_org_id, user_id=user_id
                )

            mock_logger.warning.assert_called()
            call_args = mock_logger.warning.call_args
            assert call_args[1]['extra']['user_id'] == user_id
            assert call_args[1]['extra']['api_key_org_id'] == str(api_key_org_id)
            assert call_args[1]['extra']['target_org_id'] == str(target_org_id)


class TestGetApiKeyOrgIdFromRequest:
    """Tests for get_api_key_org_id_from_request helper function."""

    @pytest.mark.asyncio
    async def test_returns_org_id_when_user_auth_has_api_key_org_id(self):
        """
        GIVEN: Request with user_auth that has api_key_org_id
        WHEN: get_api_key_org_id_from_request is called
        THEN: Returns the api_key_org_id
        """
        # Arrange
        org_id = uuid4()
        mock_request = _create_mock_request(api_key_org_id=org_id)

        # Act
        result = await get_api_key_org_id_from_request(mock_request)

        # Assert
        assert result == org_id

    @pytest.mark.asyncio
    async def test_returns_none_when_user_auth_has_no_api_key_org_id(self):
        """
        GIVEN: Request with user_auth that has no api_key_org_id (cookie auth)
        WHEN: get_api_key_org_id_from_request is called
        THEN: Returns None
        """
        # Arrange
        mock_request = _create_mock_request(api_key_org_id=None)

        # Act
        result = await get_api_key_org_id_from_request(mock_request)

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_user_auth_in_request(self):
        """
        GIVEN: Request without user_auth in state
        WHEN: get_api_key_org_id_from_request is called
        THEN: Returns None
        """
        # Arrange
        mock_request = MagicMock()
        mock_request.state.user_auth = None

        # Act
        result = await get_api_key_org_id_from_request(mock_request)

        # Assert
        assert result is None


# =============================================================================
# Tests for require_financial_data_access dependency
# =============================================================================


def _create_mock_request_with_email(api_key_org_id=None, user_email='user@example.com'):
    """Helper to create a mock request with optional api_key_org_id and email."""
    mock_request = MagicMock()
    mock_user_auth = MagicMock()
    # get_api_key_org_id is sync, not async
    mock_user_auth.get_api_key_org_id.return_value = api_key_org_id
    # get_user_email is async
    mock_user_auth.get_user_email = AsyncMock(return_value=user_email)
    mock_request.state.user_auth = mock_user_auth
    return mock_request


class TestRequireFinancialDataAccess:
    """Tests for require_financial_data_access compound authorization dependency."""

    @pytest.mark.asyncio
    async def test_grants_access_for_openhands_email(self):
        """
        GIVEN: User with @openhands.dev email
        WHEN: require_financial_data_access is called
        THEN: Returns user_id (access granted)
        """
        from server.auth.authorization import require_financial_data_access

        # Arrange
        user_id = str(uuid4())
        org_id = uuid4()
        mock_request = _create_mock_request_with_email(user_email='admin@openhands.dev')

        with patch(
            'server.auth.authorization.get_user_auth',
            AsyncMock(return_value=mock_request.state.user_auth),
        ):
            # Act
            result = await require_financial_data_access(
                request=mock_request, org_id=org_id, user_id=user_id
            )

            # Assert
            assert result == user_id

    @pytest.mark.asyncio
    async def test_grants_access_for_owner_role(self):
        """
        GIVEN: User with owner role in organization (non-@openhands.dev email)
        WHEN: require_financial_data_access is called
        THEN: Returns user_id (access granted)
        """
        from server.auth.authorization import require_financial_data_access

        # Arrange
        user_id = str(uuid4())
        org_id = uuid4()
        mock_request = _create_mock_request_with_email(user_email='user@company.com')
        mock_role = MagicMock()
        mock_role.name = 'owner'

        with (
            patch(
                'server.auth.authorization.get_user_auth',
                AsyncMock(return_value=mock_request.state.user_auth),
            ),
            patch(
                'server.auth.authorization.get_user_org_role',
                AsyncMock(return_value=mock_role),
            ),
        ):
            # Act
            result = await require_financial_data_access(
                request=mock_request, org_id=org_id, user_id=user_id
            )

            # Assert
            assert result == user_id

    @pytest.mark.asyncio
    async def test_grants_access_for_admin_role(self):
        """
        GIVEN: User with admin role in organization (non-@openhands.dev email)
        WHEN: require_financial_data_access is called
        THEN: Returns user_id (access granted)
        """
        from server.auth.authorization import require_financial_data_access

        # Arrange
        user_id = str(uuid4())
        org_id = uuid4()
        mock_request = _create_mock_request_with_email(user_email='user@company.com')
        mock_role = MagicMock()
        mock_role.name = 'admin'

        with (
            patch(
                'server.auth.authorization.get_user_auth',
                AsyncMock(return_value=mock_request.state.user_auth),
            ),
            patch(
                'server.auth.authorization.get_user_org_role',
                AsyncMock(return_value=mock_role),
            ),
        ):
            # Act
            result = await require_financial_data_access(
                request=mock_request, org_id=org_id, user_id=user_id
            )

            # Assert
            assert result == user_id

    @pytest.mark.asyncio
    async def test_denies_access_for_member_role_without_openhands_email(self):
        """
        GIVEN: User with member role (not admin/owner) and non-@openhands.dev email
        WHEN: require_financial_data_access is called
        THEN: Raises 403 Forbidden
        """
        from server.auth.authorization import require_financial_data_access

        # Arrange
        user_id = str(uuid4())
        org_id = uuid4()
        mock_request = _create_mock_request_with_email(user_email='user@company.com')
        mock_role = MagicMock()
        mock_role.name = 'member'

        with (
            patch(
                'server.auth.authorization.get_user_auth',
                AsyncMock(return_value=mock_request.state.user_auth),
            ),
            patch(
                'server.auth.authorization.get_user_org_role',
                AsyncMock(return_value=mock_role),
            ),
        ):
            # Act & Assert
            with pytest.raises(HTTPException) as exc_info:
                await require_financial_data_access(
                    request=mock_request, org_id=org_id, user_id=user_id
                )

            assert exc_info.value.status_code == 403
            assert 'admins, owners, or OpenHands' in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_denies_access_for_non_member(self):
        """
        GIVEN: User who is not a member of the organization
        WHEN: require_financial_data_access is called
        THEN: Raises 403 Forbidden
        """
        from server.auth.authorization import require_financial_data_access

        # Arrange
        user_id = str(uuid4())
        org_id = uuid4()
        mock_request = _create_mock_request_with_email(user_email='user@company.com')

        with (
            patch(
                'server.auth.authorization.get_user_auth',
                AsyncMock(return_value=mock_request.state.user_auth),
            ),
            patch(
                'server.auth.authorization.get_user_org_role',
                AsyncMock(return_value=None),
            ),
        ):
            # Act & Assert
            with pytest.raises(HTTPException) as exc_info:
                await require_financial_data_access(
                    request=mock_request, org_id=org_id, user_id=user_id
                )

            assert exc_info.value.status_code == 403
            assert 'not a member' in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_denies_access_when_not_authenticated(self):
        """
        GIVEN: No user_id (not authenticated)
        WHEN: require_financial_data_access is called
        THEN: Raises 401 Unauthorized
        """
        from server.auth.authorization import require_financial_data_access

        # Arrange
        org_id = uuid4()
        mock_request = _create_mock_request_with_email()

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await require_financial_data_access(
                request=mock_request, org_id=org_id, user_id=None
            )

        assert exc_info.value.status_code == 401
        assert 'not authenticated' in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_denies_access_when_api_key_org_mismatch(self):
        """
        GIVEN: API key created for Org A, but user tries to access Org B
        WHEN: require_financial_data_access is called
        THEN: Raises 403 Forbidden with org mismatch message
        """
        from server.auth.authorization import require_financial_data_access

        # Arrange
        user_id = str(uuid4())
        api_key_org_id = uuid4()  # Org A
        target_org_id = uuid4()  # Org B
        mock_request = _create_mock_request_with_email(
            api_key_org_id=api_key_org_id, user_email='admin@openhands.dev'
        )

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await require_financial_data_access(
                request=mock_request, org_id=target_org_id, user_id=user_id
            )

        assert exc_info.value.status_code == 403
        assert 'API key is not authorized' in exc_info.value.detail


# =============================================================================
# Tests for super-role helpers and fallback logic
# =============================================================================


class TestSuperRoleName:
    """Tests for the ``super_role_name`` helper."""

    def test_prepends_super_prefix(self):
        """
        GIVEN: a regular role name
        WHEN: super_role_name is called
        THEN: the returned label is the regular role name prefixed with "super"
        """
        assert super_role_name('owner') == 'superowner'
        assert super_role_name('admin') == 'superadmin'
        assert super_role_name('member') == 'supermember'


class TestSuperRolePermissions:
    """Tests for explicit super-role permission mappings."""

    def test_super_role_permissions_are_explicit(self):
        """
        GIVEN: SUPER_ROLE_PERMISSIONS mapping
        WHEN: comparing it to the org-scoped role mappings
        THEN: super roles do not inherit org-scoped permissions implicitly.
        """
        assert SUPER_ROLE_PERMISSIONS[RoleName.OWNER] == frozenset()
        assert SUPER_ROLE_PERMISSIONS[RoleName.ADMIN] == frozenset(
            [
                Permission.CREATE_ORGANIZATION,
                Permission.PROVISION_USER,
                Permission.MANAGE_SUPER_ADMINS,
            ]
        )
        assert SUPER_ROLE_PERMISSIONS[RoleName.MEMBER] == frozenset()

    def test_super_role_keys_match_regular_role_keys(self):
        """
        GIVEN: super roles parallel regular roles 1:1
        WHEN: comparing the keysets of ROLE_PERMISSIONS and SUPER_ROLE_PERMISSIONS
        THEN: they match exactly
        """
        assert set(SUPER_ROLE_PERMISSIONS.keys()) == set(ROLE_PERMISSIONS.keys())

    def test_get_super_role_permissions_owner_is_not_functional_yet(self):
        """
        GIVEN: role name 'owner'
        WHEN: get_super_role_permissions is called
        THEN: no instance-level permissions are currently assigned.
        """
        perms = get_super_role_permissions('owner')
        assert perms == frozenset()

    def test_get_super_role_permissions_invalid_role(self):
        """
        GIVEN: invalid role name
        WHEN: get_super_role_permissions is called
        THEN: an empty frozenset is returned
        """
        assert get_super_role_permissions('not_a_role') == frozenset()

    def test_only_superadmin_grants_instance_org_management(self):
        """
        GIVEN: SUPER_ROLE_PERMISSIONS mapping
        WHEN: looking up instance-level org management permissions
        THEN: only superadmin carries CREATE_ORGANIZATION and PROVISION_USER.
        """
        assert (
            Permission.CREATE_ORGANIZATION not in SUPER_ROLE_PERMISSIONS[RoleName.OWNER]
        )
        assert Permission.CREATE_ORGANIZATION in SUPER_ROLE_PERMISSIONS[RoleName.ADMIN]
        assert (
            Permission.CREATE_ORGANIZATION
            not in SUPER_ROLE_PERMISSIONS[RoleName.MEMBER]
        )
        assert Permission.PROVISION_USER not in SUPER_ROLE_PERMISSIONS[RoleName.OWNER]
        assert Permission.PROVISION_USER in SUPER_ROLE_PERMISSIONS[RoleName.ADMIN]
        assert Permission.PROVISION_USER not in SUPER_ROLE_PERMISSIONS[RoleName.MEMBER]


class TestHasPermissionSuper:
    """Tests for ``has_permission`` with the ``is_super`` flag."""

    def test_super_admin_has_create_organization(self):
        """
        GIVEN: an admin role evaluated as a super role
        WHEN: checking CREATE_ORGANIZATION
        THEN: the explicit instance-level permission is granted.
        """
        mock_role = MagicMock()
        mock_role.name = 'admin'
        assert (
            has_permission(mock_role, Permission.CREATE_ORGANIZATION, is_super=True)
            is True
        )

    def test_super_admin_does_not_inherit_org_admin_permissions(self):
        """
        GIVEN: an admin role evaluated as a super role
        WHEN: checking an org-admin permission
        THEN: the permission is not granted unless explicitly listed.
        """
        mock_role = MagicMock()
        mock_role.name = 'admin'
        assert (
            has_permission(mock_role, Permission.MANAGE_SECRETS, is_super=True) is False
        )

    def test_super_owner_does_not_inherit_org_owner_permissions(self):
        """
        GIVEN: owner role evaluated as a super role
        WHEN: checking DELETE_ORGANIZATION
        THEN: the permission is not granted until explicitly listed.
        """
        mock_role = MagicMock()
        mock_role.name = 'owner'
        assert (
            has_permission(mock_role, Permission.DELETE_ORGANIZATION, is_super=True)
            is False
        )


class TestGetUserSuperRole:
    """Tests for the ``get_user_super_role`` helper."""

    @pytest.mark.asyncio
    async def test_returns_role_when_user_has_role_id(self):
        """
        GIVEN: a user with a non-null role_id
        WHEN: get_user_super_role is called
        THEN: the corresponding Role is returned
        """
        user_id = str(uuid4())

        mock_user = MagicMock()
        mock_user.role_id = 42

        mock_role = MagicMock()
        mock_role.name = 'admin'

        with (
            patch(
                'server.auth.authorization.UserStore.get_user_by_id',
                AsyncMock(return_value=mock_user),
            ),
            patch(
                'server.auth.authorization.RoleStore.get_role_by_id',
                AsyncMock(return_value=mock_role),
            ) as mock_get_role,
        ):
            result = await get_user_super_role(user_id)
            assert result is mock_role
            mock_get_role.assert_awaited_once_with(42)

    @pytest.mark.asyncio
    async def test_returns_none_when_user_missing(self):
        """
        GIVEN: no user with the given id
        WHEN: get_user_super_role is called
        THEN: None is returned
        """
        user_id = str(uuid4())
        with patch(
            'server.auth.authorization.UserStore.get_user_by_id',
            AsyncMock(return_value=None),
        ):
            assert await get_user_super_role(user_id) is None

    @pytest.mark.asyncio
    async def test_returns_none_when_role_id_is_none(self):
        """
        GIVEN: a user with role_id == None
        WHEN: get_user_super_role is called
        THEN: None is returned (no DB lookup for the role)
        """
        user_id = str(uuid4())

        mock_user = MagicMock()
        mock_user.role_id = None

        with (
            patch(
                'server.auth.authorization.UserStore.get_user_by_id',
                AsyncMock(return_value=mock_user),
            ),
            patch(
                'server.auth.authorization.RoleStore.get_role_by_id',
                AsyncMock(return_value=MagicMock()),
            ) as mock_get_role,
        ):
            assert await get_user_super_role(user_id) is None
            mock_get_role.assert_not_called()


def _mock_role(name: str) -> MagicMock:
    role = MagicMock()
    role.name = name
    return role


class TestRequirePermissionSuperRoleFallback:
    """Tests covering the super-role fallback in ``require_permission``."""

    @pytest.mark.asyncio
    async def test_super_role_does_not_grant_unlisted_permission(self):
        """
        GIVEN: a member in the org and a non-functional superowner at the user level
        WHEN: require_permission(DELETE_ORGANIZATION) runs
        THEN: the super role does not help because DELETE_ORGANIZATION
              is not in the explicit super-role permission set.
        """
        user_id = str(uuid4())
        org_id = uuid4()
        mock_request = _create_mock_request()

        with (
            patch(
                'server.auth.authorization.get_user_org_role',
                AsyncMock(return_value=_mock_role('member')),
            ),
            patch(
                'server.auth.authorization.get_user_super_role',
                AsyncMock(return_value=_mock_role('owner')),
            ),
        ):
            permission_checker = require_permission(Permission.DELETE_ORGANIZATION)
            with pytest.raises(HTTPException) as exc_info:
                await permission_checker(
                    request=mock_request, org_id=org_id, user_id=user_id
                )
            assert exc_info.value.status_code == 403
            assert 'delete_organization' in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_super_role_grants_explicit_permission_for_non_member(self):
        """
        GIVEN: a user with no org membership who has a superadmin role
               on the user record
        WHEN: require_permission(PROVISION_USER) runs
        THEN: the explicit super-role permission grants access.
        """
        user_id = str(uuid4())
        org_id = uuid4()
        mock_request = _create_mock_request()

        with (
            patch(
                'server.auth.authorization.get_user_org_role',
                AsyncMock(return_value=None),
            ),
            patch(
                'server.auth.authorization.get_user_super_role',
                AsyncMock(return_value=_mock_role('admin')),
            ),
        ):
            permission_checker = require_permission(Permission.PROVISION_USER)
            result = await permission_checker(
                request=mock_request, org_id=org_id, user_id=user_id
            )
            assert result == user_id

    @pytest.mark.asyncio
    async def test_super_role_does_not_help_when_lacks_permission(self):
        """
        GIVEN: a member in the org and a supermember at the user level
               (neither role has DELETE_ORGANIZATION)
        WHEN: require_permission(DELETE_ORGANIZATION) runs
        THEN: 403 Forbidden is raised with 'Requires ...' detail
        """
        user_id = str(uuid4())
        org_id = uuid4()
        mock_request = _create_mock_request()

        with (
            patch(
                'server.auth.authorization.get_user_org_role',
                AsyncMock(return_value=_mock_role('member')),
            ),
            patch(
                'server.auth.authorization.get_user_super_role',
                AsyncMock(return_value=_mock_role('member')),
            ),
        ):
            permission_checker = require_permission(Permission.DELETE_ORGANIZATION)
            with pytest.raises(HTTPException) as exc_info:
                await permission_checker(
                    request=mock_request, org_id=org_id, user_id=user_id
                )
            assert exc_info.value.status_code == 403
            assert 'delete_organization' in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_non_member_with_insufficient_super_role_returns_403(self):
        """
        GIVEN: a non-member user whose super role lacks the required permission
        WHEN: require_permission(DELETE_ORGANIZATION) runs
        THEN: 403 with 'not a member' detail is raised
        """
        user_id = str(uuid4())
        org_id = uuid4()
        mock_request = _create_mock_request()

        with (
            patch(
                'server.auth.authorization.get_user_org_role',
                AsyncMock(return_value=None),
            ),
            patch(
                'server.auth.authorization.get_user_super_role',
                AsyncMock(return_value=_mock_role('member')),
            ),
        ):
            permission_checker = require_permission(Permission.DELETE_ORGANIZATION)
            with pytest.raises(HTTPException) as exc_info:
                await permission_checker(
                    request=mock_request, org_id=org_id, user_id=user_id
                )
            assert exc_info.value.status_code == 403
            assert 'not a member' in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_org_role_short_circuits_super_role_lookup(self):
        """
        GIVEN: an admin in the org who also has a super role
        WHEN: require_permission(VIEW_LLM_SETTINGS) runs (admin has it)
        THEN: ``get_user_super_role`` is not called -- the org role
              already grants access
        """
        user_id = str(uuid4())
        org_id = uuid4()
        mock_request = _create_mock_request()

        super_role_mock = AsyncMock(return_value=_mock_role('owner'))

        with (
            patch(
                'server.auth.authorization.get_user_org_role',
                AsyncMock(return_value=_mock_role('admin')),
            ),
            patch(
                'server.auth.authorization.get_user_super_role',
                super_role_mock,
            ),
        ):
            permission_checker = require_permission(Permission.VIEW_LLM_SETTINGS)
            result = await permission_checker(
                request=mock_request, org_id=org_id, user_id=user_id
            )
            assert result == user_id
            super_role_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_implicit_org_route_super_role_grants_explicit_permission_for_non_member(
        self,
    ):
        """
        GIVEN: a route without an ``{org_id}`` path parameter, an
               ``X-Org-Id`` header that targets an org the user is NOT
               a member of, and a superadmin role on the user record
        WHEN: ``require_permission(PROVISION_USER)`` runs
        THEN: the resolver skips the membership rejection and the
              explicit super-role permission grants access.
        """
        user_id = str(uuid4())
        target_org_id = uuid4()
        mock_request = _create_mock_request()

        # The resolver must be invoked because org_id is None on the route.
        # It returns the target org without performing a membership check.
        with (
            patch(
                'server.auth.org_context.resolve_target_org_id_for_permission_check',
                AsyncMock(return_value=target_org_id),
            ) as resolver,
            patch(
                'server.auth.authorization.get_user_org_role',
                AsyncMock(return_value=None),
            ),
            patch(
                'server.auth.authorization.get_user_super_role',
                AsyncMock(return_value=_mock_role('admin')),
            ),
        ):
            permission_checker = require_permission(Permission.PROVISION_USER)
            result = await permission_checker(
                request=mock_request, org_id=None, user_id=user_id
            )

        assert result == user_id
        resolver.assert_awaited_once_with(mock_request)

    @pytest.mark.asyncio
    async def test_implicit_org_route_non_member_without_super_role_returns_403(self):
        """
        GIVEN: a route without ``{org_id}``, the X-Org-Id header targets
               a non-member org, and the user has no super role
        WHEN: require_permission runs
        THEN: the resolver returns the target org, the org-role lookup
              returns None, the super-role lookup returns None, so 403
              is raised
        """
        user_id = str(uuid4())
        target_org_id = uuid4()
        mock_request = _create_mock_request()

        with (
            patch(
                'server.auth.org_context.resolve_target_org_id_for_permission_check',
                AsyncMock(return_value=target_org_id),
            ),
            patch(
                'server.auth.authorization.get_user_org_role',
                AsyncMock(return_value=None),
            ),
            patch(
                'server.auth.authorization.get_user_super_role',
                AsyncMock(return_value=None),
            ),
        ):
            permission_checker = require_permission(Permission.VIEW_LLM_SETTINGS)
            with pytest.raises(HTTPException) as exc_info:
                await permission_checker(
                    request=mock_request, org_id=None, user_id=user_id
                )
        assert exc_info.value.status_code == 403
        assert 'not a member' in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_log_records_super_role_label(self):
        """
        GIVEN: an insufficient-permission scenario where the user has a
               supermember role (also insufficient)
        WHEN: require_permission(DELETE_ORGANIZATION) logs the denial
        THEN: the warning's ``extra`` dict carries the ``supermember``
              label (i.e. uses the conceptual super-role name)
        """
        user_id = str(uuid4())
        org_id = uuid4()
        mock_request = _create_mock_request()

        with (
            patch(
                'server.auth.authorization.get_user_org_role',
                AsyncMock(return_value=_mock_role('admin')),
            ),
            patch(
                'server.auth.authorization.get_user_super_role',
                AsyncMock(return_value=_mock_role('member')),
            ),
            patch('server.auth.authorization.logger') as mock_logger,
        ):
            permission_checker = require_permission(Permission.DELETE_ORGANIZATION)
            with pytest.raises(HTTPException):
                await permission_checker(
                    request=mock_request, org_id=org_id, user_id=user_id
                )

            mock_logger.warning.assert_called()
            extra = mock_logger.warning.call_args[1]['extra']
            assert extra['user_role'] == 'admin'
            assert extra['super_role'] == 'supermember'
