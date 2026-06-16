"""Unit tests for the user-provisioning admin endpoint.

These tests exercise the route handler directly (rather than through the
FastAPI test client) so they can mock the underlying Keycloak, database,
and LiteLLM dependencies without bringing up the entire SAAS stack. The
permission wiring itself is exercised separately by asserting on
``ROLE_PERMISSIONS``.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from keycloak.exceptions import KeycloakError
from pydantic import SecretStr
from server.auth.authorization import (
    ROLE_PERMISSIONS,
    Permission,
    RoleName,
)
from server.routes.user_provisioning import (
    DEFAULT_PROVISIONED_ROLE,
    ProvisionUserRequest,
    _generate_password,
    provision_user,
)


class TestGeneratePassword:
    """The generated password must satisfy a basic complexity policy."""

    def test_length_and_complexity(self):
        for _ in range(5):
            pw = _generate_password()
            assert len(pw) == 24
            assert any(c.islower() for c in pw)
            assert any(c.isupper() for c in pw)
            assert any(c.isdigit() for c in pw)
            assert any(c in '!@#$%^&*-_=+' for c in pw)

    def test_custom_length(self):
        pw = _generate_password(length=32)
        assert len(pw) == 32


class TestProvisionUserPermissionWiring:
    """The new permission must be present and assigned only to OWNER/ADMIN."""

    def test_permission_enum_includes_provision_user(self):
        assert Permission.PROVISION_USER.value == 'provision_user'

    def test_owner_has_permission(self):
        assert Permission.PROVISION_USER in ROLE_PERMISSIONS[RoleName.OWNER]

    def test_admin_has_permission(self):
        assert Permission.PROVISION_USER in ROLE_PERMISSIONS[RoleName.ADMIN]

    def test_member_does_not_have_permission(self):
        assert Permission.PROVISION_USER not in ROLE_PERMISSIONS[RoleName.MEMBER]


class TestProvisionUserRequestValidation:
    def test_email_is_required(self):
        with pytest.raises(ValueError):
            ProvisionUserRequest(email='not-an-email')  # type: ignore[arg-type]

    def test_password_min_length(self):
        with pytest.raises(ValueError):
            ProvisionUserRequest(email='a@b.com', password='short')

    def test_optional_password(self):
        req = ProvisionUserRequest(email='a@b.com')
        assert req.password is None


class TestProvisionUserHandler:
    """End-to-end handler test with all external collaborators mocked."""

    @pytest.fixture
    def caller_user_id(self) -> str:
        return '11111111-1111-1111-1111-111111111111'

    @pytest.fixture
    def target_org_id(self) -> uuid.UUID:
        return uuid.UUID('22222222-2222-2222-2222-222222222222')

    @pytest.fixture
    def new_user_id(self) -> str:
        # Distinct from target_org_id so the route takes the
        # "add to non-personal org" branch.
        return '33333333-3333-3333-3333-333333333333'

    def _patch_dependencies(
        self,
        new_user_id: str,
        target_org_id: uuid.UUID,
        *,
        org_exists: bool = True,
        keycloak_raises: Exception | None = None,
    ):
        """Return a stack of patches as a list of context managers.

        Tests enter all of them via ``contextlib.ExitStack`` so each
        patch's mock can be asserted on individually.
        """
        token_manager_mock = MagicMock()
        token_manager_mock.create_keycloak_user = AsyncMock(
            side_effect=keycloak_raises if keycloak_raises else None,
            return_value=new_user_id,
        )
        token_manager_mock.request_offline_token = AsyncMock(
            return_value='offline-refresh-token'
        )
        token_manager_mock.store_offline_token = AsyncMock()
        token_manager_mock.delete_keycloak_user = AsyncMock(return_value=True)

        new_user = MagicMock()
        new_user.id = uuid.UUID(new_user_id)

        settings_mock = MagicMock()
        settings_mock.agent_settings.llm.api_key = SecretStr('litellm-key')

        role_mock = MagicMock()
        role_mock.id = 42

        api_key_store_mock = MagicMock()
        api_key_store_mock.create_api_key = AsyncMock(
            return_value='sk-oh-generated-api-key'
        )

        org = MagicMock() if org_exists else None

        patches = [
            patch(
                'server.routes.user_provisioning.TokenManager',
                return_value=token_manager_mock,
            ),
            patch(
                'server.routes.user_provisioning.OrgStore.get_org_by_id',
                new_callable=AsyncMock,
                return_value=org,
            ),
            patch(
                'server.routes.user_provisioning.UserStore.create_user',
                new_callable=AsyncMock,
                return_value=new_user,
            ),
            patch(
                'server.routes.user_provisioning._set_user_provisioned_flags',
                new_callable=AsyncMock,
            ),
            patch(
                'server.routes.user_provisioning.OrgService.create_litellm_integration',
                new_callable=AsyncMock,
                return_value=settings_mock,
            ),
            patch(
                'server.routes.user_provisioning.RoleStore.get_role_by_name',
                new_callable=AsyncMock,
                return_value=role_mock,
            ),
            patch(
                'server.routes.user_provisioning.OrgMemberStore.add_user_to_org',
                new_callable=AsyncMock,
            ),
            patch(
                'server.routes.user_provisioning.ApiKeyStore.get_instance',
                return_value=api_key_store_mock,
            ),
        ]
        return patches, {
            'token_manager': token_manager_mock,
            'api_key_store': api_key_store_mock,
        }

    @pytest.mark.asyncio
    async def test_happy_path_with_supplied_password(
        self, caller_user_id, target_org_id, new_user_id
    ):
        patches, handles = self._patch_dependencies(new_user_id, target_org_id)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
        ):
            resp = await provision_user(
                body=ProvisionUserRequest(
                    email='Alice@Example.com',
                    password='SuperSecret-1234',
                ),
                caller_user_id=caller_user_id,
                target_org_id=target_org_id,
            )

        assert resp.email == 'alice@example.com'
        assert resp.password == 'SuperSecret-1234'
        assert resp.api_key == 'sk-oh-generated-api-key'
        assert resp.user_id == new_user_id
        assert resp.org_id == str(target_org_id)

        # Offline token must have been stored against the newly created
        # Keycloak user id, not against the caller.
        handles['token_manager'].store_offline_token.assert_awaited_once_with(
            user_id=new_user_id, offline_token='offline-refresh-token'
        )
        # API key must be bound to the target org, not the personal one.
        handles['api_key_store'].create_api_key.assert_awaited_once()
        kwargs = handles['api_key_store'].create_api_key.await_args.kwargs
        assert kwargs['org_id'] == target_org_id
        assert kwargs['user_id'] == new_user_id

    @pytest.mark.asyncio
    async def test_generates_password_when_not_supplied(
        self, caller_user_id, target_org_id, new_user_id
    ):
        patches, handles = self._patch_dependencies(new_user_id, target_org_id)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
        ):
            resp = await provision_user(
                body=ProvisionUserRequest(email='bob@example.com'),
                caller_user_id=caller_user_id,
                target_org_id=target_org_id,
            )

        assert len(resp.password) >= 8
        # Verify the same generated password was used for the Keycloak
        # account creation, not regenerated each time.
        kc_call = handles['token_manager'].create_keycloak_user.await_args
        assert kc_call.kwargs['password'] == resp.password

    @pytest.mark.asyncio
    async def test_target_org_not_found_returns_404(
        self, caller_user_id, target_org_id, new_user_id
    ):
        patches, handles = self._patch_dependencies(
            new_user_id, target_org_id, org_exists=False
        )
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
        ):
            with pytest.raises(HTTPException) as exc_info:
                await provision_user(
                    body=ProvisionUserRequest(email='bob@example.com'),
                    caller_user_id=caller_user_id,
                    target_org_id=target_org_id,
                )
        assert exc_info.value.status_code == 404
        # Keycloak must not have been touched.
        handles['token_manager'].create_keycloak_user.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_keycloak_failure_returns_409(
        self, caller_user_id, target_org_id, new_user_id
    ):
        patches, handles = self._patch_dependencies(
            new_user_id,
            target_org_id,
            keycloak_raises=KeycloakError('user already exists'),
        )
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
        ):
            with pytest.raises(HTTPException) as exc_info:
                await provision_user(
                    body=ProvisionUserRequest(email='dup@example.com'),
                    caller_user_id=caller_user_id,
                    target_org_id=target_org_id,
                )
        assert exc_info.value.status_code == 409
        # Cleanup should not run if Keycloak creation itself failed.
        handles['token_manager'].delete_keycloak_user.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cleanup_on_post_keycloak_failure(
        self, caller_user_id, target_org_id, new_user_id
    ):
        patches, handles = self._patch_dependencies(new_user_id, target_org_id)
        # Make the offline-token step blow up after Keycloak succeeded.
        handles['token_manager'].request_offline_token.side_effect = RuntimeError(
            'boom'
        )

        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
        ):
            with pytest.raises(HTTPException) as exc_info:
                await provision_user(
                    body=ProvisionUserRequest(email='bob@example.com'),
                    caller_user_id=caller_user_id,
                    target_org_id=target_org_id,
                )
        assert exc_info.value.status_code == 500
        # The Keycloak user that was just created must be cleaned up
        # so we don't orphan identities.
        handles['token_manager'].delete_keycloak_user.assert_awaited_once_with(
            new_user_id
        )

    @pytest.mark.asyncio
    async def test_skips_add_to_org_when_target_is_personal_org(
        self, caller_user_id, target_org_id, new_user_id
    ):
        # When the X-Org-Id matches the user's freshly-created personal
        # org (id == user_id), re-adding would violate the unique
        # constraint. The route must skip the explicit add.
        personal_org_id = uuid.UUID(new_user_id)
        patches, handles = self._patch_dependencies(new_user_id, personal_org_id)
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
        ):
            # Wrap the OrgMemberStore.add_user_to_org mock so we can
            # assert it was *not* called.
            with patch(
                'server.routes.user_provisioning.OrgMemberStore.add_user_to_org',
                new_callable=AsyncMock,
            ) as add_member_mock:
                await provision_user(
                    body=ProvisionUserRequest(email='solo@example.com'),
                    caller_user_id=caller_user_id,
                    target_org_id=personal_org_id,
                )
                add_member_mock.assert_not_awaited()

    def test_default_role_is_member(self):
        # Document the policy: provisioned users are not auto-promoted.
        assert DEFAULT_PROVISIONED_ROLE == 'member'
