"""Unit tests for the user-provisioning admin endpoint.

These tests exercise the route handler directly (rather than through the
FastAPI test client) so they can mock the underlying Keycloak, database,
and LiteLLM dependencies without bringing up the entire SAAS stack. The
permission wiring itself is exercised separately by asserting on
``ROLE_PERMISSIONS``.
"""

from __future__ import annotations

import contextlib
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
    """The provision permission is available to org admins and super roles."""

    def test_permission_enum_includes_provision_user(self):
        assert Permission.PROVISION_USER.value == 'provision_user'

    def test_owner_has_permission(self):
        assert Permission.PROVISION_USER in ROLE_PERMISSIONS[RoleName.OWNER]

    def test_admin_has_permission(self):
        assert Permission.PROVISION_USER in ROLE_PERMISSIONS[RoleName.ADMIN]

    def test_member_does_not_have_permission(self):
        assert Permission.PROVISION_USER not in ROLE_PERMISSIONS[RoleName.MEMBER]

    def test_superadmin_has_permission(self):
        from server.auth.authorization import SUPER_ROLE_PERMISSIONS

        assert Permission.PROVISION_USER in SUPER_ROLE_PERMISSIONS[RoleName.ADMIN]

    def test_superowner_does_not_have_permission_yet(self):
        from server.auth.authorization import SUPER_ROLE_PERMISSIONS

        assert Permission.PROVISION_USER not in SUPER_ROLE_PERMISSIONS[RoleName.OWNER]

    def test_supermember_does_not_have_permission(self):
        from server.auth.authorization import SUPER_ROLE_PERMISSIONS

        assert Permission.PROVISION_USER not in SUPER_ROLE_PERMISSIONS[RoleName.MEMBER]


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

    def test_default_role_is_member(self):
        req = ProvisionUserRequest(email='a@b.com')
        assert req.role == 'member'

    def test_admin_role_is_allowed(self):
        req = ProvisionUserRequest(email='a@b.com', role='admin')
        assert req.role == 'admin'

    def test_owner_role_is_allowed(self):
        req = ProvisionUserRequest(email='a@b.com', role='owner')
        assert req.role == 'owner'


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
        role_store_mock = AsyncMock(return_value=role_mock)

        api_key_store_mock = MagicMock()
        api_key_store_mock.create_api_key = AsyncMock(
            return_value='sk-oh-generated-api-key'
        )

        org = MagicMock() if org_exists else None

        # Rollback collaborators. We construct these as standalone
        # ``AsyncMock``s so tests can assert on call counts and
        # arguments without having to dig back into the patch object.
        delete_org_cascade_mock = AsyncMock(return_value=org)
        remove_member_mock = AsyncMock(return_value=True)
        set_flags_mock = AsyncMock()
        add_user_to_org_mock = AsyncMock()

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
                set_flags_mock,
            ),
            patch(
                'server.routes.user_provisioning.OrgService.create_litellm_integration',
                new_callable=AsyncMock,
                return_value=settings_mock,
            ),
            patch(
                'server.routes.user_provisioning.RoleStore.get_role_by_name',
                role_store_mock,
            ),
            patch(
                'server.routes.user_provisioning.OrgMemberStore.add_user_to_org',
                add_user_to_org_mock,
            ),
            patch(
                'server.routes.user_provisioning.ApiKeyStore.get_instance',
                return_value=api_key_store_mock,
            ),
            # Rollback path: ``_rollback_partial_provision`` calls
            # ``OrgMemberStore.remove_user_from_org`` first, then
            # ``OrgStore.delete_org_cascade`` on the personal org,
            # then ``TokenManager.delete_keycloak_user``. Patch the
            # first two here so the tests can assert on the rollback
            # order without hitting a real DB.
            patch(
                'server.routes.user_provisioning.OrgMemberStore.remove_user_from_org',
                remove_member_mock,
            ),
            patch(
                'server.routes.user_provisioning.OrgStore.delete_org_cascade',
                delete_org_cascade_mock,
            ),
        ]
        return patches, {
            'token_manager': token_manager_mock,
            'api_key_store': api_key_store_mock,
            'set_flags': set_flags_mock,
            'add_user_to_org': add_user_to_org_mock,
            'role_store': role_store_mock,
            'remove_member': remove_member_mock,
            'delete_org_cascade': delete_org_cascade_mock,
        }

    @staticmethod
    def _enter_all(patches):
        """Enter every patch in ``patches`` via an ``ExitStack``.

        Returning the stack lets the caller use ``with stack:`` to
        guarantee tear-down. This avoids the brittle
        ``with (patches[0], patches[1], ...)`` pattern that has to be
        edited every time the patch list grows.
        """
        stack = contextlib.ExitStack()
        for p in patches:
            stack.enter_context(p)
        return stack

    @pytest.mark.asyncio
    async def test_happy_path_with_supplied_password(
        self, caller_user_id, target_org_id, new_user_id
    ):
        patches, handles = self._patch_dependencies(new_user_id, target_org_id)
        with self._enter_all(patches):
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
        assert resp.role == 'member'

        # Offline token must have been stored against the newly created
        # Keycloak user id, not against the caller.
        handles['token_manager'].store_offline_token.assert_awaited_once_with(
            user_id=new_user_id, offline_token='offline-refresh-token'
        )
        handles['role_store'].assert_awaited_once_with('member')
        handles['add_user_to_org'].assert_awaited_once()
        add_kwargs = handles['add_user_to_org'].await_args.kwargs
        assert add_kwargs['role_id'] == 42

        # API key must be bound to the target org, not the personal one.
        handles['api_key_store'].create_api_key.assert_awaited_once()
        kwargs = handles['api_key_store'].create_api_key.await_args.kwargs
        assert kwargs['org_id'] == target_org_id
        assert kwargs['user_id'] == new_user_id
        # On the happy path no rollback should run.
        handles['delete_org_cascade'].assert_not_awaited()
        handles['remove_member'].assert_not_awaited()
        handles['token_manager'].delete_keycloak_user.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_can_provision_admin_role(
        self, caller_user_id, target_org_id, new_user_id
    ):
        patches, handles = self._patch_dependencies(new_user_id, target_org_id)
        with self._enter_all(patches):
            resp = await provision_user(
                body=ProvisionUserRequest(
                    email='admin@example.com',
                    password='SuperSecret-1234',
                    role='admin',
                ),
                caller_user_id=caller_user_id,
                target_org_id=target_org_id,
            )

        assert resp.role == 'admin'
        handles['role_store'].assert_awaited_once_with('admin')
        handles['add_user_to_org'].assert_awaited_once()
        add_kwargs = handles['add_user_to_org'].await_args.kwargs
        assert add_kwargs['role_id'] == 42

    @pytest.mark.asyncio
    async def test_can_provision_owner_role(
        self, caller_user_id, target_org_id, new_user_id
    ):
        patches, handles = self._patch_dependencies(new_user_id, target_org_id)
        with self._enter_all(patches):
            resp = await provision_user(
                body=ProvisionUserRequest(
                    email='owner@example.com',
                    password='SuperSecret-1234',
                    role='owner',
                ),
                caller_user_id=caller_user_id,
                target_org_id=target_org_id,
            )

        assert resp.role == 'owner'
        handles['role_store'].assert_awaited_once_with('owner')
        handles['add_user_to_org'].assert_awaited_once()
        add_kwargs = handles['add_user_to_org'].await_args.kwargs
        assert add_kwargs['role_id'] == 42

    @pytest.mark.asyncio
    async def test_generates_password_when_not_supplied(
        self, caller_user_id, target_org_id, new_user_id
    ):
        patches, handles = self._patch_dependencies(new_user_id, target_org_id)
        with self._enter_all(patches):
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
        with self._enter_all(patches):
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
    async def test_personal_workspace_rejected_with_403(
        self, caller_user_id, new_user_id
    ):
        """``target_org_id == caller's user id`` is a personal workspace.

        Every user is the owner of their personal org (Org.id ==
        User.id == UUID(keycloak.sub)), so a bare permission check
        on ``PROVISION_USER`` would otherwise let any normal user
        provision additional Keycloak/OpenHands accounts inside
        their own personal workspace and walk away with the
        credentials. Mirrors the personal-workspace rejection in
        ``server.services.org_invitation_service`` (403 Forbidden)
        and must run *before* Keycloak is touched.
        """
        caller_personal_org_id = uuid.UUID(caller_user_id)
        patches, handles = self._patch_dependencies(new_user_id, caller_personal_org_id)
        with self._enter_all(patches):
            with pytest.raises(HTTPException) as exc_info:
                await provision_user(
                    body=ProvisionUserRequest(email='bob@example.com'),
                    caller_user_id=caller_user_id,
                    target_org_id=caller_personal_org_id,
                )

        assert exc_info.value.status_code == 403
        assert 'personal workspace' in exc_info.value.detail.lower()
        # The rejection must short-circuit before any side effects.
        handles['token_manager'].create_keycloak_user.assert_not_awaited()
        handles['delete_org_cascade'].assert_not_awaited()
        handles['remove_member'].assert_not_awaited()
        handles['token_manager'].delete_keycloak_user.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_keycloak_failure_returns_409(
        self, caller_user_id, target_org_id, new_user_id
    ):
        patches, handles = self._patch_dependencies(
            new_user_id,
            target_org_id,
            keycloak_raises=KeycloakError('user already exists'),
        )
        with self._enter_all(patches):
            with pytest.raises(HTTPException) as exc_info:
                await provision_user(
                    body=ProvisionUserRequest(email='dup@example.com'),
                    caller_user_id=caller_user_id,
                    target_org_id=target_org_id,
                )
        assert exc_info.value.status_code == 409
        # Cleanup should not run if Keycloak creation itself failed —
        # there is nothing to roll back.
        handles['token_manager'].delete_keycloak_user.assert_not_awaited()
        handles['delete_org_cascade'].assert_not_awaited()
        handles['remove_member'].assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rollback_before_user_created_only_cleans_keycloak(
        self, caller_user_id, target_org_id, new_user_id
    ):
        """Failure before ``UserStore.create_user`` only undoes Keycloak.

        The offline-token step runs *before* ``UserStore.create_user``,
        so when it blows up there are no OpenHands DB rows to
        compensate. The rollback must touch only the Keycloak user
        — exercising ``delete_org_cascade`` on a never-created
        personal org would log a misleading "not found".
        """
        patches, handles = self._patch_dependencies(new_user_id, target_org_id)
        # Make the offline-token step blow up after Keycloak succeeded
        # but before any OpenHands DB row was created.
        handles['token_manager'].request_offline_token.side_effect = RuntimeError(
            'boom'
        )

        with self._enter_all(patches):
            with pytest.raises(HTTPException) as exc_info:
                await provision_user(
                    body=ProvisionUserRequest(email='bob@example.com'),
                    caller_user_id=caller_user_id,
                    target_org_id=target_org_id,
                )
        assert exc_info.value.status_code == 500
        # Only the Keycloak user gets removed; no DB rollback.
        handles['token_manager'].delete_keycloak_user.assert_awaited_once_with(
            new_user_id
        )
        handles['delete_org_cascade'].assert_not_awaited()
        handles['remove_member'].assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rollback_when_set_flags_fails(
        self, caller_user_id, target_org_id, new_user_id
    ):
        """``_set_user_provisioned_flags`` failure cascades the personal org.

        ``UserStore.create_user`` succeeded, so the User + personal Org
        + owner OrgMember + default settings now exist. The rollback
        must wipe them via ``delete_org_cascade`` and then delete the
        Keycloak user. No target-org membership was added yet, so
        ``remove_user_from_org`` must not run.
        """
        patches, handles = self._patch_dependencies(new_user_id, target_org_id)
        handles['set_flags'].side_effect = RuntimeError('flag update exploded')

        with self._enter_all(patches):
            with pytest.raises(HTTPException) as exc_info:
                await provision_user(
                    body=ProvisionUserRequest(email='bob@example.com'),
                    caller_user_id=caller_user_id,
                    target_org_id=target_org_id,
                )

        assert exc_info.value.status_code == 500
        handles['remove_member'].assert_not_awaited()
        handles['delete_org_cascade'].assert_awaited_once_with(
            uuid.UUID(new_user_id), requester_user_id=new_user_id
        )
        handles['token_manager'].delete_keycloak_user.assert_awaited_once_with(
            new_user_id
        )

    @pytest.mark.asyncio
    async def test_rollback_when_litellm_integration_fails(
        self, caller_user_id, target_org_id, new_user_id
    ):
        """Failure in ``create_litellm_integration`` cascades personal org.

        We have a User + personal org but no target-org membership
        yet, so the rollback should skip ``remove_user_from_org`` and
        only cascade the personal org before deleting Keycloak.
        """
        patches, handles = self._patch_dependencies(new_user_id, target_org_id)

        with self._enter_all(patches):
            with patch(
                'server.routes.user_provisioning.OrgService.create_litellm_integration',
                new_callable=AsyncMock,
                side_effect=RuntimeError('litellm down'),
            ):
                with pytest.raises(HTTPException) as exc_info:
                    await provision_user(
                        body=ProvisionUserRequest(email='bob@example.com'),
                        caller_user_id=caller_user_id,
                        target_org_id=target_org_id,
                    )

        assert exc_info.value.status_code == 500
        handles['remove_member'].assert_not_awaited()
        handles['delete_org_cascade'].assert_awaited_once_with(
            uuid.UUID(new_user_id), requester_user_id=new_user_id
        )
        handles['token_manager'].delete_keycloak_user.assert_awaited_once_with(
            new_user_id
        )

    @pytest.mark.asyncio
    async def test_rollback_when_add_user_to_org_fails(
        self, caller_user_id, target_org_id, new_user_id
    ):
        """``add_user_to_org`` failure: same shape as litellm failure.

        ``add_user_to_org`` is the call that sets
        ``target_membership_added`` after returning. If it raises
        instead, the membership row was never inserted, so the
        rollback must NOT call ``remove_user_from_org`` — that would
        try to remove a row that does not exist.
        """
        patches, handles = self._patch_dependencies(new_user_id, target_org_id)
        handles['add_user_to_org'].side_effect = RuntimeError('insert exploded')

        with self._enter_all(patches):
            with pytest.raises(HTTPException) as exc_info:
                await provision_user(
                    body=ProvisionUserRequest(email='bob@example.com'),
                    caller_user_id=caller_user_id,
                    target_org_id=target_org_id,
                )

        assert exc_info.value.status_code == 500
        handles['remove_member'].assert_not_awaited()
        handles['delete_org_cascade'].assert_awaited_once_with(
            uuid.UUID(new_user_id), requester_user_id=new_user_id
        )
        handles['token_manager'].delete_keycloak_user.assert_awaited_once_with(
            new_user_id
        )

    @pytest.mark.asyncio
    async def test_rollback_when_api_key_creation_fails(
        self, caller_user_id, target_org_id, new_user_id
    ):
        """API-key failure exercises the *full* rollback path.

        By the time ``create_api_key`` runs, the target-org membership
        has already been inserted, so the rollback must:

        1. Remove the target-org ``OrgMember`` row, *before* calling
           ``delete_org_cascade`` — otherwise the cascade would only
           reassign ``current_org_id`` and leave the User row behind.
        2. Cascade-delete the personal org (User + personal Org +
           settings + personal-org LiteLLM team).
        3. Delete the Keycloak user.

        We also assert step ordering by inspecting ``mock_calls`` on
        a shared parent mock.
        """
        patches, handles = self._patch_dependencies(new_user_id, target_org_id)
        handles['api_key_store'].create_api_key.side_effect = RuntimeError(
            'api key insert exploded'
        )

        # Shared parent mock to assert call ordering across the three
        # rollback collaborators.
        order_tracker = MagicMock()
        order_tracker.attach_mock(handles['remove_member'], 'remove_member')
        order_tracker.attach_mock(handles['delete_org_cascade'], 'delete_org_cascade')
        order_tracker.attach_mock(
            handles['token_manager'].delete_keycloak_user,
            'delete_keycloak_user',
        )

        with self._enter_all(patches):
            with pytest.raises(HTTPException) as exc_info:
                await provision_user(
                    body=ProvisionUserRequest(email='bob@example.com'),
                    caller_user_id=caller_user_id,
                    target_org_id=target_org_id,
                )

        assert exc_info.value.status_code == 500
        handles['remove_member'].assert_awaited_once_with(
            target_org_id, uuid.UUID(new_user_id)
        )
        handles['delete_org_cascade'].assert_awaited_once_with(
            uuid.UUID(new_user_id), requester_user_id=new_user_id
        )
        handles['token_manager'].delete_keycloak_user.assert_awaited_once_with(
            new_user_id
        )
        # Ordering: target-membership ➜ personal-org cascade ➜ Keycloak.
        call_names = [c[0] for c in order_tracker.mock_calls]
        assert call_names == [
            'remove_member',
            'delete_org_cascade',
            'delete_keycloak_user',
        ]

    @pytest.mark.asyncio
    async def test_rollback_swallows_secondary_failures(
        self, caller_user_id, target_org_id, new_user_id
    ):
        """Each cleanup step is wrapped so secondary errors do not mask the original.

        If ``remove_user_from_org`` *and* ``delete_org_cascade`` both
        raise during rollback, the route must still surface the
        original ``HTTPException(500)`` from the provisioning
        failure, and the remaining cleanup steps must keep running.
        """
        patches, handles = self._patch_dependencies(new_user_id, target_org_id)
        handles['api_key_store'].create_api_key.side_effect = RuntimeError(
            'api key insert exploded'
        )
        handles['remove_member'].side_effect = RuntimeError(
            'rollback step 1 also failed'
        )
        handles['delete_org_cascade'].side_effect = RuntimeError(
            'rollback step 2 also failed'
        )

        with self._enter_all(patches):
            with pytest.raises(HTTPException) as exc_info:
                await provision_user(
                    body=ProvisionUserRequest(email='bob@example.com'),
                    caller_user_id=caller_user_id,
                    target_org_id=target_org_id,
                )

        assert exc_info.value.status_code == 500
        # Despite earlier rollback failures, the Keycloak deletion
        # must still be attempted so the upstream identity does not
        # outlive the failed provisioning attempt.
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
        with self._enter_all(patches):
            await provision_user(
                body=ProvisionUserRequest(email='solo@example.com'),
                caller_user_id=caller_user_id,
                target_org_id=personal_org_id,
            )
            handles['add_user_to_org'].assert_not_awaited()

    def test_default_role_is_member(self):
        # Document the policy: provisioned users are not auto-promoted.
        assert DEFAULT_PROVISIONED_ROLE == 'member'
