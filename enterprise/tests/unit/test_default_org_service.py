import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import SecretStr
from server.routes.org_models import OrgNameExistsError
from storage.default_org_service import (
    DEFAULT_ORG_NAME,
    DefaultOrgBootstrapService,
    get_default_org_config,
)
from storage.org import Org
from storage.role import Role
from storage.user import User


def _settings(api_key: str = 'test-key'):
    return SimpleNamespace(
        agent_settings=SimpleNamespace(
            llm=SimpleNamespace(api_key=SecretStr(api_key)),
        )
    )


def _user(email: str) -> User:
    user_id = uuid.uuid4()
    return User(id=user_id, email=email, current_org_id=user_id)


def _org(name: str = DEFAULT_ORG_NAME) -> Org:
    return Org(id=uuid.uuid4(), name=name)


def test_default_org_config_defaults_org_name(monkeypatch):
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_ENABLED', '1')
    monkeypatch.delenv('OPENHANDS_DEFAULT_ORG_NAME', raising=False)
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_AUTO_ADD_USERS', 'true')

    config = get_default_org_config()

    assert config.enabled is True
    assert config.org_name == DEFAULT_ORG_NAME
    assert config.auto_add_users is True
    assert config.hide_personal_workspaces is False


def test_default_org_config_allows_org_name_override(monkeypatch):
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_ENABLED', '1')
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_NAME', 'Acme')

    config = get_default_org_config()

    assert config.org_name == 'Acme'


def test_default_org_config_reads_hide_personal_workspaces(monkeypatch):
    monkeypatch.setenv('HIDE_PERSONAL_WORKSPACES', 'true')

    config = get_default_org_config()

    assert config.hide_personal_workspaces is True


@pytest.mark.asyncio
async def test_disabled_default_org_does_nothing(monkeypatch):
    monkeypatch.delenv('OPENHANDS_DEFAULT_ORG_ENABLED', raising=False)
    user = _user('member@example.com')

    with patch(
        'storage.default_org_service.OrgStore.get_default_org',
        new_callable=AsyncMock,
    ) as mock_get_org:
        result = await DefaultOrgBootstrapService.apply_for_user(
            user,
            is_new_user=True,
        )

    assert result is user
    mock_get_org.assert_not_called()


@pytest.mark.asyncio
async def test_first_user_creates_default_org_and_becomes_owner(monkeypatch):
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_ENABLED', 'true')
    monkeypatch.delenv('OPENHANDS_DEFAULT_ORG_NAME', raising=False)
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_AUTO_ADD_USERS', 'true')
    user = _user('first@example.com')
    org = _org()
    owner_membership = SimpleNamespace(role_id=1)
    updated_user = _user('first@example.com')
    updated_user.id = user.id
    updated_user.current_org_id = org.id

    with (
        patch(
            'storage.default_org_service.OrgStore.get_default_org',
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            'storage.default_org_service.OrgStore.list_team_orgs',
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            'storage.default_org_service.OrgService.create_org_with_owner',
            new_callable=AsyncMock,
            return_value=org,
        ) as mock_create_org,
        patch(
            'storage.default_org_service.OrgStore.mark_org_as_default',
            new_callable=AsyncMock,
            return_value=org,
        ) as mock_mark_default,
        patch(
            'storage.default_org_service.OrgMemberStore.get_org_member',
            new_callable=AsyncMock,
            return_value=owner_membership,
        ),
        patch(
            'storage.default_org_service.UserStore.update_current_org',
            new_callable=AsyncMock,
            return_value=updated_user,
        ) as mock_update_current_org,
        patch(
            'storage.default_org_service.UserStore.get_user_by_id',
            new_callable=AsyncMock,
            return_value=updated_user,
        ),
    ):
        result = await DefaultOrgBootstrapService.apply_for_user(
            user,
            is_new_user=True,
        )

    mock_create_org.assert_awaited_once_with(
        name=DEFAULT_ORG_NAME,
        contact_name='first@example.com',
        contact_email='first@example.com',
        user_id=str(user.id),
    )
    mock_mark_default.assert_awaited_once_with(org.id)
    mock_update_current_org.assert_awaited_once_with(str(user.id), org.id)
    assert result.current_org_id == org.id


@pytest.mark.asyncio
async def test_first_user_creates_org_even_with_auto_add_disabled(monkeypatch):
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_ENABLED', 'true')
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_AUTO_ADD_USERS', 'false')
    user = _user('first@example.com')
    org = _org()
    owner_membership = SimpleNamespace(role_id=1)

    with (
        patch(
            'storage.default_org_service.OrgStore.get_default_org',
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            'storage.default_org_service.OrgStore.list_team_orgs',
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            'storage.default_org_service.OrgService.create_org_with_owner',
            new_callable=AsyncMock,
            return_value=org,
        ) as mock_create_org,
        patch(
            'storage.default_org_service.OrgStore.mark_org_as_default',
            new_callable=AsyncMock,
            return_value=org,
        ),
        patch(
            'storage.default_org_service.OrgMemberStore.get_org_member',
            new_callable=AsyncMock,
            return_value=owner_membership,
        ),
        patch(
            'storage.default_org_service.UserStore.update_current_org',
            new_callable=AsyncMock,
            return_value=user,
        ) as mock_update_current_org,
        patch(
            'storage.default_org_service.UserStore.get_user_by_id',
            new_callable=AsyncMock,
            return_value=user,
        ),
    ):
        await DefaultOrgBootstrapService.apply_for_user(user, is_new_user=False)

    # Auto-add gates *other* users; the creating user always owns the org.
    mock_create_org.assert_awaited_once()
    mock_update_current_org.assert_awaited_once_with(str(user.id), org.id)


@pytest.mark.asyncio
async def test_sole_existing_team_org_is_adopted(monkeypatch):
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_ENABLED', 'true')
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_AUTO_ADD_USERS', 'true')
    member = _user('member@example.com')
    org = _org('Acme')

    with (
        patch(
            'storage.default_org_service.OrgStore.get_default_org',
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            'storage.default_org_service.OrgStore.list_team_orgs',
            new_callable=AsyncMock,
            return_value=[org],
        ),
        patch(
            'storage.default_org_service.OrgStore.mark_org_as_default',
            new_callable=AsyncMock,
            return_value=org,
        ) as mock_mark_default,
        patch(
            'storage.default_org_service.OrgService.create_org_with_owner',
            new_callable=AsyncMock,
        ) as mock_create_org,
        patch(
            'storage.default_org_service.OrgMemberStore.get_org_member',
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            'storage.default_org_service.RoleStore.get_role_by_name',
            new_callable=AsyncMock,
            return_value=Role(id=3, name='member', rank=3),
        ),
        patch(
            'storage.default_org_service.OrgService.create_litellm_integration',
            new_callable=AsyncMock,
            return_value=_settings(),
        ),
        patch(
            'storage.default_org_service.OrgMemberStore.add_user_to_org',
            new_callable=AsyncMock,
        ) as mock_add_member,
        patch(
            'storage.default_org_service.UserStore.update_current_org',
            new_callable=AsyncMock,
            return_value=member,
        ),
        patch(
            'storage.default_org_service.UserStore.get_user_by_id',
            new_callable=AsyncMock,
            return_value=member,
        ),
    ):
        await DefaultOrgBootstrapService.apply_for_user(member, is_new_user=False)

    mock_mark_default.assert_awaited_once_with(org.id)
    mock_create_org.assert_not_called()
    mock_add_member.assert_awaited_once()


@pytest.mark.asyncio
async def test_multiple_unflagged_team_orgs_skip_bootstrap(monkeypatch):
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_ENABLED', 'true')
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_AUTO_ADD_USERS', 'true')
    user = _user('member@example.com')

    with (
        patch(
            'storage.default_org_service.OrgStore.get_default_org',
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            'storage.default_org_service.OrgStore.list_team_orgs',
            new_callable=AsyncMock,
            return_value=[_org('A'), _org('B')],
        ),
        patch(
            'storage.default_org_service.OrgService.create_org_with_owner',
            new_callable=AsyncMock,
        ) as mock_create_org,
        patch(
            'storage.default_org_service.OrgMemberStore.add_user_to_org',
            new_callable=AsyncMock,
        ) as mock_add_member,
    ):
        result = await DefaultOrgBootstrapService.apply_for_user(
            user,
            is_new_user=True,
        )

    assert result is user
    mock_create_org.assert_not_called()
    mock_add_member.assert_not_called()


@pytest.mark.asyncio
async def test_concurrent_creation_race_adopts_winner(monkeypatch):
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_ENABLED', 'true')
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_AUTO_ADD_USERS', 'true')
    user = _user('second@example.com')
    org = _org()
    existing_membership = SimpleNamespace(role_id=3)

    with (
        patch(
            'storage.default_org_service.OrgStore.get_default_org',
            new_callable=AsyncMock,
            # First lookup misses; after the create collides, the winner's
            # flagged org is found.
            side_effect=[None, org],
        ),
        patch(
            'storage.default_org_service.OrgStore.list_team_orgs',
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            'storage.default_org_service.OrgService.create_org_with_owner',
            new_callable=AsyncMock,
            side_effect=OrgNameExistsError('exists'),
        ),
        patch(
            'storage.default_org_service.OrgMemberStore.get_org_member',
            new_callable=AsyncMock,
            return_value=existing_membership,
        ),
        patch(
            'storage.default_org_service.UserStore.update_current_org',
            new_callable=AsyncMock,
        ) as mock_update_current_org,
        patch(
            'storage.default_org_service.UserStore.get_user_by_id',
            new_callable=AsyncMock,
            return_value=user,
        ),
    ):
        await DefaultOrgBootstrapService.apply_for_user(user, is_new_user=False)

    # The loser of the race is already a member (or treated as existing);
    # no new org is created and no workspace move is forced.
    mock_update_current_org.assert_not_called()


@pytest.mark.asyncio
async def test_existing_user_auto_added_is_moved_into_default_org(monkeypatch):
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_ENABLED', 'true')
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_AUTO_ADD_USERS', 'true')
    member = _user('member@example.com')
    org = _org()

    with (
        patch(
            'storage.default_org_service.OrgStore.get_default_org',
            new_callable=AsyncMock,
            return_value=org,
        ),
        patch(
            'storage.default_org_service.OrgMemberStore.get_org_member',
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            'storage.default_org_service.RoleStore.get_role_by_name',
            new_callable=AsyncMock,
            return_value=Role(id=3, name='member', rank=3),
        ),
        patch(
            'storage.default_org_service.OrgService.create_litellm_integration',
            new_callable=AsyncMock,
            return_value=_settings(),
        ),
        patch(
            'storage.default_org_service.OrgMemberStore.add_user_to_org',
            new_callable=AsyncMock,
        ) as mock_add_member,
        patch(
            'storage.default_org_service.UserStore.update_current_org',
            new_callable=AsyncMock,
            return_value=member,
        ) as mock_update_current_org,
        patch(
            'storage.default_org_service.UserStore.get_user_by_id',
            new_callable=AsyncMock,
            return_value=member,
        ),
    ):
        await DefaultOrgBootstrapService.apply_for_user(member, is_new_user=False)

    mock_add_member.assert_awaited_once_with(
        org_id=org.id,
        user_id=member.id,
        role_id=3,
        llm_api_key='test-key',
        status='active',
        agent_settings_diff={},
        conversation_settings_diff={},
    )
    mock_update_current_org.assert_awaited_once_with(str(member.id), org.id)


@pytest.mark.asyncio
async def test_auto_add_disabled_leaves_later_users_alone(monkeypatch):
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_ENABLED', 'true')
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_AUTO_ADD_USERS', 'false')
    user = _user('member@example.com')
    org = _org()

    with (
        patch(
            'storage.default_org_service.OrgStore.get_default_org',
            new_callable=AsyncMock,
            return_value=org,
        ),
        patch(
            'storage.default_org_service.OrgMemberStore.get_org_member',
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            'storage.default_org_service.OrgMemberStore.add_user_to_org',
            new_callable=AsyncMock,
        ) as mock_add_member,
        patch(
            'storage.default_org_service.UserStore.update_current_org',
            new_callable=AsyncMock,
        ) as mock_update_current_org,
        patch(
            'storage.default_org_service.UserStore.get_user_by_id',
            new_callable=AsyncMock,
            return_value=user,
        ),
    ):
        result = await DefaultOrgBootstrapService.apply_for_user(
            user,
            is_new_user=False,
        )

    assert result is user
    mock_add_member.assert_not_called()
    mock_update_current_org.assert_not_called()


@pytest.mark.asyncio
async def test_existing_member_login_keeps_current_workspace(monkeypatch):
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_ENABLED', 'true')
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_AUTO_ADD_USERS', 'true')
    member = _user('member@example.com')
    org = _org()
    existing_membership = SimpleNamespace(role_id=3)

    with (
        patch(
            'storage.default_org_service.OrgStore.get_default_org',
            new_callable=AsyncMock,
            return_value=org,
        ),
        patch(
            'storage.default_org_service.OrgMemberStore.get_org_member',
            new_callable=AsyncMock,
            return_value=existing_membership,
        ),
        patch(
            'storage.default_org_service.OrgMemberStore.add_user_to_org',
            new_callable=AsyncMock,
        ) as mock_add_member,
        patch(
            'storage.default_org_service.UserStore.update_current_org',
            new_callable=AsyncMock,
        ) as mock_update_current_org,
        patch(
            'storage.default_org_service.UserStore.get_user_by_id',
            new_callable=AsyncMock,
            return_value=member,
        ),
    ):
        await DefaultOrgBootstrapService.apply_for_user(member, is_new_user=False)

    # Already a member: the one-time move happened in the past; the user's
    # current workspace choice (e.g. a deliberate switch back to personal)
    # is preserved on later logins.
    mock_add_member.assert_not_called()
    mock_update_current_org.assert_not_called()


@pytest.mark.asyncio
async def test_hide_personal_workspaces_moves_member_parked_on_personal(monkeypatch):
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_ENABLED', 'true')
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_AUTO_ADD_USERS', 'true')
    monkeypatch.setenv('HIDE_PERSONAL_WORKSPACES', 'true')
    # _user() parks the user on their personal org (current_org_id == id)
    member = _user('member@example.com')
    org = _org()
    existing_membership = SimpleNamespace(role_id=3)

    with (
        patch(
            'storage.default_org_service.OrgStore.get_default_org',
            new_callable=AsyncMock,
            return_value=org,
        ),
        patch(
            'storage.default_org_service.OrgMemberStore.get_org_member',
            new_callable=AsyncMock,
            return_value=existing_membership,
        ),
        patch(
            'storage.default_org_service.UserStore.update_current_org',
            new_callable=AsyncMock,
            return_value=member,
        ) as mock_update_current_org,
        patch(
            'storage.default_org_service.UserStore.get_user_by_id',
            new_callable=AsyncMock,
            return_value=member,
        ),
    ):
        await DefaultOrgBootstrapService.apply_for_user(member, is_new_user=False)

    # Personal workspaces are hidden, so a member parked on their personal
    # org is moved into the default org on every login.
    mock_update_current_org.assert_awaited_once_with(str(member.id), org.id)


@pytest.mark.asyncio
async def test_hide_personal_workspaces_does_not_move_user_in_another_team_org(
    monkeypatch,
):
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_ENABLED', 'true')
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_AUTO_ADD_USERS', 'true')
    monkeypatch.setenv('HIDE_PERSONAL_WORKSPACES', 'true')
    member = _user('member@example.com')
    # The user works in some other (team) org — not their personal workspace.
    member.current_org_id = uuid.uuid4()
    org = _org()
    existing_membership = SimpleNamespace(role_id=3)

    with (
        patch(
            'storage.default_org_service.OrgStore.get_default_org',
            new_callable=AsyncMock,
            return_value=org,
        ),
        patch(
            'storage.default_org_service.OrgMemberStore.get_org_member',
            new_callable=AsyncMock,
            return_value=existing_membership,
        ),
        patch(
            'storage.default_org_service.UserStore.update_current_org',
            new_callable=AsyncMock,
        ) as mock_update_current_org,
        patch(
            'storage.default_org_service.UserStore.get_user_by_id',
            new_callable=AsyncMock,
            return_value=member,
        ),
    ):
        await DefaultOrgBootstrapService.apply_for_user(member, is_new_user=False)

    mock_update_current_org.assert_not_called()


@pytest.mark.asyncio
async def test_hide_personal_workspaces_leaves_non_member_on_personal(monkeypatch):
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_ENABLED', 'true')
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_AUTO_ADD_USERS', 'false')
    monkeypatch.setenv('HIDE_PERSONAL_WORKSPACES', 'true')
    member = _user('member@example.com')
    org = _org()

    with (
        patch(
            'storage.default_org_service.OrgStore.get_default_org',
            new_callable=AsyncMock,
            return_value=org,
        ),
        patch(
            'storage.default_org_service.OrgMemberStore.get_org_member',
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            'storage.default_org_service.UserStore.update_current_org',
            new_callable=AsyncMock,
        ) as mock_update_current_org,
        patch(
            'storage.default_org_service.UserStore.get_user_by_id',
            new_callable=AsyncMock,
            return_value=member,
        ),
    ):
        await DefaultOrgBootstrapService.apply_for_user(member, is_new_user=False)

    # Not a member of the default org (auto-add off): never strand the user
    # with zero workspaces — they stay on personal and the UI keeps showing it.
    mock_update_current_org.assert_not_called()
