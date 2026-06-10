import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import SecretStr
from storage.default_org_service import (
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


def _org(name: str = 'Acme') -> Org:
    return Org(id=uuid.uuid4(), name=name)


def _personal_org() -> Org:
    org_id = uuid.uuid4()
    return Org(id=org_id, name=f'user_{org_id}_org')


def test_default_org_config_accepts_one_as_truthy(monkeypatch):
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_ENABLED', '1')
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_NAME', 'Acme')
    monkeypatch.setenv(
        'OPENHANDS_DEFAULT_ORG_OWNER_EMAILS',
        'Owner@Example.com, second@example.com\nthird@example.com',
    )
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_AUTO_ADD_USERS', 'true')

    config = get_default_org_config()

    assert config.enabled is True
    assert config.org_name == 'Acme'
    assert config.owner_emails == frozenset(
        {'owner@example.com', 'second@example.com', 'third@example.com'}
    )
    assert config.auto_add_users is True


@pytest.mark.asyncio
async def test_disabled_default_org_does_nothing(monkeypatch):
    monkeypatch.delenv('OPENHANDS_DEFAULT_ORG_ENABLED', raising=False)
    user = _user('member@example.com')

    with patch(
        'storage.default_org_service.OrgStore.get_org_by_name',
        new_callable=AsyncMock,
    ) as mock_get_org:
        result = await DefaultOrgBootstrapService.apply_for_user(
            user,
            is_new_user=True,
        )

    assert result is user
    mock_get_org.assert_not_called()


@pytest.mark.asyncio
async def test_non_owner_waits_when_no_configured_owner_exists(monkeypatch):
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_ENABLED', 'true')
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_NAME', 'Acme')
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_OWNER_EMAILS', 'owner@example.com')
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_AUTO_ADD_USERS', 'true')
    user = _user('member@example.com')

    with (
        patch(
            'storage.default_org_service.OrgStore.get_org_by_name',
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            'storage.default_org_service.UserStore.get_user_by_email',
            new_callable=AsyncMock,
            return_value=None,
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
async def test_first_configured_owner_creates_default_org_and_switches_new_user(
    monkeypatch,
):
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_ENABLED', 'true')
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_NAME', 'Acme')
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_OWNER_EMAILS', 'owner@example.com')
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_AUTO_ADD_USERS', 'true')
    user = _user('owner@example.com')
    org = _org()
    owner_membership = SimpleNamespace(role_id=1)
    updated_user = _user('owner@example.com')
    updated_user.id = user.id
    updated_user.current_org_id = org.id

    with (
        patch(
            'storage.default_org_service.OrgStore.get_org_by_name',
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            'storage.default_org_service.OrgService.create_org_with_owner',
            new_callable=AsyncMock,
            return_value=org,
        ) as mock_create_org,
        patch(
            'storage.default_org_service.OrgMemberStore.get_org_member',
            new_callable=AsyncMock,
            side_effect=[owner_membership, owner_membership],
        ),
        patch(
            'storage.default_org_service.RoleStore.get_role_by_id',
            new_callable=AsyncMock,
            return_value=Role(id=1, name='owner', rank=1),
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
        name='Acme',
        contact_name='owner@example.com',
        contact_email='owner@example.com',
        user_id=str(user.id),
    )
    mock_update_current_org.assert_awaited_once_with(str(user.id), org.id)
    assert result.current_org_id == org.id


@pytest.mark.asyncio
async def test_existing_owner_user_can_create_org_for_member_auto_join(monkeypatch):
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_ENABLED', 'true')
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_NAME', 'Acme')
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_OWNER_EMAILS', 'owner@example.com')
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_AUTO_ADD_USERS', 'true')
    member = _user('member@example.com')
    owner = _user('owner@example.com')
    org = _org()

    with (
        patch(
            'storage.default_org_service.OrgStore.get_org_by_name',
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            'storage.default_org_service.UserStore.get_user_by_email',
            new_callable=AsyncMock,
            return_value=owner,
        ),
        patch(
            'storage.default_org_service.OrgService.create_org_with_owner',
            new_callable=AsyncMock,
            return_value=org,
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
            'storage.default_org_service.UserStore.get_user_by_id',
            new_callable=AsyncMock,
            return_value=member,
        ),
    ):
        await DefaultOrgBootstrapService.apply_for_user(member, is_new_user=False)

    mock_create_org.assert_awaited_once_with(
        name='Acme',
        contact_name='owner@example.com',
        contact_email='owner@example.com',
        user_id=str(owner.id),
    )
    mock_add_member.assert_awaited_once_with(
        org_id=org.id,
        user_id=member.id,
        role_id=3,
        llm_api_key='test-key',
        status='active',
        agent_settings_diff={},
        conversation_settings_diff={},
    )


@pytest.mark.asyncio
async def test_configured_owner_is_promoted_without_demoting_others(monkeypatch):
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_ENABLED', 'true')
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_NAME', 'Acme')
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_OWNER_EMAILS', 'owner@example.com')
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_AUTO_ADD_USERS', 'false')
    user = _user('owner@example.com')
    org = _org()
    existing_membership = SimpleNamespace(role_id=3)

    with (
        patch(
            'storage.default_org_service.OrgStore.get_org_by_name',
            new_callable=AsyncMock,
            return_value=org,
        ),
        patch(
            'storage.default_org_service.OrgMemberStore.get_org_member',
            new_callable=AsyncMock,
            return_value=existing_membership,
        ),
        patch(
            'storage.default_org_service.RoleStore.get_role_by_id',
            new_callable=AsyncMock,
            return_value=Role(id=3, name='member', rank=3),
        ),
        patch(
            'storage.default_org_service.RoleStore.get_role_by_name',
            new_callable=AsyncMock,
            return_value=Role(id=1, name='owner', rank=1),
        ),
        patch(
            'storage.default_org_service.OrgMemberStore.update_user_role_in_org',
            new_callable=AsyncMock,
        ) as mock_update_role,
        patch(
            'storage.default_org_service.OrgMemberStore.add_user_to_org',
            new_callable=AsyncMock,
        ) as mock_add_member,
        patch(
            'storage.default_org_service.UserStore.get_user_by_id',
            new_callable=AsyncMock,
            return_value=user,
        ),
    ):
        await DefaultOrgBootstrapService.apply_for_user(user, is_new_user=False)

    mock_update_role.assert_awaited_once_with(
        org_id=org.id,
        user_id=user.id,
        role_id=1,
        status='active',
    )
    mock_add_member.assert_not_called()


@pytest.mark.asyncio
async def test_existing_personal_workspace_org_is_not_adopted(monkeypatch):
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_ENABLED', 'true')
    personal_org = _personal_org()
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_NAME', personal_org.name)
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_OWNER_EMAILS', 'owner@example.com')
    monkeypatch.setenv('OPENHANDS_DEFAULT_ORG_AUTO_ADD_USERS', 'true')
    user = _user('owner@example.com')

    with (
        patch(
            'storage.default_org_service.OrgStore.get_org_by_name',
            new_callable=AsyncMock,
            return_value=personal_org,
        ),
        patch(
            'storage.default_org_service.OrgMemberStore.get_org_member',
            new_callable=AsyncMock,
        ) as mock_get_member,
        patch(
            'storage.default_org_service.OrgService.create_org_with_owner',
            new_callable=AsyncMock,
        ) as mock_create_org,
    ):
        result = await DefaultOrgBootstrapService.apply_for_user(
            user,
            is_new_user=True,
        )

    assert result is user
    mock_get_member.assert_not_called()
    mock_create_org.assert_not_called()
