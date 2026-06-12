import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from openhands.app_server.settings.settings_models import Settings
from openhands.app_server.settings.settings_models import Settings as DataSettings


def _agent_value(settings: Settings, key: str):
    """Navigate into settings.agent_settings using a dot-separated key."""
    obj = settings.agent_settings
    for part in key.split('.'):
        obj = getattr(obj, part)
    return obj


def _secret_value(settings: Settings, key: str):
    """Navigate into settings.agent_settings and unwrap SecretStr values."""
    secret = _agent_value(settings, key)
    return secret.get_secret_value() if secret else None


def _make_settings(
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    max_iterations: int | None = None,
    agent: str | None = None,
    language: str | None = None,
    **extra_agent: object,
) -> DataSettings:
    """Build a DataSettings with diff-only nested settings payloads."""
    top_level: dict = {}
    if language is not None:
        top_level['language'] = language
    s = DataSettings(**top_level)
    llm: dict = {}
    if model is not None:
        llm['model'] = model
    if base_url is not None:
        llm['base_url'] = base_url
    if api_key is not None:
        llm['api_key'] = api_key
    agent_settings_diff: dict = {}
    if agent is not None:
        agent_settings_diff['agent'] = agent
    if llm:
        agent_settings_diff['llm'] = llm
    agent_settings_diff.update(extra_agent)
    payload: dict = {}
    if agent_settings_diff:
        payload['agent_settings_diff'] = agent_settings_diff
    conversation_settings_diff: dict = {}
    if max_iterations is not None:
        conversation_settings_diff['max_iterations'] = max_iterations
    if conversation_settings_diff:
        payload['conversation_settings_diff'] = conversation_settings_diff
    if payload:
        s.update(payload)
    return s


# Mock the database module before importing
with patch('storage.database.a_session_maker'):
    from server.constants import (
        LITE_LLM_API_URL,
    )
    from storage.encrypt_utils import decrypt_legacy_value, encrypt_legacy_value
    from storage.saas_settings_store import SaasSettingsStore
    from storage.user_settings import UserSettings


def test_member_settings_persist_full_effective_agent_settings():
    settings = Settings()
    settings.update(
        {
            'agent_settings_diff': {
                'agent': 'CodeActAgent',
                'llm': {
                    'model': 'anthropic/claude-sonnet-4-5-20250929',
                    'base_url': 'https://api.example.com',
                },
                'condenser': {
                    'enabled': False,
                    'max_size': 128,
                },
            },
            'conversation_settings_diff': {
                'max_iterations': 42,
                'confirmation_mode': True,
                'security_analyzer': 'llm',
            },
        }
    )

    agent = settings.agent_settings
    assert agent.agent == 'CodeActAgent'
    assert agent.llm.model == 'anthropic/claude-sonnet-4-5-20250929'
    assert agent.llm.base_url == 'https://api.example.com'
    assert agent.condenser.enabled is False
    assert agent.condenser.max_size == 128

    # Conversation settings live on the Settings object, not in agent_settings
    assert settings.conversation_settings.max_iterations == 42
    assert settings.conversation_settings.confirmation_mode is True
    assert settings.conversation_settings.security_analyzer == 'llm'


@pytest.fixture
def settings_store(async_session_maker):
    store = SaasSettingsStore('5594c7b6-f959-4b81-92e9-b09c206f5081')
    store.a_session_maker = async_session_maker

    # Patch the load method to read from UserSettings table directly (for testing)
    async def patched_load():
        async with store.a_session_maker() as session:
            from sqlalchemy import select

            result = await session.execute(
                select(UserSettings).filter(
                    UserSettings.keycloak_user_id == store.user_id
                )
            )
            user_settings = result.scalars().first()
            if not user_settings:
                # Return default settings
                return _make_settings(
                    model='anthropic/claude-sonnet-4-5-20250929',
                    api_key='test_api_key',
                    base_url='http://test.url',
                    agent='CodeActAgent',
                    language='en',
                )

            # Decrypt and reconstruct Settings
            agent_dict = user_settings.agent_settings or {}
            # Decrypt llm_api_key into agent_settings.llm.api_key
            if user_settings.llm_api_key:
                decrypted = decrypt_legacy_value(user_settings.llm_api_key)
                agent_dict.setdefault('llm', {})['api_key'] = decrypted

            settings = Settings(
                language=user_settings.language,
                email='test@example.com',
                email_verified=True,
            )
            payload: dict = {}
            if agent_dict:
                payload['agent_settings_diff'] = agent_dict
            if payload:
                settings.update(payload)
            return settings

    # Patch the store method to write to UserSettings table directly (for testing)
    async def patched_store(item):
        if item:
            # Make a copy of the item without email and email_verified
            item_dict = item.model_dump(context={'expose_secrets': True})
            item_dict['llm_api_key'] = _secret_value(item, 'llm.api_key')
            if 'email' in item_dict:
                del item_dict['email']
            if 'email_verified' in item_dict:
                del item_dict['email_verified']
            if 'secrets_store' in item_dict:
                del item_dict['secrets_store']
            if 'llm_profiles' in item_dict:
                del item_dict['llm_profiles']

            # Encrypt the data before storing
            for key in ('llm_api_key', 'search_api_key', 'sandbox_api_key'):
                value = item_dict.get(key)
                if value is not None:
                    item_dict[key] = encrypt_legacy_value(value)
            item_dict['agent_settings'] = item.agent_settings.model_dump(
                mode='json', exclude_none=True
            )

            # Continue with the original implementation
            from sqlalchemy import select

            async with store.a_session_maker() as session:
                result = await session.execute(
                    select(UserSettings).filter(
                        UserSettings.keycloak_user_id == store.user_id
                    )
                )
                existing = result.scalars().first()

                if existing:
                    # Update existing entry
                    for key, value in item_dict.items():
                        if key in existing.__class__.__table__.columns:
                            setattr(existing, key, value)
                    await session.merge(existing)
                else:
                    item_dict['keycloak_user_id'] = store.user_id
                    settings = UserSettings(**item_dict)
                    session.add(settings)
                await session.commit()

    # Replace the methods with our patched versions
    store.store = patched_store
    store.load = patched_load
    return store


@pytest.mark.asyncio
async def test_store_and_load_keycloak_user(settings_store):
    # Set a UUID-like Keycloak user ID
    settings_store.user_id = '550e8400-e29b-41d4-a716-446655440000'
    settings = DataSettings(
        email='test@example.com',
        email_verified=True,
    )
    settings.update(
        {
            'agent_settings_diff': {
                'agent': 'smith',
                'llm': {
                    'model': 'anthropic/claude-sonnet-4-5-20250929',
                    'api_key': 'secret_key',
                    'base_url': LITE_LLM_API_URL,
                },
                'verification': {
                    'critic_mode': 'all_actions',
                    'critic_enabled': True,
                },
            },
        }
    )

    await settings_store.store(settings)

    # Load and verify settings
    loaded_settings = await settings_store.load()
    assert loaded_settings is not None
    assert _agent_value(loaded_settings, 'verification.critic_mode') == 'all_actions'
    assert _agent_value(loaded_settings, 'verification.critic_enabled') is True
    assert _secret_value(loaded_settings, 'llm.api_key') == 'secret_key'
    assert _agent_value(loaded_settings, 'agent') == 'smith'

    # Verify it was stored in user_settings table with keycloak_user_id
    from sqlalchemy import select

    async with settings_store.a_session_maker() as session:
        result = await session.execute(
            select(UserSettings).filter(
                UserSettings.keycloak_user_id == '550e8400-e29b-41d4-a716-446655440000'
            )
        )
        stored = result.scalars().first()
        assert stored is not None
        assert stored.agent_settings['agent'] == 'smith'


@pytest.mark.asyncio
async def test_load_returns_default_when_not_found(settings_store, async_session_maker):
    file_store = MagicMock()
    file_store.read.side_effect = FileNotFoundError()

    with (
        patch('storage.saas_settings_store.a_session_maker', async_session_maker),
    ):
        loaded_settings = await settings_store.load()
        assert loaded_settings is not None
        assert loaded_settings.language == 'en'
        assert _agent_value(loaded_settings, 'agent') == 'CodeActAgent'
        assert _secret_value(loaded_settings, 'llm.api_key') == 'test_api_key'
        assert _agent_value(loaded_settings, 'llm.base_url') == 'http://test.url'


@pytest.mark.asyncio
async def test_encryption(settings_store):
    settings_store.user_id = '5594c7b6-f959-4b81-92e9-b09c206f5081'  # GitHub user ID
    settings = DataSettings(
        email='test@example.com',
        email_verified=True,
    )
    settings.update(
        {
            'agent_settings_diff': {
                'agent': 'smith',
                'llm': {
                    'model': 'anthropic/claude-sonnet-4-5-20250929',
                    'api_key': 'secret_key',
                    'base_url': LITE_LLM_API_URL,
                },
            },
        }
    )
    await settings_store.store(settings)
    from sqlalchemy import select

    async with settings_store.a_session_maker() as session:
        result = await session.execute(
            select(UserSettings).filter(
                UserSettings.keycloak_user_id == '5594c7b6-f959-4b81-92e9-b09c206f5081'
            )
        )
        stored = result.scalars().first()
        # The stored key should be encrypted
        assert stored.llm_api_key != 'secret_key'
        # But we should be able to decrypt it when loading
        loaded_settings = await settings_store.load()
        assert _secret_value(loaded_settings, 'llm.api_key') == 'secret_key'


@pytest.mark.asyncio
async def test_ensure_api_key_keeps_valid_key():
    """When the existing key is valid, it should be kept unchanged."""
    store = SaasSettingsStore('test-user-id-123')
    existing_key = 'sk-existing-key'
    item = _make_settings(model='openhands/gpt-4', api_key=existing_key)

    with patch(
        'storage.saas_settings_store.LiteLlmManager.verify_existing_key',
        new_callable=AsyncMock,
        return_value=True,
    ):
        await store._ensure_api_key(item, 'org-123', openhands_type=True)

        # Key should remain unchanged when it's valid
        assert _secret_value(item, 'llm.api_key') is not None
        assert _secret_value(item, 'llm.api_key') == existing_key


@pytest.mark.asyncio
async def test_ensure_api_key_generates_new_key_when_verification_fails():
    """When verification fails, a new key should be generated."""
    store = SaasSettingsStore('test-user-id-123')
    new_key = 'sk-new-key'
    item = _make_settings(model='openhands/gpt-4', api_key='sk-invalid-key')

    with (
        patch(
            'storage.saas_settings_store.LiteLlmManager.verify_existing_key',
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch(
            'storage.saas_settings_store.LiteLlmManager.generate_key',
            new_callable=AsyncMock,
            return_value=new_key,
        ),
    ):
        await store._ensure_api_key(item, 'org-123', openhands_type=True)

        assert _secret_value(item, 'llm.api_key') is not None
        assert _secret_value(item, 'llm.api_key') == new_key


@pytest.fixture
def org_with_multiple_members_fixture(session_maker):
    """Set up an organization with multiple members for testing LLM settings propagation.

    Uses sync session to avoid UUID conversion issues with async SQLite.
    """
    from storage.encrypt_utils import decrypt_value
    from storage.org import Org
    from storage.org_member import OrgMember
    from storage.role import Role
    from storage.user import User

    # Use realistic UUIDs that work well with SQLite
    org_id = uuid.UUID('5594c7b6-f959-4b81-92e9-b09c206f5081')
    admin_user_id = uuid.UUID('5594c7b6-f959-4b81-92e9-b09c206f5082')
    member1_user_id = uuid.UUID('5594c7b6-f959-4b81-92e9-b09c206f5083')
    member2_user_id = uuid.UUID('5594c7b6-f959-4b81-92e9-b09c206f5084')

    with session_maker() as session:
        # Create role
        role = Role(id=10, name='member', rank=3)
        session.add(role)

        # Create org
        org = Org(
            id=org_id,
            name='test-org',
            org_version=1,
            enable_proactive_conversation_starters=True,
        )
        session.add(org)

        # Create users
        admin_user = User(
            id=admin_user_id, current_org_id=org_id, user_consents_to_analytics=True
        )
        session.add(admin_user)

        member1_user = User(
            id=member1_user_id, current_org_id=org_id, user_consents_to_analytics=True
        )
        session.add(member1_user)

        member2_user = User(
            id=member2_user_id, current_org_id=org_id, user_consents_to_analytics=True
        )
        session.add(member2_user)

        # Create org members with DIFFERENT initial LLM settings
        admin_member = OrgMember(
            org_id=org_id,
            user_id=admin_user_id,
            role_id=10,
            llm_api_key='admin-initial-key',
            agent_settings_diff={
                'llm': {'model': 'old-model-v1', 'base_url': 'http://old-url-1.com'},
            },
            conversation_settings_diff={'max_iterations': 10},
            status='active',
        )
        session.add(admin_member)

        member1 = OrgMember(
            org_id=org_id,
            user_id=member1_user_id,
            role_id=10,
            llm_api_key='member1-initial-key',
            agent_settings_diff={
                'llm': {'model': 'old-model-v2', 'base_url': 'http://old-url-2.com'},
            },
            conversation_settings_diff={'max_iterations': 20},
            status='active',
        )
        session.add(member1)

        member2 = OrgMember(
            org_id=org_id,
            user_id=member2_user_id,
            role_id=10,
            llm_api_key='member2-initial-key',
            agent_settings_diff={
                'llm': {'model': 'old-model-v3', 'base_url': 'http://old-url-3.com'},
            },
            conversation_settings_diff={'max_iterations': 30},
            status='active',
        )
        session.add(member2)

        session.commit()

    return {
        'org_id': org_id,
        'admin_user_id': admin_user_id,
        'member1_user_id': member1_user_id,
        'member2_user_id': member2_user_id,
        'decrypt_value': decrypt_value,
    }


@pytest.mark.asyncio
async def test_load_canonicalizes_legacy_litellm_proxy_active_llm(
    async_session_maker, org_with_multiple_members_fixture
):
    from sqlalchemy import update
    from storage.org_member import OrgMember
    from storage.user import User

    fixture = org_with_multiple_members_fixture
    admin_user_id = fixture['admin_user_id']
    org_id = fixture['org_id']

    async with async_session_maker() as session:
        await session.execute(
            update(OrgMember)
            .where(OrgMember.org_id == org_id, OrgMember.user_id == admin_user_id)
            .values(
                agent_settings_diff={
                    'llm': {
                        'model': 'litellm_proxy/claude-opus-4-8',
                        'base_url': LITE_LLM_API_URL,
                    },
                }
            )
        )
        await session.execute(
            update(User)
            .where(User.id == admin_user_id)
            .values(enable_sound_notifications=False)
        )
        await session.commit()

    store = SaasSettingsStore(str(admin_user_id))
    with (
        patch('storage.saas_settings_store.a_session_maker', async_session_maker),
        patch('storage.user_store.a_session_maker', async_session_maker),
        patch('storage.org_store.a_session_maker', async_session_maker),
    ):
        loaded = await store.load()

    assert loaded is not None
    assert loaded.agent_settings.llm.model == 'openhands/claude-opus-4-8'
    assert loaded.agent_settings.llm.base_url is None


@pytest.mark.asyncio
async def test_load_canonicalizes_legacy_litellm_proxy_llm_profiles(
    async_session_maker, org_with_multiple_members_fixture
):
    from sqlalchemy import update
    from storage.org import Org
    from storage.user import User

    fixture = org_with_multiple_members_fixture
    admin_user_id = fixture['admin_user_id']
    org_id = fixture['org_id']

    async with async_session_maker() as session:
        await session.execute(
            update(Org)
            .where(Org.id == org_id)
            .values(
                llm_profiles={
                    'profiles': {
                        'legacy': {
                            'model': 'litellm_proxy/claude-opus-4-8',
                            'base_url': LITE_LLM_API_URL,
                        },
                        'custom': {
                            'model': 'litellm_proxy/custom-alias',
                            'base_url': LITE_LLM_API_URL,
                        },
                    },
                    'active': 'legacy',
                }
            )
        )
        await session.execute(
            update(User)
            .where(User.id == admin_user_id)
            .values(enable_sound_notifications=False)
        )
        await session.commit()

    store = SaasSettingsStore(str(admin_user_id))
    with (
        patch('storage.saas_settings_store.a_session_maker', async_session_maker),
        patch('storage.user_store.a_session_maker', async_session_maker),
        patch('storage.org_store.a_session_maker', async_session_maker),
    ):
        loaded = await store.load()

    assert loaded is not None
    assert loaded.llm_profiles.active == 'legacy'

    legacy = loaded.llm_profiles.require('legacy')
    assert legacy.model == 'openhands/claude-opus-4-8'
    assert legacy.base_url is None

    custom = loaded.llm_profiles.require('custom')
    assert custom.model == 'litellm_proxy/custom-alias'
    assert custom.base_url == LITE_LLM_API_URL


@pytest.mark.asyncio
async def test_store_updates_org_defaults_and_all_members_for_shared_keys(
    session_maker, async_session_maker, org_with_multiple_members_fixture
):
    """External provider keys should still sync as an org-wide shared snapshot."""
    from sqlalchemy import select
    from storage.org import Org
    from storage.org_member import OrgMember

    fixture = org_with_multiple_members_fixture
    org_id = fixture['org_id']
    decrypt_value = fixture['decrypt_value']

    store = SaasSettingsStore(str(fixture['admin_user_id']))
    new_settings = _make_settings(
        model='anthropic/claude-sonnet-4',
        base_url='https://api.anthropic.com/v1',
        max_iterations=100,
        api_key='shared-external-api-key',
    )

    with patch('storage.saas_settings_store.a_session_maker', async_session_maker):
        await store.store(new_settings)

    with session_maker() as session:
        org = session.execute(select(Org).where(Org.id == org_id)).scalars().first()
        assert org is not None
        assert org.agent_settings['llm']['model'] == 'anthropic/claude-sonnet-4'
        assert org.agent_settings['llm']['base_url'] == 'https://api.anthropic.com/v1'
        assert org.conversation_settings['max_iterations'] == 100

        members = {
            str(member.user_id): member
            for member in session.execute(
                select(OrgMember).where(OrgMember.org_id == org_id)
            )
            .scalars()
            .all()
        }
        assert len(members) == 3

        for member in members.values():
            assert (
                member.agent_settings_diff['llm']['model']
                == 'anthropic/claude-sonnet-4'
            )
            assert (
                member.agent_settings_diff['llm']['base_url']
                == 'https://api.anthropic.com/v1'
            )
            assert member.conversation_settings_diff['max_iterations'] == 100
            assert decrypt_value(member._llm_api_key) == 'shared-external-api-key'


@pytest.mark.asyncio
async def test_store_keeps_openhands_managed_keys_member_specific(
    session_maker, async_session_maker, org_with_multiple_members_fixture
):
    """Managed OpenHands keys should not be copied from one member to everyone else."""
    from sqlalchemy import select
    from storage.org import Org
    from storage.org_member import OrgMember

    fixture = org_with_multiple_members_fixture
    org_id = fixture['org_id']
    admin_user_id = str(fixture['admin_user_id'])
    decrypt_value = fixture['decrypt_value']

    store = SaasSettingsStore(admin_user_id)
    new_settings = _make_settings(
        model='openhands/claude-opus-4-5-20251101',
        base_url=LITE_LLM_API_URL,
        max_iterations=75,
        api_key='admin-managed-api-key',
    )

    with (
        patch('storage.saas_settings_store.a_session_maker', async_session_maker),
        patch(
            'storage.saas_settings_store.LiteLlmManager.verify_existing_key',
            new_callable=AsyncMock,
            return_value=True,
        ),
    ):
        await store.store(new_settings)

    with session_maker() as session:
        org = session.execute(select(Org).where(Org.id == org_id)).scalars().first()
        assert org is not None
        # Settings keeps the public openhands/ provider prefix in persisted data
        assert (
            org.agent_settings['llm']['model'] == 'openhands/claude-opus-4-5-20251101'
        )
        assert org.agent_settings['llm']['base_url'] == LITE_LLM_API_URL
        assert org.conversation_settings['max_iterations'] == 75

        members = {
            str(member.user_id): member
            for member in session.execute(
                select(OrgMember).where(OrgMember.org_id == org_id)
            )
            .scalars()
            .all()
        }
        assert len(members) == 3

        admin_member = members[admin_user_id]
        assert decrypt_value(admin_member._llm_api_key) == 'admin-managed-api-key'

        member1 = members[str(fixture['member1_user_id'])]
        member2 = members[str(fixture['member2_user_id'])]
        assert decrypt_value(member1._llm_api_key) == 'member1-initial-key'
        assert decrypt_value(member2._llm_api_key) == 'member2-initial-key'

        for member in members.values():
            assert (
                member.agent_settings_diff['llm']['model']
                == 'openhands/claude-opus-4-5-20251101'
            )
            assert member.agent_settings_diff['llm']['base_url'] == LITE_LLM_API_URL
            assert member.conversation_settings_diff['max_iterations'] == 75


@pytest.mark.asyncio
async def test_store_keeps_mcp_config_private_to_acting_member(
    session_maker, async_session_maker, org_with_multiple_members_fixture
):
    """A member's MCP servers must stay scoped to that member's row.

    After Member 1 saves an mcp_config, no other org member sees those
    servers on load, and ``org.agent_settings`` carries no mcp_config so
    new joiners don't inherit them via the org defaults.
    """
    from sqlalchemy import select
    from storage.org import Org
    from storage.org_member import OrgMember

    # Arrange
    fixture = org_with_multiple_members_fixture
    org_id = fixture['org_id']
    admin_user_id = str(fixture['admin_user_id'])
    member1_user_id = str(fixture['member1_user_id'])
    member2_user_id = str(fixture['member2_user_id'])

    user_mcp_config = {
        'mcpServers': {
            'user1': {'url': 'https://user1-mcp-server.com', 'transport': 'sse'}
        },
    }
    new_settings = DataSettings()
    new_settings.update(
        {
            'agent_settings_diff': {
                'llm': {
                    'model': 'test-model',
                    'base_url': 'http://non-litellm-url.com',
                    'api_key': 'test-api-key',
                },
                'mcp_config': user_mcp_config,
            },
        }
    )

    # Act — Member 1 (admin) saves the mcp_config
    store = SaasSettingsStore(admin_user_id)
    with patch('storage.saas_settings_store.a_session_maker', async_session_maker):
        await store.store(new_settings)

    # Assert — only the acting member's row carries mcp_config; org and
    # other members do not.
    with session_maker() as session:
        org = session.execute(select(Org).where(Org.id == org_id)).scalars().first()
        members = {
            str(m.user_id): m
            for m in session.execute(
                select(OrgMember).where(OrgMember.org_id == org_id)
            )
            .scalars()
            .all()
        }

    assert 'mcp_config' not in org.agent_settings
    assert (
        members[admin_user_id].agent_settings_diff.get('mcp_config') == user_mcp_config
    )
    assert 'mcp_config' not in members[member1_user_id].agent_settings_diff
    assert 'mcp_config' not in members[member2_user_id].agent_settings_diff


@pytest.mark.asyncio
async def test_store_skips_ensure_api_key_for_non_openhands_model_without_base_url(
    session_maker, async_session_maker, org_with_multiple_members_fixture
):
    """When saving a non-OpenHands model with no base URL (basic view BYOR),
    _ensure_api_key should NOT be called, preserving the user's custom API key.

    This is the primary bug fix: users selecting e.g. OpenAI in basic view and
    providing their own API key should not have it overwritten by a proxy key.
    """
    fixture = org_with_multiple_members_fixture
    admin_user_id = str(fixture['admin_user_id'])
    store = SaasSettingsStore(admin_user_id)

    settings = _make_settings(
        model='openai/gpt-5.2',
        api_key='sk-user-custom-openai-key',
    )

    with (
        patch('storage.saas_settings_store.a_session_maker', async_session_maker),
        patch.object(store, '_ensure_api_key', new_callable=AsyncMock) as mock_ensure,
    ):
        await store.store(settings)

    mock_ensure.assert_not_called()


@pytest.mark.asyncio
async def test_store_calls_ensure_api_key_for_openhands_model_without_base_url(
    session_maker, async_session_maker, org_with_multiple_members_fixture
):
    """OpenHands models still require proxy-key verification without a base URL."""
    fixture = org_with_multiple_members_fixture
    admin_user_id = str(fixture['admin_user_id'])
    store = SaasSettingsStore(admin_user_id)

    settings = _make_settings(
        model='openhands/claude-opus-4-5-20251101',
        api_key='sk-stale-openai-key',
    )

    with (
        patch('storage.saas_settings_store.a_session_maker', async_session_maker),
        patch.object(store, '_ensure_api_key', new_callable=AsyncMock) as mock_ensure,
    ):
        await store.store(settings)

    mock_ensure.assert_called_once()


@pytest.mark.asyncio
async def test_store_calls_ensure_api_key_when_base_url_is_litellm_proxy(
    session_maker, async_session_maker, org_with_multiple_members_fixture
):
    """Explicit LiteLLM proxy usage should always verify/generate the API key."""
    fixture = org_with_multiple_members_fixture
    admin_user_id = str(fixture['admin_user_id'])
    store = SaasSettingsStore(admin_user_id)

    settings = _make_settings(
        model='openai/gpt-5.2',
        base_url=LITE_LLM_API_URL,
        api_key='sk-some-key',
    )

    with (
        patch('storage.saas_settings_store.a_session_maker', async_session_maker),
        patch.object(store, '_ensure_api_key', new_callable=AsyncMock) as mock_ensure,
    ):
        await store.store(settings)

    mock_ensure.assert_called_once()


@pytest.mark.asyncio
async def test_store_and_load_mcp_config_via_agent_settings(
    async_session_maker, org_with_multiple_members_fixture
):
    """mcp_config is persisted inside agent_settings / agent_settings_diff and
    round-trips correctly through store → load."""
    fixture = org_with_multiple_members_fixture
    admin_user_id = str(fixture['admin_user_id'])

    admin_mcp_config = {
        'mcpServers': {
            'admin': {'url': 'https://admin-private-server.com', 'transport': 'sse'}
        },
    }

    admin_store = SaasSettingsStore(admin_user_id)

    admin_settings = DataSettings()
    admin_settings.update(
        {
            'agent_settings_diff': {
                'llm': {
                    'model': 'test-model',
                    'base_url': 'http://non-litellm-url.com',
                    'api_key': 'test-api-key',
                },
                'mcp_config': admin_mcp_config,
            },
        }
    )

    with patch('storage.saas_settings_store.a_session_maker', async_session_maker):
        await admin_store.store(admin_settings)

    with (
        patch('storage.saas_settings_store.a_session_maker', async_session_maker),
        patch('storage.user_store.a_session_maker', async_session_maker),
        patch('storage.org_store.a_session_maker', async_session_maker),
    ):
        loaded = await admin_store.load()

    assert loaded is not None
    assert loaded.agent_settings.mcp_config is not None
    assert (
        loaded.agent_settings.mcp_config.mcpServers['admin'].url
        == 'https://admin-private-server.com'
    )


@pytest.mark.asyncio
async def test_load_drops_legacy_org_level_mcp_config(
    session_maker, async_session_maker, org_with_multiple_members_fixture
):
    """Legacy org-level mcp_config (from before the fix) must not leak
    to members on load. Members without their own mcp_config see ``None``
    even if the org row still carries a stale value in the database.
    """
    from sqlalchemy import select
    from storage.org import Org
    from storage.user import User

    # Arrange — simulate pre-fix data: org carries an mcp_config that
    # was broadcast at the org level. member1 has no mcp_config of their
    # own.
    fixture = org_with_multiple_members_fixture
    org_id = fixture['org_id']
    member1_user_id = str(fixture['member1_user_id'])

    legacy_org_mcp_config = {
        'mcpServers': {
            'leaked': {'url': 'https://leaked-server.com', 'transport': 'sse'}
        },
    }
    with session_maker() as session:
        org = session.execute(select(Org).where(Org.id == org_id)).scalars().first()
        org.agent_settings = {
            'agent_kind': 'openhands',
            'mcp_config': legacy_org_mcp_config,
        }
        # Populate the nullable bool defaults that the Settings model
        # requires non-None when load() rebuilds the Settings object.
        user = (
            session.execute(select(User).where(User.id == fixture['member1_user_id']))
            .scalars()
            .first()
        )
        user.enable_sound_notifications = False
        session.commit()

    # Act
    store = SaasSettingsStore(member1_user_id)
    with (
        patch('storage.saas_settings_store.a_session_maker', async_session_maker),
        patch('storage.user_store.a_session_maker', async_session_maker),
        patch('storage.org_store.a_session_maker', async_session_maker),
    ):
        loaded = await store.load()

    # Assert — legacy org mcp_config is not inherited by member1
    assert loaded is not None
    assert loaded.agent_settings.mcp_config is None


@pytest.mark.asyncio
async def test_store_and_load_llm_profiles_round_trip(
    async_session_maker, org_with_multiple_members_fixture
):
    """Saved llm_profiles must persist on the User row and round-trip through
    store → load. Without the user.llm_profiles column they are silently
    dropped on store and always default to empty on load."""
    from openhands.sdk.llm import LLM

    fixture = org_with_multiple_members_fixture
    admin_user_id = str(fixture['admin_user_id'])
    admin_store = SaasSettingsStore(admin_user_id)

    settings = DataSettings()
    settings.update(
        {
            'agent_settings_diff': {
                'llm': {
                    'model': 'anthropic/claude-sonnet-4-5-20250929',
                    'base_url': 'https://api.anthropic.com/v1',
                    'api_key': 'active-key',
                },
            },
        }
    )
    settings.llm_profiles.save(
        'work',
        LLM(
            model='anthropic/claude-sonnet-4-5-20250929',
            base_url='https://api.anthropic.com/v1',
            api_key=SecretStr('work-key'),
        ),
    )
    settings.llm_profiles.save(
        'personal',
        LLM(
            model='openai/gpt-5.2',
            base_url='https://api.openai.com/v1',
            api_key=SecretStr('personal-key'),
        ),
    )
    settings.llm_profiles.active = 'work'

    with patch('storage.saas_settings_store.a_session_maker', async_session_maker):
        await admin_store.store(settings)

    with (
        patch('storage.saas_settings_store.a_session_maker', async_session_maker),
        patch('storage.user_store.a_session_maker', async_session_maker),
        patch('storage.org_store.a_session_maker', async_session_maker),
    ):
        loaded = await admin_store.load()

    assert loaded is not None
    assert set(loaded.llm_profiles.profiles.keys()) == {'work', 'personal'}
    assert loaded.llm_profiles.active == 'work'

    work = loaded.llm_profiles.require('work')
    assert work.model == 'anthropic/claude-sonnet-4-5-20250929'
    assert work.base_url == 'https://api.anthropic.com/v1'
    assert work.api_key is not None
    assert work.api_key.get_secret_value() == 'work-key'

    personal = loaded.llm_profiles.require('personal')
    assert personal.model == 'openai/gpt-5.2'
    assert personal.api_key.get_secret_value() == 'personal-key'


@pytest.mark.parametrize(
    'llm_profiles_value',
    [
        pytest.param(None, id='pre-migration: llm_profiles is null'),
        pytest.param(
            {'profiles': {}, 'active': None},
            id='already-migrated: profiles dict is empty',
        ),
    ],
)
@pytest.mark.asyncio
async def test_load_with_null_or_empty_llm_profiles_seeds_default_profile(
    async_session_maker, org_with_multiple_members_fixture, llm_profiles_value
):
    """Seed Default profile from legacy config when no profiles exist.

    Rows predating the llm_profiles column read back as None, and already-
    migrated orgs may have an empty profiles dict. Rather than presenting an
    empty profiles UI on upgrade, load() seeds a "Default" profile from the
    legacy agent_settings.llm config (mirroring the OSS FileSettingsStore
    behaviour), with that profile marked active.
    """
    from sqlalchemy import update
    from storage.user import User

    fixture = org_with_multiple_members_fixture
    admin_user_id = fixture['admin_user_id']
    admin_store = SaasSettingsStore(str(admin_user_id))

    seed_settings = _make_settings(
        model='anthropic/claude-sonnet-4-5-20250929',
        api_key='seed-key',
        base_url='https://api.anthropic.com/v1',
    )
    with patch('storage.saas_settings_store.a_session_maker', async_session_maker):
        await admin_store.store(seed_settings)

    async with async_session_maker() as session:
        await session.execute(
            update(User)
            .where(User.id == admin_user_id)
            .values(llm_profiles=llm_profiles_value)
        )
        await session.commit()

    with (
        patch('storage.saas_settings_store.a_session_maker', async_session_maker),
        patch('storage.user_store.a_session_maker', async_session_maker),
        patch('storage.org_store.a_session_maker', async_session_maker),
    ):
        loaded = await admin_store.load()

    assert loaded is not None
    assert set(loaded.llm_profiles.profiles.keys()) == {'Default'}
    assert loaded.llm_profiles.active == 'Default'
    default = loaded.llm_profiles.require('Default')
    assert default.model == 'anthropic/claude-sonnet-4-5-20250929'


@pytest.mark.asyncio
async def test_load_persists_seeded_default_profile_onto_org(
    async_session_maker, org_with_multiple_members_fixture
):
    """The seeded Default profile must be written back to org.llm_profiles.

    The seed is otherwise in-memory only, so the org-profiles management API
    (which reads org.llm_profiles directly) would still see an empty list.
    load() backfills it once so the user's last LLM becomes a real stored
    profile on first use of LLM profiles.
    """
    from sqlalchemy import select, update
    from storage.org import Org

    fixture = org_with_multiple_members_fixture
    admin_user_id = fixture['admin_user_id']
    org_id = fixture['org_id']
    admin_store = SaasSettingsStore(str(admin_user_id))

    seed_settings = _make_settings(
        model='anthropic/claude-sonnet-4-5-20250929',
        api_key='seed-key',
        base_url='https://api.anthropic.com/v1',
    )
    with patch('storage.saas_settings_store.a_session_maker', async_session_maker):
        await admin_store.store(seed_settings)

    # Simulate a pre-migration org: no profiles stored yet.
    async with async_session_maker() as session:
        await session.execute(
            update(Org).where(Org.id == org_id).values(llm_profiles=None)
        )
        await session.commit()

    with (
        patch('storage.saas_settings_store.a_session_maker', async_session_maker),
        patch('storage.user_store.a_session_maker', async_session_maker),
        patch('storage.org_store.a_session_maker', async_session_maker),
    ):
        await admin_store.load()

    async with async_session_maker() as session:
        org = (
            (await session.execute(select(Org).where(Org.id == org_id)))
            .scalars()
            .first()
        )

    assert org.llm_profiles is not None
    assert set(org.llm_profiles['profiles'].keys()) == {'Default'}
    assert org.llm_profiles['active'] == 'Default'
    persisted_default = org.llm_profiles['profiles']['Default']
    assert persisted_default['model'] == 'anthropic/claude-sonnet-4-5-20250929'
    assert persisted_default['base_url'] == 'https://api.anthropic.com/v1'
    # API key from the legacy config must survive the round-trip so the user
    # doesn't have to re-enter it after the profiles upgrade.
    assert persisted_default['api_key'] == 'seed-key'


@pytest.mark.asyncio
async def test_llm_profiles_are_encrypted_at_rest(
    async_session_maker, org_with_multiple_members_fixture
):
    """The raw value in the user.llm_profiles column must be ciphertext, not
    a JSON dict — profile api_keys would otherwise leak in DB dumps,
    replicas, and backups. Mirrors the encryption invariant org and
    org_member already enforce on _llm_api_key."""
    from sqlalchemy import select, text
    from storage.user import User

    from openhands.sdk.llm import LLM

    fixture = org_with_multiple_members_fixture
    admin_user_id = fixture['admin_user_id']
    admin_store = SaasSettingsStore(str(admin_user_id))

    settings = DataSettings()
    settings.update(
        {
            'agent_settings_diff': {
                'llm': {
                    'model': 'anthropic/claude-sonnet-4-5-20250929',
                    'base_url': 'https://api.anthropic.com/v1',
                    'api_key': 'active-key',
                },
            },
        }
    )
    settings.llm_profiles.save(
        'work',
        LLM(
            model='anthropic/claude-sonnet-4-5-20250929',
            base_url='https://api.anthropic.com/v1',
            api_key=SecretStr('super-secret-byok'),
        ),
    )
    with patch('storage.saas_settings_store.a_session_maker', async_session_maker):
        await admin_store.store(settings)

    async with async_session_maker() as session:
        # Bypass the ORM-level TypeDecorator by reading the raw cell.
        # SQLite stores UUIDs hyphen-stripped, so normalize both sides.
        rows = (
            await session.execute(text('SELECT id, llm_profiles FROM "user"'))
        ).all()
    raw = next(
        (r[1] for r in rows if str(r[0]).replace('-', '') == admin_user_id.hex),
        None,
    )
    assert raw is not None
    # The plaintext secret must not appear anywhere in the at-rest payload.
    assert 'super-secret-byok' not in raw
    # And the raw payload must not be parseable as JSON — i.e. it's
    # encrypted, not a serialized profiles dict.
    import json as _json

    with pytest.raises(_json.JSONDecodeError):
        _json.loads(raw)

    # Sanity: ORM read still decrypts correctly.
    async with async_session_maker() as session:
        user = (
            await session.execute(select(User).where(User.id == admin_user_id))
        ).scalar_one()
    assert user.llm_profiles is not None
    assert user.llm_profiles['profiles']['work']['api_key'] == 'super-secret-byok'


@pytest.mark.asyncio
async def test_store_replaces_mcp_config_on_delete(
    session_maker, async_session_maker, org_with_multiple_members_fixture
):
    """Deleting a server from a member's mcp_config sticks on the acting
    member's row and never touches other members' rows.

    Combines the APP-1862 wholesale-replacement contract (deletes are not
    resurrected by deep_merge) with the per-member privacy contract.
    """
    from sqlalchemy import select
    from storage.org_member import OrgMember

    # Arrange — Member 1 (admin) starts with 3 MCP servers
    fixture = org_with_multiple_members_fixture
    org_id = fixture['org_id']
    admin_user_id = str(fixture['admin_user_id'])
    member1_user_id = str(fixture['member1_user_id'])

    store = SaasSettingsStore(admin_user_id)
    initial_mcp_config = {
        'mcpServers': {
            'server1': {'url': 'https://server1.com', 'transport': 'sse'},
            'server2': {'url': 'https://server2.com', 'transport': 'sse'},
            'server3': {'url': 'https://server3.com', 'transport': 'sse'},
        },
    }
    initial_settings = DataSettings()
    initial_settings.update(
        {
            'agent_settings_diff': {
                'llm': {
                    'model': 'test-model',
                    'base_url': 'http://test-url.com',
                    'api_key': 'test-key',
                },
                'mcp_config': initial_mcp_config,
            },
        }
    )
    with patch('storage.saas_settings_store.a_session_maker', async_session_maker):
        await store.store(initial_settings)

    # Act — re-save with server3 removed
    updated_mcp_config = {
        'mcpServers': {
            'server1': {'url': 'https://server1.com', 'transport': 'sse'},
            'server2': {'url': 'https://server2.com', 'transport': 'sse'},
        },
    }
    updated_settings = DataSettings()
    updated_settings.update(
        {
            'agent_settings_diff': {
                'llm': {
                    'model': 'test-model',
                    'base_url': 'http://test-url.com',
                    'api_key': 'test-key',
                },
                'mcp_config': updated_mcp_config,
            },
        }
    )
    with patch('storage.saas_settings_store.a_session_maker', async_session_maker):
        await store.store(updated_settings)

    # Assert — server3 is gone from the acting member; other members were
    # never touched by either save.
    with session_maker() as session:
        members = {
            str(m.user_id): m
            for m in session.execute(
                select(OrgMember).where(OrgMember.org_id == org_id)
            )
            .scalars()
            .all()
        }

    admin_servers = (
        members[admin_user_id]
        .agent_settings_diff.get('mcp_config', {})
        .get('mcpServers', {})
    )
    assert set(admin_servers.keys()) == {'server1', 'server2'}
    assert 'mcp_config' not in members[member1_user_id].agent_settings_diff
