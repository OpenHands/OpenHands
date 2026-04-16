import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from openhands.core.config.openhands_config import OpenHandsConfig
from openhands.server.settings import Settings
from openhands.storage.data_models.settings import Settings as DataSettings


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
    """Build a DataSettings with the new nested agent_settings API."""
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
    agent_settings: dict = {}
    if agent is not None:
        agent_settings['agent'] = agent
    if llm:
        agent_settings['llm'] = llm
    agent_settings.update(extra_agent)
    payload: dict = {}
    if agent_settings:
        payload['agent_settings'] = agent_settings
    conv: dict = {}
    if max_iterations is not None:
        conv['max_iterations'] = max_iterations
    if conv:
        payload['conversation_settings'] = conv
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


@pytest.fixture
def mock_config():
    config = MagicMock(spec=OpenHandsConfig)
    config.jwt_secret = SecretStr('test_secret')
    config.file_store = 'google_cloud'
    config.file_store_path = 'bucket'
    return config


def test_member_settings_persist_full_effective_agent_settings(mock_config):
    settings = Settings()
    settings.update(
        {
            'agent_settings': {
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
            'conversation_settings': {
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
def settings_store(async_session_maker, mock_config):
    store = SaasSettingsStore('5594c7b6-f959-4b81-92e9-b09c206f5081', mock_config)
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
                payload['agent_settings'] = agent_dict
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
            'agent_settings': {
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
            'agent_settings': {
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
async def test_ensure_api_key_keeps_valid_key(mock_config):
    """When the existing key is valid, it should be kept unchanged."""
    store = SaasSettingsStore('test-user-id-123', mock_config)
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
async def test_ensure_api_key_generates_new_key_when_verification_fails(
    mock_config,
):
    """When verification fails, a new key should be generated."""
    store = SaasSettingsStore('test-user-id-123', mock_config)
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
async def test_store_updates_org_defaults_and_all_members_for_shared_keys(
    session_maker, async_session_maker, mock_config, org_with_multiple_members_fixture
):
    """External provider keys should still sync as an org-wide shared snapshot."""
    from sqlalchemy import select
    from storage.org import Org
    from storage.org_member import OrgMember

    fixture = org_with_multiple_members_fixture
    org_id = fixture['org_id']
    decrypt_value = fixture['decrypt_value']

    store = SaasSettingsStore(str(fixture['admin_user_id']), mock_config)
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
    session_maker, async_session_maker, mock_config, org_with_multiple_members_fixture
):
    """Managed OpenHands keys should not be copied from one member to everyone else."""
    from sqlalchemy import select
    from storage.org import Org
    from storage.org_member import OrgMember

    fixture = org_with_multiple_members_fixture
    org_id = fixture['org_id']
    admin_user_id = str(fixture['admin_user_id'])
    decrypt_value = fixture['decrypt_value']

    store = SaasSettingsStore(admin_user_id, mock_config)
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
        # Settings normalizes openhands/ → litellm_proxy/ during construction
        assert (
            org.agent_settings['llm']['model']
            == 'litellm_proxy/claude-opus-4-5-20251101'
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
                == 'litellm_proxy/claude-opus-4-5-20251101'
            )
            assert member.agent_settings_diff['llm']['base_url'] == LITE_LLM_API_URL
            assert member.conversation_settings_diff['max_iterations'] == 75


@pytest.mark.asyncio
async def test_store_saves_mcp_config_in_agent_settings(
    session_maker, async_session_maker, mock_config, org_with_multiple_members_fixture
):
    """mcp_config now flows through agent_settings / agent_settings_diff,
    so it is persisted on both the org and all members."""
    from sqlalchemy import select
    from storage.org import Org
    from storage.org_member import OrgMember

    fixture = org_with_multiple_members_fixture
    org_id = fixture['org_id']
    admin_user_id = str(fixture['admin_user_id'])
    member1_user_id = str(fixture['member1_user_id'])
    member2_user_id = str(fixture['member2_user_id'])

    store = SaasSettingsStore(admin_user_id, mock_config)
    user_mcp_config = {
        'mcpServers': {
            'user1': {'url': 'https://user1-mcp-server.com', 'transport': 'sse'}
        },
    }
    new_settings = DataSettings()
    new_settings.update(
        {
            'agent_settings': {
                'llm': {
                    'model': 'test-model',
                    'base_url': 'http://non-litellm-url.com',
                    'api_key': 'test-api-key',
                },
                'mcp_config': user_mcp_config,
            },
        }
    )

    with patch('storage.saas_settings_store.a_session_maker', async_session_maker):
        await store.store(new_settings)

    with session_maker() as session:
        org = session.execute(select(Org).where(Org.id == org_id)).scalars().first()
        assert org is not None
        assert org.agent_settings.get('mcp_config') == user_mcp_config

        members = {
            str(m.user_id): m
            for m in session.execute(
                select(OrgMember).where(OrgMember.org_id == org_id)
            )
            .scalars()
            .all()
        }
        assert (
            members[admin_user_id].agent_settings_diff.get('mcp_config')
            == user_mcp_config
        )
        assert (
            members[member1_user_id].agent_settings_diff.get('mcp_config')
            == user_mcp_config
        )
        assert (
            members[member2_user_id].agent_settings_diff.get('mcp_config')
            == user_mcp_config
        )


@pytest.mark.asyncio
async def test_store_skips_ensure_api_key_for_non_openhands_model_without_base_url(
    session_maker, async_session_maker, mock_config, org_with_multiple_members_fixture
):
    """When saving a non-OpenHands model with no base URL (basic view BYOR),
    _ensure_api_key should NOT be called, preserving the user's custom API key.

    This is the primary bug fix: users selecting e.g. OpenAI in basic view and
    providing their own API key should not have it overwritten by a proxy key.
    """
    fixture = org_with_multiple_members_fixture
    admin_user_id = str(fixture['admin_user_id'])
    store = SaasSettingsStore(admin_user_id, mock_config)

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
    session_maker, async_session_maker, mock_config, org_with_multiple_members_fixture
):
    """OpenHands models still require proxy-key verification without a base URL."""
    fixture = org_with_multiple_members_fixture
    admin_user_id = str(fixture['admin_user_id'])
    store = SaasSettingsStore(admin_user_id, mock_config)

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
    session_maker, async_session_maker, mock_config, org_with_multiple_members_fixture
):
    """Explicit LiteLLM proxy usage should always verify/generate the API key."""
    fixture = org_with_multiple_members_fixture
    admin_user_id = str(fixture['admin_user_id'])
    store = SaasSettingsStore(admin_user_id, mock_config)

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
    async_session_maker, mock_config, org_with_multiple_members_fixture
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

    admin_store = SaasSettingsStore(admin_user_id, mock_config)

    admin_settings = DataSettings()
    admin_settings.update(
        {
            'agent_settings': {
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
