import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from openhands.core.config.openhands_config import OpenHandsConfig
from openhands.server.settings import Settings
from openhands.storage.data_models.settings import Settings as DataSettings

# Mock the database module before importing
with patch('storage.database.a_session_maker'):
    from server.constants import (
        LITE_LLM_API_URL,
    )
    from storage.saas_settings_store import SaasSettingsStore
    from storage.user_settings import UserSettings


@pytest.fixture
def mock_config():
    config = MagicMock(spec=OpenHandsConfig)
    config.jwt_secret = SecretStr('test_secret')
    config.file_store = 'google_cloud'
    config.file_store_path = 'bucket'
    return config


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
                return Settings(
                    llm_api_key=SecretStr('test_api_key'),
                    llm_base_url='http://test.url',
                    agent='CodeActAgent',
                    language='en',
                )

            # Decrypt and convert to Settings
            kwargs = {}
            for column in UserSettings.__table__.columns:
                if column.name != 'keycloak_user_id':
                    value = getattr(user_settings, column.name, None)
                    if value is not None:
                        kwargs[column.name] = value

            store._decrypt_kwargs(kwargs)
            settings = Settings(**kwargs)
            settings.email = 'test@example.com'
            settings.email_verified = True
            return settings

    # Patch the store method to write to UserSettings table directly (for testing)
    async def patched_store(item):
        if item:
            # Make a copy of the item without email and email_verified
            item_dict = item.model_dump(context={'expose_secrets': True})
            if 'email' in item_dict:
                del item_dict['email']
            if 'email_verified' in item_dict:
                del item_dict['email_verified']
            if 'secrets_store' in item_dict:
                del item_dict['secrets_store']

            # Encrypt the data before storing
            store._encrypt_kwargs(item_dict)

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
    settings = Settings(
        llm_api_key=SecretStr('secret_key'),
        llm_base_url=LITE_LLM_API_URL,
        agent='smith',
        email='test@example.com',
        email_verified=True,
    )

    await settings_store.store(settings)

    # Load and verify settings
    loaded_settings = await settings_store.load()
    assert loaded_settings is not None
    assert loaded_settings.llm_api_key.get_secret_value() == 'secret_key'
    assert loaded_settings.agent == 'smith'

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
        assert stored.agent == 'smith'


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
        assert loaded_settings.agent == 'CodeActAgent'
        assert loaded_settings.llm_api_key.get_secret_value() == 'test_api_key'
        assert loaded_settings.llm_base_url == 'http://test.url'


@pytest.mark.asyncio
async def test_encryption(settings_store):
    settings_store.user_id = '5594c7b6-f959-4b81-92e9-b09c206f5081'  # GitHub user ID
    settings = Settings(
        llm_api_key=SecretStr('secret_key'),
        agent='smith',
        llm_base_url=LITE_LLM_API_URL,
        email='test@example.com',
        email_verified=True,
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
        assert loaded_settings.llm_api_key.get_secret_value() == 'secret_key'


@pytest.mark.asyncio
async def test_ensure_api_key_keeps_valid_key(mock_config):
    """When the existing key is valid, it should be kept unchanged."""
    store = SaasSettingsStore('test-user-id-123', mock_config)
    existing_key = 'sk-existing-key'
    item = DataSettings(
        llm_model='openhands/gpt-4', llm_api_key=SecretStr(existing_key)
    )

    with patch(
        'storage.saas_settings_store.LiteLlmManager.verify_existing_key',
        new_callable=AsyncMock,
        return_value=True,
    ):
        await store._ensure_api_key(item, 'org-123', openhands_type=True)

        # Key should remain unchanged when it's valid
        assert item.llm_api_key is not None
        assert item.llm_api_key.get_secret_value() == existing_key


@pytest.mark.asyncio
async def test_ensure_api_key_generates_new_key_when_verification_fails(
    mock_config,
):
    """When verification fails, a new key should be generated."""
    store = SaasSettingsStore('test-user-id-123', mock_config)
    new_key = 'sk-new-key'
    item = DataSettings(
        llm_model='openhands/gpt-4', llm_api_key=SecretStr('sk-invalid-key')
    )

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

        assert item.llm_api_key is not None
        assert item.llm_api_key.get_secret_value() == new_key


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
            enable_default_condenser=True,
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
            llm_model='old-model-v1',
            llm_base_url='http://old-url-1.com',
            max_iterations=10,
            status='active',
        )
        session.add(admin_member)

        member1 = OrgMember(
            org_id=org_id,
            user_id=member1_user_id,
            role_id=10,
            llm_api_key='member1-initial-key',
            llm_model='old-model-v2',
            llm_base_url='http://old-url-2.com',
            max_iterations=20,
            status='active',
        )
        session.add(member1)

        member2 = OrgMember(
            org_id=org_id,
            user_id=member2_user_id,
            role_id=10,
            llm_api_key='member2-initial-key',
            llm_model='old-model-v3',
            llm_base_url='http://old-url-3.com',
            max_iterations=30,
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
async def test_store_propagates_llm_settings_to_all_org_members(
    session_maker, async_session_maker, mock_config, org_with_multiple_members_fixture
):
    """When admin saves LLM settings, all org members should receive the updated settings.

    This test verifies using a real database that:
    1. The bulk UPDATE targets the correct organization (WHERE clause is correct)
    2. All LLM fields are correctly set (llm_model, llm_base_url, max_iterations, llm_api_key)
    3. The llm_api_key is properly encrypted
    4. All members in the org receive the same updated values
    """
    from sqlalchemy import select
    from storage.org_member import OrgMember

    # Arrange
    fixture = org_with_multiple_members_fixture
    org_id = fixture['org_id']
    admin_user_id = str(fixture['admin_user_id'])
    decrypt_value = fixture['decrypt_value']

    store = SaasSettingsStore(admin_user_id, mock_config)

    new_settings = DataSettings(
        llm_model='new-shared-model/gpt-4',
        llm_base_url='http://new-shared-url.com',
        max_iterations=100,
        llm_api_key=SecretStr('new-shared-api-key'),
    )

    # Act - call store() with async session
    with patch('storage.saas_settings_store.a_session_maker', async_session_maker):
        await store.store(new_settings)

    # Assert - verify ALL org members have the updated LLM settings using sync session
    with session_maker() as session:
        result = session.execute(select(OrgMember).where(OrgMember.org_id == org_id))
        members = result.scalars().all()

        # Verify we have all 3 members
        assert len(members) == 3, f'Expected 3 org members, got {len(members)}'

        for member in members:
            # Verify LLM model is updated
            assert (
                member.llm_model == 'new-shared-model/gpt-4'
            ), f'Expected llm_model to be updated for member {member.user_id}'

            # Verify LLM base URL is updated
            assert (
                member.llm_base_url == 'http://new-shared-url.com'
            ), f'Expected llm_base_url to be updated for member {member.user_id}'

            # Verify max_iterations is updated
            assert (
                member.max_iterations == 100
            ), f'Expected max_iterations to be 100 for member {member.user_id}'

            # Verify the API key is encrypted and decrypts to the correct value
            decrypted_key = decrypt_value(member._llm_api_key)
            assert (
                decrypted_key == 'new-shared-api-key'
            ), f'Expected llm_api_key to decrypt to new-shared-api-key for member {member.user_id}'


@pytest.mark.asyncio
async def test_store_updates_org_default_llm_settings(
    session_maker, async_session_maker, mock_config, org_with_multiple_members_fixture
):
    """When admin saves LLM settings, org's default_llm_model/base_url/max_iterations should be updated.

    This test verifies that the Org table's default settings are updated so that
    new members joining later will inherit the correct LLM configuration.
    """
    from sqlalchemy import select
    from storage.org import Org

    # Arrange
    fixture = org_with_multiple_members_fixture
    org_id = fixture['org_id']
    admin_user_id = str(fixture['admin_user_id'])

    store = SaasSettingsStore(admin_user_id, mock_config)

    new_settings = DataSettings(
        llm_model='anthropic/claude-sonnet-4',
        llm_base_url='https://api.anthropic.com/v1',
        max_iterations=75,
        llm_api_key=SecretStr('test-api-key'),
    )

    # Act
    with patch('storage.saas_settings_store.a_session_maker', async_session_maker):
        await store.store(new_settings)

    # Assert - verify org's default fields were updated
    with session_maker() as session:
        result = session.execute(select(Org).where(Org.id == org_id))
        org = result.scalars().first()

        assert org is not None
        assert org.default_llm_model == 'anthropic/claude-sonnet-4'
        assert org.default_llm_base_url == 'https://api.anthropic.com/v1'
        assert org.default_max_iterations == 75
