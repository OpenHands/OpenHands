from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from openhands.core.config.openhands_config import OpenHandsConfig
from openhands.server.settings import Settings
from openhands.storage.data_models.settings import Settings as DataSettings

# Mock the database module before importing
with patch('storage.database.engine'), patch('storage.database.a_engine'):
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
def settings_store(session_maker, mock_config):
    store = SaasSettingsStore(
        '5594c7b6-f959-4b81-92e9-b09c206f5081', session_maker, mock_config
    )

    # Patch the load method to read from UserSettings table directly (for testing)
    async def patched_load():
        with store.session_maker() as session:
            user_settings = (
                session.query(UserSettings)
                .filter(UserSettings.keycloak_user_id == store.user_id)
                .first()
            )
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

            # Continue with the original implementation
            with store.session_maker() as session:
                existing = None
                if item_dict:
                    store._encrypt_kwargs(item_dict)
                    query = session.query(UserSettings).filter(
                        UserSettings.keycloak_user_id == store.user_id
                    )

                    # First check if we have an existing entry in the new table
                    existing = query.first()

                if existing:
                    # Update existing entry
                    for key, value in item_dict.items():
                        if key in existing.__class__.__table__.columns:
                            setattr(existing, key, value)
                    session.merge(existing)
                else:
                    item_dict['keycloak_user_id'] = store.user_id
                    settings = UserSettings(**item_dict)
                    session.add(settings)
                session.commit()

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
    with settings_store.session_maker() as session:
        stored = (
            session.query(UserSettings)
            .filter(
                UserSettings.keycloak_user_id == '550e8400-e29b-41d4-a716-446655440000'
            )
            .first()
        )
        assert stored is not None
        assert stored.agent == 'smith'


@pytest.mark.asyncio
async def test_load_returns_default_when_not_found(settings_store, session_maker):
    file_store = MagicMock()
    file_store.read.side_effect = FileNotFoundError()

    with (
        patch('storage.saas_settings_store.session_maker', session_maker),
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
    with settings_store.session_maker() as session:
        stored = (
            session.query(UserSettings)
            .filter(
                UserSettings.keycloak_user_id == '5594c7b6-f959-4b81-92e9-b09c206f5081'
            )
            .first()
        )
        # The stored key should be encrypted
        assert stored.llm_api_key != 'secret_key'
        # But we should be able to decrypt it when loading
        loaded_settings = await settings_store.load()
        assert loaded_settings.llm_api_key.get_secret_value() == 'secret_key'


@pytest.mark.asyncio
async def test_ensure_api_key_keeps_valid_key(mock_config):
    """When the existing key is valid, it should be kept unchanged."""
    store = SaasSettingsStore('test-user-id-123', MagicMock(), mock_config)
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
    store = SaasSettingsStore('test-user-id-123', MagicMock(), mock_config)
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
