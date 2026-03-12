"""Unit tests for timeout field handling in SaaS settings store."""

# Mock the database module before importing
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from openhands.server.settings import Settings

with patch('storage.database.a_session_maker'):
    from enterprise.server.constants import (
        LITE_LLM_API_URL,
    )
    from enterprise.storage.saas_settings_store import SaasSettingsStore
    from enterprise.storage.user_settings import UserSettings


@pytest.fixture
def mock_config():
    config = patch('storage.saas_settings_store.OpenHandsConfig').return_value
    config.jwt_secret = SecretStr('test_secret_key')
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
                    llm_api_key=SecretStr('secret_key'),
                    llm_base_url='http://test.url',
                    agent='CodeActAgent',
                    language='en',
                    timeout=None,  # Explicitly set timeout to None for enterprise
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
            # Make a copy of the item without email, email_verified, secrets_store, and timeout
            item_dict = item.model_dump(context={'expose_secrets': True})
            if 'email' in item_dict:
                del item_dict['email']
            if 'email_verified' in item_dict:
                del item_dict['email_verified']
            if 'secrets_store' in item_dict:
                del item_dict['secrets_store']
            if 'timeout' in item_dict:
                del item_dict['timeout']

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
async def test_timeout_field_excluded_from_enterprise(settings_store):
    """Test that timeout field is excluded from enterprise settings storage."""
    # Create settings with timeout field
    settings = Settings(
        llm_api_key=SecretStr('secret_key'),
        llm_base_url=LITE_LLM_API_URL,
        agent='smith',
        email='test@example.com',
        email_verified=True,
        timeout=60,  # Set a timeout value
    )

    # Store settings
    await settings_store.store(settings)

    # Load settings back
    loaded_settings = await settings_store.load()

    # Verify timeout is None (excluded from enterprise)
    assert loaded_settings is not None
    assert loaded_settings.timeout is None
    assert loaded_settings.llm_api_key.get_secret_value() == 'secret_key'
    assert loaded_settings.agent == 'smith'


@pytest.mark.asyncio
async def test_timeout_filtering_logic_directly(settings_store):
    """Test the actual _encrypt_kwargs and _decrypt_kwargs timeout filtering logic."""
    # Create settings with timeout field and nested mcp_config with timeout
    settings = Settings(
        llm_api_key=SecretStr('secret_key'),
        llm_base_url=LITE_LLM_API_URL,
        agent='smith',
        email='test@example.com',
        email_verified=True,
        timeout=60,  # Top-level timeout
    )

    # Test _encrypt_kwargs directly
    settings_dict = settings.model_dump(context={'expose_secrets': True})

    # Add nested mcp_config with timeout to test that it's not removed
    if 'mcp_config' not in settings_dict:
        settings_dict['mcp_config'] = {}
    settings_dict['mcp_config']['timeout'] = 120  # Nested timeout should be preserved

    # Call _encrypt_kwargs directly
    settings_store._encrypt_kwargs(settings_dict)

    # Verify top-level timeout is removed but nested one is preserved
    assert 'timeout' not in settings_dict  # Top-level timeout removed
    assert settings_dict['mcp_config']['timeout'] == 120  # Nested timeout preserved

    # Test _decrypt_kwargs directly
    encrypted_settings = settings_dict.copy()
    settings_store._decrypt_kwargs(encrypted_settings)

    # Verify top-level timeout is removed but nested one is preserved
    assert 'timeout' not in encrypted_settings  # Top-level timeout removed
    if 'mcp_config' in encrypted_settings:
        assert (
            encrypted_settings['mcp_config']['timeout'] == 120
        )  # Nested timeout preserved
