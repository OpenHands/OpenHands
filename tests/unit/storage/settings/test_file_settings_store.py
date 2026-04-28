import os
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from openhands.app_server.settings.file_settings_store import FileSettingsStore
from openhands.app_server.settings.settings_models import Settings
from openhands.core.config.openhands_config import OpenHandsConfig
from openhands.sdk.llm import LLM
from openhands.sdk.settings import AgentSettings, ConversationSettings


@pytest.fixture(autouse=True)
def allow_short_context_windows():
    with patch.dict(os.environ, {'ALLOW_SHORT_CONTEXT_WINDOWS': 'true'}, clear=False):
        yield


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def file_settings_store(temp_dir):
    return FileSettingsStore(root_dir=temp_dir)


@pytest.mark.asyncio
async def test_load_nonexistent_data(file_settings_store):
    with patch(
        'openhands.app_server.settings.settings_models.load_openhands_config',
        MagicMock(return_value=OpenHandsConfig()),
    ):
        assert await file_settings_store.load() is None


@pytest.mark.asyncio
async def test_store_and_load_data(file_settings_store, temp_dir):
    # Test data
    init_data = Settings(
        language='python',
        agent_settings=AgentSettings(
            agent='test-agent',
            llm=LLM(
                model='test-model',
                api_key=SecretStr('test-key'),
                base_url='https://test.com',
            ),
        ),
        conversation_settings=ConversationSettings(
            max_iterations=100,
            security_analyzer='llm',
            confirmation_mode=True,
        ),
    )

    # Store data
    await file_settings_store.store(init_data)

    # Verify file was written
    expected_path = temp_dir / 'settings.json'
    assert expected_path.exists()

    # Load and verify data
    loaded_data = await file_settings_store.load()
    assert loaded_data is not None
    assert loaded_data.language == init_data.language
    assert loaded_data.agent_settings.agent == init_data.agent_settings.agent
    assert (
        loaded_data.conversation_settings.max_iterations
        == init_data.conversation_settings.max_iterations
    )
    assert (
        loaded_data.conversation_settings.security_analyzer
        == init_data.conversation_settings.security_analyzer
    )
    assert (
        loaded_data.conversation_settings.confirmation_mode
        == init_data.conversation_settings.confirmation_mode
    )
    assert loaded_data.agent_settings.llm.model == init_data.agent_settings.llm.model
    assert loaded_data.agent_settings.llm.api_key is not None
    assert init_data.agent_settings.llm.api_key is not None
    assert (
        loaded_data.agent_settings.llm.api_key.get_secret_value()
        == init_data.agent_settings.llm.api_key.get_secret_value()
    )
    assert (
        loaded_data.agent_settings.llm.base_url == init_data.agent_settings.llm.base_url
    )


@pytest.mark.asyncio
async def test_get_instance(tmp_path):
    config = OpenHandsConfig(file_store='local', file_store_path=str(tmp_path))

    store = await FileSettingsStore.get_instance(config, None)

    assert isinstance(store, FileSettingsStore)
    assert store.root_dir == tmp_path
