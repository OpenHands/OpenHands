import warnings
from unittest.mock import patch

from pydantic import SecretStr

from openhands.core.config.llm_config import LLMConfig
from openhands.core.config.openhands_config import OpenHandsConfig
from openhands.core.config.sandbox_config import SandboxConfig
from openhands.core.config.security_config import SecurityConfig
from openhands.server.routes.settings import convert_to_settings
from openhands.storage.data_models.settings import Settings


def test_settings_from_config():
    # Mock configuration
    mock_app_config = OpenHandsConfig(
        default_agent='test-agent',
        max_iterations=100,
        security=SecurityConfig(
            security_analyzer='test-analyzer',
            confirmation_mode=True,
        ),
        llms={
            'llm': LLMConfig(
                model='test-model',
                api_key=SecretStr('test-key'),
                base_url='https://test.example.com',
            )
        },
        sandbox=SandboxConfig(remote_runtime_resource_factor=2),
    )

    with patch(
        'openhands.storage.data_models.settings.load_openhands_config',
        return_value=mock_app_config,
    ):
        settings = Settings.from_config()

        assert settings is not None
        assert settings.language == 'en'
        assert settings.agent == 'test-agent'
        assert settings.max_iterations == 100
        assert settings.security_analyzer == 'test-analyzer'
        assert settings.confirmation_mode is True
        assert settings.llm_model == 'test-model'
        assert settings.llm_api_key.get_secret_value() == 'test-key'
        assert settings.llm_base_url == 'https://test.example.com'
        assert settings.remote_runtime_resource_factor == 2
        assert not settings.secrets_store.provider_tokens


def test_settings_from_config_no_api_key():
    # Mock configuration without API key
    mock_app_config = OpenHandsConfig(
        default_agent='test-agent',
        max_iterations=100,
        security=SecurityConfig(
            security_analyzer='test-analyzer',
            confirmation_mode=True,
        ),
        llms={
            'llm': LLMConfig(
                model='test-model', api_key=None, base_url='https://test.example.com'
            )
        },
        sandbox=SandboxConfig(remote_runtime_resource_factor=2),
    )

    with patch(
        'openhands.storage.data_models.settings.load_openhands_config',
        return_value=mock_app_config,
    ):
        settings = Settings.from_config()
        assert settings is None


def test_settings_handles_sensitive_data():
    settings = Settings(
        language='en',
        agent='test-agent',
        max_iterations=100,
        security_analyzer='test-analyzer',
        confirmation_mode=True,
        llm_model='test-model',
        llm_api_key='test-key',
        llm_base_url='https://test.example.com',
        remote_runtime_resource_factor=2,
    )

    assert str(settings.llm_api_key) == '**********'
    assert settings.llm_api_key.get_secret_value() == 'test-key'


def test_convert_to_settings():
    settings_with_token_data = Settings(
        llm_api_key='test-key',
    )

    settings = convert_to_settings(settings_with_token_data)

    assert settings.llm_api_key.get_secret_value() == 'test-key'


def test_settings_no_pydantic_frozen_field_warning():
    """Test that Settings model does not trigger Pydantic UnsupportedFieldAttributeWarning.

    This test ensures that the 'frozen' parameter is not incorrectly used in Field()
    definitions, which would cause warnings in Pydantic v2 for union types.
    See: https://github.com/All-Hands-AI/infra/issues/860
    """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')

        # Re-import to trigger any warnings during model definition
        import importlib

        import openhands.storage.data_models.settings

        importlib.reload(openhands.storage.data_models.settings)

        # Check for warnings containing 'frozen' which would indicate
        # incorrect usage of frozen=True in Field()
        frozen_warnings = [
            warning for warning in w if 'frozen' in str(warning.message).lower()
        ]

        assert len(frozen_warnings) == 0, (
            f'Pydantic frozen field warnings found: {[str(w.message) for w in frozen_warnings]}'
        )
