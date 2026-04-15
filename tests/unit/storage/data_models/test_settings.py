import warnings
from unittest.mock import patch

from fastmcp.mcp_config import MCPConfig
from pydantic import SecretStr

from openhands.core.config.llm_config import LLMConfig
from openhands.core.config.openhands_config import OpenHandsConfig
from openhands.core.config.sandbox_config import SandboxConfig
from openhands.core.config.security_config import SecurityConfig
from openhands.sdk.llm import LLM
from openhands.sdk.settings import (
    AGENT_SETTINGS_SCHEMA_VERSION,
    AgentSettings,
    ConversationSettings,
)
from openhands.sdk.settings.model import CondenserSettings, VerificationSettings
from openhands.storage.data_models.settings import Settings


def test_settings_from_config():
    mock_app_config = OpenHandsConfig(
        default_agent='test-agent',
        max_iterations=100,
        security=SecurityConfig(
            security_analyzer='llm',
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
        assert settings.agent_settings.agent == 'test-agent'
        assert settings.conversation_settings.max_iterations == 100
        assert settings.conversation_settings.security_analyzer == 'llm'
        assert settings.conversation_settings.confirmation_mode is True
        assert settings.agent_settings.llm.model == 'test-model'
        assert settings.agent_settings.llm.api_key.get_secret_value() == 'test-key'
        assert settings.agent_settings.llm.base_url == 'https://test.example.com'
        assert settings.remote_runtime_resource_factor == 2
        assert not settings.secrets_store.provider_tokens


def test_settings_from_config_no_api_key():
    mock_app_config = OpenHandsConfig(
        default_agent='test-agent',
        max_iterations=100,
        security=SecurityConfig(
            security_analyzer='llm',
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
        agent_settings=AgentSettings(
            agent='test-agent',
            llm=LLM(
                model='test-model',
                api_key=SecretStr('test-key'),
                base_url='https://test.example.com',
            ),
        ),
        conversation_settings=ConversationSettings(
            max_iterations=100,
            security_analyzer='llm',
            confirmation_mode=True,
        ),
        remote_runtime_resource_factor=2,
    )

    llm_api_key = settings.agent_settings.llm.api_key
    assert str(llm_api_key) == '**********'
    assert llm_api_key.get_secret_value() == 'test-key'


def test_settings_update_deep_merges_agent_settings():
    """Updating agent_settings with a partial dict must not overwrite sibling sub-fields."""
    settings = Settings(
        agent_settings=AgentSettings(
            llm=LLM(model='existing-model', api_key=SecretStr('existing-key')),
            condenser=CondenserSettings(enabled=True, max_size=200),
        ),
    )

    settings.update({'agent_settings': {'condenser': {'max_size': 300}}})

    assert settings.agent_settings.llm.model == 'existing-model'
    assert settings.agent_settings.llm.api_key.get_secret_value() == 'existing-key'
    assert settings.agent_settings.condenser.max_size == 300
    assert settings.agent_settings.condenser.enabled is True


def test_settings_preserve_agent_settings():
    settings = Settings(
        agent_settings=AgentSettings(
            llm=LLM(
                model='test-model',
                api_key=SecretStr('test-key'),
                litellm_extra_body={'metadata': {'tier': 'pro'}},
            ),
            verification=VerificationSettings(
                critic_enabled=True,
                critic_mode='all_actions',
            ),
        ),
    )

    assert settings.agent_settings.llm.api_key.get_secret_value() == 'test-key'
    dump = settings.agent_settings.model_dump(
        mode='json', context={'expose_secrets': True}
    )

    assert dump['schema_version'] == AGENT_SETTINGS_SCHEMA_VERSION
    assert dump['llm']['model'] == 'test-model'
    assert dump['llm']['api_key'] == 'test-key'
    assert dump['verification']['critic_enabled'] is True
    assert dump['verification']['critic_mode'] == 'all_actions'
    assert dump['llm']['litellm_extra_body'] == {'metadata': {'tier': 'pro'}}


def test_settings_to_agent_settings_uses_agent_vals():
    settings = Settings(
        agent_settings=AgentSettings(
            llm=LLM(
                model='sdk-model',
                base_url='https://sdk.example.com',
                litellm_extra_body={'metadata': {'tier': 'enterprise'}},
            ),
            condenser=CondenserSettings(enabled=False, max_size=88),
            verification=VerificationSettings(
                critic_enabled=True, critic_mode='all_actions'
            ),
        ),
    )

    agent_settings = settings.to_agent_settings()

    assert agent_settings.llm.model == 'sdk-model'
    assert agent_settings.llm.base_url == 'https://sdk.example.com'
    assert agent_settings.llm.litellm_extra_body == {'metadata': {'tier': 'enterprise'}}
    assert agent_settings.condenser.enabled is False
    assert agent_settings.condenser.max_size == 88
    assert agent_settings.verification.critic_enabled is True
    assert agent_settings.verification.critic_mode == 'all_actions'


def test_settings_agent_settings_keeps_sdk_mcp_shape_canonical():
    settings = Settings(
        agent_settings=AgentSettings(
            llm=LLM(model='sdk-model'),
            mcp_config=MCPConfig(
                mcpServers={
                    'sse_server': {
                        'url': 'https://example.com/sse',
                        'transport': 'sse',
                    }
                },
            ),
        ),
    )

    mcp_config = settings.agent_settings.mcp_config
    assert mcp_config is not None
    servers = mcp_config.mcpServers
    assert 'sse_server' in servers
    assert servers['sse_server'].transport == 'sse'
    assert servers['sse_server'].url == 'https://example.com/sse'

    api_values = settings.agent_settings.model_dump(mode='json')
    assert 'sse_server' in api_values['mcp_config']['mcpServers']


def test_settings_update_mcp_config():
    settings = Settings(agent_settings=AgentSettings(llm=LLM(model='sdk-model')))

    settings.update(
        {
            'agent_settings': {
                'mcp_config': {
                    'mcpServers': {
                        'custom': {
                            'transport': 'http',
                            'url': 'https://example.com/mcp',
                        }
                    }
                }
            }
        }
    )

    mcp = settings.agent_settings.mcp_config
    assert mcp is not None
    assert 'custom' in mcp.mcpServers
    assert mcp.mcpServers['custom'].transport == 'http'
    assert mcp.mcpServers['custom'].url == 'https://example.com/mcp'


def test_settings_update_replaces_existing_mcp_servers():
    settings = Settings(
        agent_settings=AgentSettings(
            llm=LLM(model='sdk-model'),
            mcp_config=MCPConfig(
                mcpServers={
                    'stale': {
                        'transport': 'sse',
                        'url': 'https://example.com/stale',
                    }
                }
            ),
        )
    )

    settings.update(
        {
            'agent_settings': {
                'mcp_config': {
                    'mcpServers': {
                        'fresh': {
                            'transport': 'http',
                            'url': 'https://example.com/fresh',
                        }
                    }
                }
            }
        }
    )

    mcp = settings.agent_settings.mcp_config
    assert mcp is not None
    assert set(mcp.mcpServers) == {'fresh'}
    assert mcp.mcpServers['fresh'].url == 'https://example.com/fresh'


def test_settings_update_can_clear_mcp_config():
    settings = Settings(
        agent_settings=AgentSettings(
            llm=LLM(model='sdk-model'),
            mcp_config=MCPConfig(
                mcpServers={
                    'custom': {
                        'transport': 'http',
                        'url': 'https://example.com/mcp',
                    }
                }
            ),
        )
    )

    settings.update({'agent_settings': {'mcp_config': None}})

    assert settings.agent_settings.mcp_config is None


def test_settings_update_batch():
    settings = Settings()
    settings.update(
        {
            'language': 'fr',
            'agent_settings': {
                'agent': 'TestAgent',
                'llm': {'model': 'new-model', 'api_key': 'new-key'},
            },
            'conversation_settings': {
                'max_iterations': 200,
            },
        }
    )
    assert settings.language == 'fr'
    assert settings.agent_settings.agent == 'TestAgent'
    assert settings.agent_settings.llm.model == 'new-model'
    assert settings.agent_settings.llm.api_key.get_secret_value() == 'new-key'
    assert settings.conversation_settings.max_iterations == 200


def test_settings_no_pydantic_frozen_field_warning():
    """Test that Settings model does not trigger Pydantic UnsupportedFieldAttributeWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')

        import importlib

        import openhands.storage.data_models.settings

        importlib.reload(openhands.storage.data_models.settings)

        frozen_warnings = [
            warning for warning in w if 'frozen' in str(warning.message).lower()
        ]

        assert len(frozen_warnings) == 0, (
            f'Pydantic frozen field warnings found: {[str(w.message) for w in frozen_warnings]}'
        )
