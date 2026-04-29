import importlib
import warnings
from unittest.mock import patch

import pytest
from fastmcp.mcp_config import MCPConfig
from pydantic import SecretStr

import openhands.app_server.settings.settings_models as settings_module
from openhands.app_server.settings.llm_profiles import ProfileNotFoundError
from openhands.app_server.settings.settings_models import Settings
from openhands.app_server.settings.settings_router import LITE_LLM_API_URL
from openhands.core.config.llm_config import LLMConfig
from openhands.core.config.openhands_config import OpenHandsConfig
from openhands.core.config.sandbox_config import SandboxConfig
from openhands.sdk.llm import LLM
from openhands.sdk.settings import (
    AGENT_SETTINGS_SCHEMA_VERSION,
    AgentSettings,
    ConversationSettings,
)
from openhands.sdk.settings.model import CondenserSettings, VerificationSettings


def test_settings_from_config():
    mock_app_config = OpenHandsConfig(
        default_agent='test-agent',
        max_iterations=100,
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
        'openhands.app_server.settings.settings_models.load_openhands_config',
        return_value=mock_app_config,
    ):
        settings = Settings.from_config()

        assert settings is not None
        assert settings.language == 'en'
        assert settings.agent_settings.agent == 'test-agent'
        assert settings.conversation_settings.max_iterations == 100
        assert settings.conversation_settings.security_analyzer is None
        assert settings.conversation_settings.confirmation_mode is False
        assert settings.agent_settings.llm.model == 'test-model'
        assert settings.agent_settings.llm.api_key.get_secret_value() == 'test-key'
        assert settings.agent_settings.llm.base_url == 'https://test.example.com'
        assert settings.remote_runtime_resource_factor == 2
        assert not settings.secrets_store.provider_tokens


def test_settings_from_config_no_api_key():
    mock_app_config = OpenHandsConfig(
        default_agent='test-agent',
        max_iterations=100,
        llms={
            'llm': LLMConfig(
                model='test-model', api_key=None, base_url='https://test.example.com'
            )
        },
        sandbox=SandboxConfig(remote_runtime_resource_factor=2),
    )

    with patch(
        'openhands.app_server.settings.settings_models.load_openhands_config',
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

    settings.update({'agent_settings_diff': {'condenser': {'max_size': 300}}})

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
            'agent_settings_diff': {
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
            'agent_settings_diff': {
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

    settings.update({'agent_settings_diff': {'mcp_config': None}})

    assert settings.agent_settings.mcp_config is None


def test_settings_update_batch():
    settings = Settings()
    settings.update(
        {
            'language': 'fr',
            'agent_settings_diff': {
                'agent': 'TestAgent',
                'llm': {'model': 'new-model', 'api_key': 'new-key'},
            },
            'conversation_settings_diff': {
                'max_iterations': 200,
            },
        }
    )
    assert settings.language == 'fr'
    assert settings.agent_settings.agent == 'TestAgent'
    assert settings.agent_settings.llm.model == 'new-model'
    assert settings.agent_settings.llm.api_key.get_secret_value() == 'new-key'
    assert settings.conversation_settings.max_iterations == 200


# ── LLM profiles: Settings-integration tests ────────────────────────
# Pure LLMProfiles behaviour lives in test_llm_profiles.py.


def test_switch_to_profile_updates_agent_settings_llm():
    settings = Settings()
    settings.llm_profiles.save('my-profile', LLM(model='openai/gpt-4o'))

    settings.switch_to_profile('my-profile')

    assert settings.agent_settings.llm.model == 'openai/gpt-4o'
    assert settings.llm_profiles.active == 'my-profile'


def test_switch_to_nonexistent_profile_raises():
    settings = Settings()

    with pytest.raises(ProfileNotFoundError) as exc_info:
        settings.switch_to_profile('nonexistent')

    assert exc_info.value.name == 'nonexistent'
    assert settings.llm_profiles.active is None


def test_llm_profiles_masking_and_roundtrip():
    """Masked by default, exposed with context, and reconstructible via ``model_validate``."""
    settings = Settings()
    settings.llm_profiles.save(
        'p', LLM(model='openai/gpt-4o', api_key=SecretStr('secret'))
    )

    masked = settings.model_dump(mode='json')
    exposed = settings.model_dump(mode='json', context={'expose_secrets': True})
    assert masked['llm_profiles']['profiles']['p']['api_key'] != 'secret'
    assert exposed['llm_profiles']['profiles']['p']['api_key'] == 'secret'

    rehydrated = Settings.model_validate(exposed)
    assert rehydrated.llm_profiles.get('p').api_key.get_secret_value() == 'secret'


def test_switch_to_profile_preserves_other_agent_settings():
    """Switching the LLM must not wipe condenser/verification/mcp_config.

    Real user: has condenser+verification configured, switches LLM profile —
    expects everything else to stay. A bare-field reassign in
    ``switch_to_profile`` would silently drop those sibling configs.
    """
    settings = Settings(
        agent_settings=AgentSettings(
            llm=LLM(model='openai/gpt-4o'),
            condenser=CondenserSettings(enabled=True, max_size=321),
            verification=VerificationSettings(
                critic_enabled=True, critic_mode='all_actions'
            ),
            mcp_config=MCPConfig(
                mcpServers={
                    's': {
                        'transport': 'http',
                        'url': 'https://example.com/mcp',
                    }
                }
            ),
        ),
    )
    settings.llm_profiles.save('p', LLM(model='anthropic/claude-opus-4'))

    settings.switch_to_profile('p')

    assert settings.agent_settings.llm.model == 'anthropic/claude-opus-4'
    assert settings.agent_settings.condenser.max_size == 321
    assert settings.agent_settings.verification.critic_mode == 'all_actions'
    assert settings.agent_settings.mcp_config is not None
    assert 's' in settings.agent_settings.mcp_config.mcpServers


def test_update_ignores_llm_profiles_payload():
    """``Settings.update`` refuses to mutate ``llm_profiles``; profile changes
    must go through the dedicated endpoints (which enforce name rules, the
    count cap, and the per-user lock)."""
    settings = Settings()

    settings.update(
        {
            'llm_profiles': {
                'profiles': {'X': {'model': 'openai/gpt-4o'}},
                'active': 'X',
            }
        }
    )

    assert settings.llm_profiles.profiles == {}
    assert settings.llm_profiles.active is None


def test_update_clears_active_when_llm_diverges():
    """Editing agent_settings.llm via ``update`` must drop a now-stale active profile."""
    settings = Settings(
        agent_settings=AgentSettings(
            llm=LLM(model='openai/gpt-4o', api_key=SecretStr('sk-a'))
        )
    )
    settings.llm_profiles.save(
        'p', LLM(model='openai/gpt-4o', api_key=SecretStr('sk-a'))
    )
    settings.switch_to_profile('p')
    assert settings.llm_profiles.active == 'p'

    settings.update(
        {'agent_settings_diff': {'llm': {'model': 'anthropic/claude-opus-4'}}}
    )

    assert settings.llm_profiles.active is None


def test_update_keeps_active_when_llm_unchanged():
    """A no-op LLM update must not spuriously clear ``active``."""
    settings = Settings(
        agent_settings=AgentSettings(
            llm=LLM(model='openai/gpt-4o', api_key=SecretStr('sk-a'))
        )
    )
    settings.llm_profiles.save(
        'p', LLM(model='openai/gpt-4o', api_key=SecretStr('sk-a'))
    )
    settings.switch_to_profile('p')

    # Update an unrelated field.
    settings.update({'language': 'fr'})

    assert settings.llm_profiles.active == 'p'


def test_settings_update_batch_accepts_diff_keys():
    settings = Settings()
    settings.update(
        {
            'agent_settings_diff': {
                'agent': 'DiffAgent',
                'llm': {'model': 'diff-model', 'api_key': 'diff-key'},
            },
            'conversation_settings_diff': {
                'max_iterations': 123,
            },
        }
    )

    assert settings.agent_settings.agent == 'DiffAgent'
    assert settings.agent_settings.llm.model == 'diff-model'
    assert settings.agent_settings.llm.api_key.get_secret_value() == 'diff-key'
    assert settings.conversation_settings.max_iterations == 123


def test_settings_update_rejects_legacy_nested_keys():
    settings = Settings()

    with pytest.raises(ValueError, match=r'Use \*_diff nested settings payloads'):
        settings.update({'agent_settings': {'agent': 'LegacyAgent'}})


def test_settings_no_pydantic_frozen_field_warning():
    """Test that Settings model does not trigger Pydantic UnsupportedFieldAttributeWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        importlib.reload(settings_module)

        frozen_warnings = [
            warning for warning in w if 'frozen' in str(warning.message).lower()
        ]

        assert len(frozen_warnings) == 0, (
            f'Pydantic frozen field warnings found: {[str(w.message) for w in frozen_warnings]}'
        )


def test_litellm_proxy_to_openhands_conversion_with_openhands_proxy():
    """Test that litellm_proxy/ is converted to openhands/ when using OpenHands proxy."""
    settings = Settings(
        agent_settings=AgentSettings(
            llm=LLM(
                model='litellm_proxy/claude-opus-4-5-20251101',
                base_url=LITE_LLM_API_URL,
            )
        )
    )

    # Internal representation should be litellm_proxy/
    assert settings.agent_settings.llm.model == 'litellm_proxy/claude-opus-4-5-20251101'

    # Display representation should convert to openhands/
    api_data = settings.get_agent_settings_display()
    assert api_data['llm']['model'] == 'openhands/claude-opus-4-5-20251101'


def test_litellm_proxy_custom_endpoint_keeps_prefix():
    """Test that custom litellm_proxy endpoints keep their litellm_proxy/ prefix."""
    settings = Settings(
        agent_settings=AgentSettings(
            llm=LLM(
                model='litellm_proxy/gpt-5.3-codex',
                base_url='http://custom-proxy.example.com:4000',
            )
        )
    )

    # Internal representation
    assert settings.agent_settings.llm.model == 'litellm_proxy/gpt-5.3-codex'

    # Display should NOT convert to openhands/ because it's a custom endpoint
    api_data = settings.get_agent_settings_display()
    assert api_data['llm']['model'] == 'litellm_proxy/gpt-5.3-codex'


def test_openhands_model_converts_to_litellm_proxy_internally():
    """Test that openhands/ models are stored as litellm_proxy/ internally."""
    settings = Settings(
        agent_settings=AgentSettings(
            llm=LLM(model='openhands/claude-opus-4-5-20251101')
        )
    )

    # Internal representation should be litellm_proxy/
    assert settings.agent_settings.llm.model == 'litellm_proxy/claude-opus-4-5-20251101'

    # Display representation should convert back to openhands/
    api_data = settings.get_agent_settings_display()
    assert api_data['llm']['model'] == 'openhands/claude-opus-4-5-20251101'
