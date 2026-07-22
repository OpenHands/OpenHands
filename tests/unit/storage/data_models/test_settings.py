import importlib
import warnings
from unittest.mock import patch

import pytest
from pydantic import SecretStr, ValidationError

import openhands.app_server.settings.settings_models as settings_module
from openhands.app_server.settings.llm_profiles import ProfileNotFoundError
from openhands.app_server.settings.settings_models import (
    GETSettingsModel,
    MarketplaceRegistration,
    Settings,
)
from openhands.app_server.settings.settings_router import LITE_LLM_API_URL
from openhands.sdk.llm import LLM
from openhands.sdk.mcp.config import dump_mcp_config
from openhands.sdk.settings import (
    AGENT_SETTINGS_SCHEMA_VERSION,
    ConversationSettings,
    OpenHandsAgentSettings,
)
from openhands.sdk.settings.model import CondenserSettings, VerificationSettings


def test_settings_handles_sensitive_data():
    settings = Settings(
        language='en',
        agent_settings=OpenHandsAgentSettings(
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


def test_settings_loads_persisted_settings_via_sdk_loaders():
    loaded_agent_settings = OpenHandsAgentSettings(agent='migrated-agent')
    loaded_conversation_settings = ConversationSettings(max_iterations=77)

    with (
        patch.object(
            settings_module,
            'validate_agent_settings',
            return_value=loaded_agent_settings,
        ) as agent_loader,
        patch.object(
            ConversationSettings,
            'from_persisted',
            return_value=loaded_conversation_settings,
        ) as conversation_loader,
    ):
        settings = Settings(
            agent_settings={'legacy': True},
            conversation_settings={'legacy': True},
        )

    agent_loader.assert_called_once_with({'legacy': True})
    conversation_loader.assert_called_once_with({'legacy': True})
    assert settings.agent_settings.agent == 'migrated-agent'
    assert settings.conversation_settings.max_iterations == 77


def test_settings_update_deep_merges_agent_settings():
    """Updating agent_settings with a partial dict must not overwrite sibling sub-fields."""
    settings = Settings(
        agent_settings=OpenHandsAgentSettings(
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
        agent_settings=OpenHandsAgentSettings(
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
        agent_settings=OpenHandsAgentSettings(
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
        agent_settings=OpenHandsAgentSettings(
            llm=LLM(model='sdk-model'),
            mcp_config={
                'sse_server': {
                    'url': 'https://example.com/sse',
                    'transport': 'sse',
                }
            },
        ),
    )

    mcp_config = settings.agent_settings.mcp_config
    assert mcp_config is not None
    assert 'sse_server' in mcp_config
    assert mcp_config['sse_server'].transport == 'sse'
    assert mcp_config['sse_server'].url == 'https://example.com/sse'

    api_values = settings.agent_settings.model_dump(mode='json')
    assert 'sse_server' in api_values['mcp_config']


def test_settings_update_mcp_config():
    settings = Settings(
        agent_settings=OpenHandsAgentSettings(llm=LLM(model='sdk-model'))
    )

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
    assert 'custom' in mcp
    assert mcp['custom'].transport == 'http'
    assert mcp['custom'].url == 'https://example.com/mcp'


def test_settings_update_replaces_existing_mcp_servers():
    settings = Settings(
        agent_settings=OpenHandsAgentSettings(
            llm=LLM(model='sdk-model'),
            mcp_config={
                'stale': {
                    'transport': 'sse',
                    'url': 'https://example.com/stale',
                }
            },
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
                            'tools': ['search'],
                        }
                    }
                }
            }
        }
    )

    mcp = settings.agent_settings.mcp_config
    assert mcp is not None
    assert set(mcp) == {'fresh'}
    assert mcp['fresh'].url == 'https://example.com/fresh'


def _settings_with_mcp_auth(
    url: str = 'https://integration.example.com/mcp',
) -> Settings:
    return Settings(
        agent_settings=OpenHandsAgentSettings(
            mcp_config={
                'integration-hub': {
                    'url': url,
                    'headers': {'Authorization': 'Bearer real-key'},
                }
            }
        )
    )


def test_settings_update_preserves_redacted_mcp_auth_for_same_endpoint():
    settings = _settings_with_mcp_auth()

    settings.update(
        {
            'agent_settings_diff': {
                'mcp_config': {
                    'integration-hub': {
                        'url': 'https://integration.example.com/mcp',
                        'auth': {'strategy': 'bearer', 'value': '**********'},
                    }
                }
            }
        }
    )

    server = settings.agent_settings.mcp_config['integration-hub']
    assert server.auth is not None
    assert server.auth.to_http_headers() == {'Authorization': 'Bearer real-key'}


def test_settings_update_preserves_redacted_mcp_header_for_same_endpoint():
    settings = _settings_with_mcp_auth()

    settings.update(
        {
            'agent_settings_diff': {
                'mcp_config': {
                    'integration-hub': {
                        'url': 'https://integration.example.com/mcp',
                        'headers': {'Authorization': 'Bearer **********'},
                    }
                }
            }
        }
    )

    server = settings.agent_settings.mcp_config['integration-hub']
    assert server.auth is not None
    assert server.auth.to_http_headers() == {'Authorization': 'Bearer real-key'}


@pytest.mark.parametrize(
    'auth',
    (
        {'strategy': 'api_key', 'value': 'real-key', 'header_name': 'X-API-Key'},
        {'strategy': 'basic', 'username': 'user', 'password': 'real-key'},
        {'strategy': 'header', 'headers': {'X-API-Key': 'real-key'}},
        {
            'strategy': 'oauth2',
            'state': {'tokens': {'access_token': 'real-key'}},
        },
    ),
)
def test_settings_update_preserves_typed_auth_from_redacted_bearer(
    auth: dict[str, object],
):
    settings = Settings(
        agent_settings=OpenHandsAgentSettings(
            mcp_config={
                'server': {
                    'url': 'https://integration.example.com/mcp',
                    'auth': auth,
                }
            }
        )
    )

    settings.update(
        {
            'agent_settings_diff': {
                'mcp_config': {
                    'server': {
                        'url': 'https://integration.example.com/mcp',
                        'auth': {'strategy': 'bearer', 'value': '**********'},
                    }
                }
            }
        }
    )

    dumped = dump_mcp_config(
        settings.agent_settings.mcp_config or {},
        context={'expose_secrets': 'plaintext'},
    )
    assert dumped['server']['auth'] == auth


def test_settings_update_preserves_custom_headers_with_typed_auth():
    settings = Settings(
        agent_settings=OpenHandsAgentSettings(
            mcp_config={
                'server': {
                    'url': 'https://integration.example.com/mcp',
                    'headers': {'X-Tenant': 'tenant-secret'},
                    'auth': {'strategy': 'bearer', 'value': 'real-key'},
                }
            }
        )
    )

    settings.update(
        {
            'agent_settings_diff': {
                'mcp_config': {
                    'server': {
                        'url': 'https://integration.example.com/mcp',
                        'auth': {'strategy': 'bearer', 'value': '**********'},
                    }
                }
            }
        }
    )

    dumped = dump_mcp_config(
        settings.agent_settings.mcp_config or {},
        context={'expose_secrets': 'plaintext'},
    )
    assert dumped['server']['headers'] == {'X-Tenant': 'tenant-secret'}
    assert dumped['server']['auth'] == {'strategy': 'bearer', 'value': 'real-key'}


@pytest.mark.parametrize(
    'credential_update',
    (
        {},
        {'auth': {'strategy': 'bearer', 'value': '**********'}},
        {'headers': {'Authorization': 'Bearer **********'}},
    ),
)
def test_settings_update_drops_mcp_auth_for_changed_endpoint(
    credential_update: dict[str, object],
):
    settings = _settings_with_mcp_auth('https://old.example.com/mcp')

    settings.update(
        {
            'agent_settings_diff': {
                'mcp_config': {
                    'integration-hub': {
                        'url': 'https://new.example.com/mcp',
                        **credential_update,
                    }
                }
            }
        }
    )

    server = settings.agent_settings.mcp_config['integration-hub']
    assert not server.headers
    assert server.auth is None or server.auth.to_http_headers() == {}


def test_settings_update_preserves_omitted_mcp_auth_for_same_endpoint():
    settings = _settings_with_mcp_auth()

    settings.update(
        {
            'agent_settings_diff': {
                'mcp_config': {
                    'integration-hub': {
                        'url': 'https://integration.example.com/mcp',
                    }
                }
            }
        }
    )

    server = settings.agent_settings.mcp_config['integration-hub']
    assert server.auth is not None
    assert server.auth.to_http_headers() == {'Authorization': 'Bearer real-key'}


def test_settings_update_clears_explicit_mcp_auth():
    settings = _settings_with_mcp_auth()

    settings.update(
        {
            'agent_settings_diff': {
                'mcp_config': {
                    'integration-hub': {
                        'url': 'https://integration.example.com/mcp',
                        'auth': None,
                    }
                }
            }
        }
    )

    server = settings.agent_settings.mcp_config['integration-hub']
    assert server.auth is None


def test_settings_update_clears_auth_without_clearing_custom_headers():
    settings = Settings(
        agent_settings=OpenHandsAgentSettings(
            mcp_config={
                'server': {
                    'url': 'https://integration.example.com/mcp',
                    'headers': {'X-Tenant': 'tenant-secret'},
                    'auth': {'strategy': 'bearer', 'value': 'real-key'},
                }
            }
        )
    )

    settings.update(
        {
            'agent_settings_diff': {
                'mcp_config': {
                    'server': {
                        'url': 'https://integration.example.com/mcp',
                        'auth': None,
                    }
                }
            }
        }
    )

    dumped = dump_mcp_config(
        settings.agent_settings.mcp_config or {},
        context={'expose_secrets': 'plaintext'},
    )
    assert dumped['server']['headers'] == {'X-Tenant': 'tenant-secret'}
    assert 'auth' not in dumped['server']


def test_settings_update_preserves_redacted_mcp_env_for_same_command():
    settings = Settings(
        agent_settings=OpenHandsAgentSettings(
            mcp_config={
                'local': {
                    'command': 'mcp-server',
                    'args': ['--stdio'],
                    'env': {'API_KEY': 'real-key'},
                }
            }
        )
    )

    settings.update(
        {
            'agent_settings_diff': {
                'mcp_config': {
                    'local': {
                        'command': 'mcp-server',
                        'args': ['--stdio'],
                        'env': {'API_KEY': '**********'},
                    }
                }
            }
        }
    )

    server = settings.agent_settings.mcp_config['local']
    assert server.env is not None
    assert server.env['API_KEY'].get_secret_value() == 'real-key'

    settings.update(
        {
            'agent_settings_diff': {
                'mcp_config': {
                    'local': {
                        'command': 'mcp-server',
                        'args': ['--stdio'],
                        'env': {},
                    }
                }
            }
        }
    )

    assert not settings.agent_settings.mcp_config['local'].env


def test_settings_update_preserves_redacted_mcp_env_across_rename():
    settings = Settings(
        agent_settings=OpenHandsAgentSettings(
            mcp_config={
                'old-name': {
                    'command': 'mcp-server',
                    'args': ['--stdio'],
                    'env': {'API_KEY': 'real-key'},
                }
            }
        )
    )

    settings.update(
        {
            'agent_settings_diff': {
                'mcp_config': {
                    'new-name': {
                        'command': 'mcp-server',
                        'args': ['--stdio'],
                        'env': {'API_KEY': '**********'},
                    }
                }
            }
        }
    )

    server = settings.agent_settings.mcp_config['new-name']
    assert server.env is not None
    assert server.env['API_KEY'].get_secret_value() == 'real-key'


def test_settings_update_does_not_copy_mcp_env_to_duplicate_server():
    settings = Settings(
        agent_settings=OpenHandsAgentSettings(
            mcp_config={
                'original': {
                    'command': 'mcp-server',
                    'args': ['--stdio'],
                    'env': {'API_KEY': 'real-key'},
                }
            }
        )
    )

    settings.update(
        {
            'agent_settings_diff': {
                'mcp_config': {
                    'original': {
                        'command': 'mcp-server',
                        'args': ['--stdio'],
                    },
                    'duplicate': {
                        'command': 'mcp-server',
                        'args': ['--stdio'],
                        'env': {'API_KEY': '**********'},
                    },
                }
            }
        }
    )

    original = settings.agent_settings.mcp_config['original']
    duplicate = settings.agent_settings.mcp_config['duplicate']
    assert original.env is not None
    assert original.env['API_KEY'].get_secret_value() == 'real-key'
    assert not duplicate.env


def test_settings_update_drops_redacted_mcp_env_when_args_change():
    settings = Settings(
        agent_settings=OpenHandsAgentSettings(
            mcp_config={
                'local': {
                    'command': 'npx',
                    'args': ['first-package'],
                    'env': {'API_KEY': 'real-key'},
                }
            }
        )
    )

    settings.update(
        {
            'agent_settings_diff': {
                'mcp_config': {
                    'local': {
                        'command': 'npx',
                        'args': ['different-package'],
                        'env': {'API_KEY': '**********'},
                    }
                }
            }
        }
    )

    assert not settings.agent_settings.mcp_config['local'].env


def _mcp_config_as_seen_by_frontend(settings: Settings) -> dict:
    """The ``mcp_config`` the frontend receives from ``GET /settings``.

    Mirrors ``settings_router.load_settings``: the response is built as
    ``GETSettingsModel(**settings.model_dump(...))`` and then serialized. That
    dump-then-revalidate strips every MCP secret to *absent* — redaction emits
    ``"**********"``, ``validate_secret`` turns that marker back into ``None`` on
    the way into ``GETSettingsModel``, and the ``None`` is dropped on the way
    out. So an unchanged credential arrives at the client as e.g.
    ``{'strategy': 'bearer'}`` with no ``value`` — not the ``"**********"``
    sentinel — which is exactly what the client echoes back on the next save.
    """
    response = GETSettingsModel(
        **settings.model_dump(exclude={'secrets_store'}),
        llm_api_key_set=settings.llm_api_key_is_set,
    )
    return response.model_dump(mode='json')['agent_settings']['mcp_config']


def test_get_settings_roundtrip_strips_mcp_auth_secret_to_absent():
    """Document the root cause: the GET response omits the secret entirely."""
    settings = Settings(
        agent_settings=OpenHandsAgentSettings(
            mcp_config={
                'server': {
                    'url': 'https://mcp.example.com',
                    'transport': 'http',
                    'auth': {'strategy': 'bearer', 'value': 'real-key'},
                }
            }
        )
    )

    seen = _mcp_config_as_seen_by_frontend(settings)

    # The secret is gone, not redacted to "**********".
    assert seen['server']['auth'] == {'strategy': 'bearer'}


def test_settings_update_preserves_mcp_auth_across_get_roundtrip_on_add():
    """Adding a server must not wipe existing servers' keys (the reported bug).

    Replays the real flow: read the values as the frontend receives them (secret
    stripped), append a brand-new server with its own key, and save the whole
    map back — exactly the payload the browser sends.
    """
    settings = Settings(
        agent_settings=OpenHandsAgentSettings(
            mcp_config={
                'shttp': {
                    'url': 'https://a.example.com',
                    'transport': 'http',
                    'auth': {'strategy': 'bearer', 'value': 'key-a'},
                },
                'shttp_1': {
                    'url': 'https://b.example.com',
                    'transport': 'http',
                    'auth': {'strategy': 'bearer', 'value': 'key-b'},
                },
            }
        )
    )

    submitted = _mcp_config_as_seen_by_frontend(settings)
    submitted['shttp_2'] = {
        'url': 'https://c.example.com',
        'auth': {'strategy': 'bearer', 'value': 'key-c'},
    }

    settings.update({'agent_settings_diff': {'mcp_config': submitted}})

    dumped = dump_mcp_config(
        settings.agent_settings.mcp_config or {},
        context={'expose_secrets': 'plaintext'},
    )
    assert dumped['shttp']['auth'] == {'strategy': 'bearer', 'value': 'key-a'}
    assert dumped['shttp_1']['auth'] == {'strategy': 'bearer', 'value': 'key-b'}
    assert dumped['shttp_2']['auth'] == {'strategy': 'bearer', 'value': 'key-c'}


@pytest.mark.parametrize(
    'auth,secret_path',
    (
        (
            {'strategy': 'api_key', 'value': 'real-key', 'header_name': 'X-API-Key'},
            ('value',),
        ),
        (
            {'strategy': 'basic', 'username': 'user', 'password': 'real-key'},
            ('password',),
        ),
        (
            {'strategy': 'header', 'headers': {'X-API-Key': 'real-key'}},
            ('headers', 'X-API-Key'),
        ),
        (
            {'strategy': 'oauth2', 'state': {'tokens': {'access_token': 'real-key'}}},
            ('state', 'tokens', 'access_token'),
        ),
    ),
)
def test_settings_update_preserves_typed_mcp_auth_across_get_roundtrip(
    auth: dict, secret_path: tuple[str, ...]
):
    """Every typed strategy survives the stripped-secret round trip on an edit."""
    settings = Settings(
        agent_settings=OpenHandsAgentSettings(
            mcp_config={
                'server': {
                    'url': 'https://mcp.example.com',
                    'transport': 'http',
                    'auth': auth,
                }
            }
        )
    )

    submitted = _mcp_config_as_seen_by_frontend(settings)
    # An unrelated edit (a new server) triggers a full-map save.
    submitted['added'] = {
        'url': 'https://added.example.com',
        'auth': {'strategy': 'bearer', 'value': 'new-key'},
    }

    settings.update({'agent_settings_diff': {'mcp_config': submitted}})

    dumped = dump_mcp_config(
        settings.agent_settings.mcp_config or {},
        context={'expose_secrets': 'plaintext'},
    )
    restored = dumped['server']['auth']
    for key in secret_path:
        restored = restored[key]
    assert restored == 'real-key'


def test_settings_update_preserves_stdio_env_across_get_roundtrip():
    """A stdio server's env secrets survive when another server is added."""
    settings = Settings(
        agent_settings=OpenHandsAgentSettings(
            mcp_config={
                'local': {
                    'command': 'mcp-server',
                    'args': ['--stdio'],
                    'env': {'API_KEY': 'real-key'},
                }
            }
        )
    )

    submitted = _mcp_config_as_seen_by_frontend(settings)
    submitted['added'] = {
        'url': 'https://added.example.com',
        'auth': {'strategy': 'bearer', 'value': 'new-key'},
    }

    settings.update({'agent_settings_diff': {'mcp_config': submitted}})

    server = settings.agent_settings.mcp_config['local']
    assert server.env is not None
    assert server.env['API_KEY'].get_secret_value() == 'real-key'


def test_settings_update_can_clear_mcp_config():
    settings = Settings(
        agent_settings=OpenHandsAgentSettings(
            llm=LLM(model='sdk-model'),
            mcp_config={
                'custom': {
                    'transport': 'http',
                    'url': 'https://example.com/mcp',
                }
            },
        )
    )

    settings.update({'agent_settings_diff': {'mcp_config': None}})

    # The SDK normalizes ``None`` to ``{}`` so cleared configs round-trip as an
    # empty server map rather than ``None``.
    assert settings.agent_settings.mcp_config == {}


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
        agent_settings=OpenHandsAgentSettings(
            llm=LLM(model='openai/gpt-4o'),
            condenser=CondenserSettings(enabled=True, max_size=321),
            verification=VerificationSettings(
                critic_enabled=True, critic_mode='all_actions'
            ),
            mcp_config={
                's': {
                    'transport': 'http',
                    'url': 'https://example.com/mcp',
                }
            },
        ),
    )
    settings.llm_profiles.save('p', LLM(model='anthropic/claude-opus-4'))

    settings.switch_to_profile('p')

    assert settings.agent_settings.llm.model == 'anthropic/claude-opus-4'
    assert settings.agent_settings.condenser.max_size == 321
    assert settings.agent_settings.verification.critic_mode == 'all_actions'
    assert settings.agent_settings.mcp_config is not None
    assert 's' in settings.agent_settings.mcp_config


def test_delete_active_profile_promotes_remaining_one():
    settings = Settings()
    settings.llm_profiles.save('a', LLM(model='openai/gpt-4o'))
    settings.llm_profiles.save('b', LLM(model='anthropic/claude-opus-4'))
    settings.switch_to_profile('a')

    assert settings.delete_profile('a') is True

    assert 'a' not in settings.llm_profiles.profiles
    assert settings.llm_profiles.active == 'b'
    assert settings.agent_settings.llm.model == 'anthropic/claude-opus-4'


def test_delete_inactive_profile_does_not_touch_active():
    settings = Settings()
    settings.llm_profiles.save('a', LLM(model='openai/gpt-4o'))
    settings.llm_profiles.save('b', LLM(model='anthropic/claude-opus-4'))
    settings.switch_to_profile('a')

    assert settings.delete_profile('b') is True

    assert settings.llm_profiles.active == 'a'
    assert settings.agent_settings.llm.model == 'openai/gpt-4o'


def test_delete_only_profile_clears_active():
    settings = Settings()
    settings.llm_profiles.save('only', LLM(model='openai/gpt-4o'))
    settings.switch_to_profile('only')

    assert settings.delete_profile('only') is True

    assert settings.llm_profiles.profiles == {}
    assert settings.llm_profiles.active is None


def test_delete_missing_profile_returns_false():
    settings = Settings()
    assert settings.delete_profile('nope') is False


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
        agent_settings=OpenHandsAgentSettings(
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
        agent_settings=OpenHandsAgentSettings(
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


def test_litellm_proxy_with_openhands_proxy_keeps_prefix_for_display():
    """Display data no longer reverse-maps LiteLLM proxy model names."""
    settings = Settings(
        agent_settings=OpenHandsAgentSettings(
            llm=LLM(
                model='litellm_proxy/claude-opus-4-5-20251101',
                base_url=LITE_LLM_API_URL,
            )
        )
    )

    api_data = settings.get_agent_settings_display()
    assert api_data['llm']['model'] == 'litellm_proxy/claude-opus-4-5-20251101'


def test_litellm_proxy_custom_endpoint_keeps_prefix():
    """Test that custom litellm_proxy endpoints keep their litellm_proxy/ prefix."""
    settings = Settings(
        agent_settings=OpenHandsAgentSettings(
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


def test_openhands_model_display_does_not_reverse_map():
    """Display data reflects the LLM model shape provided by the SDK."""
    settings = Settings(
        agent_settings=OpenHandsAgentSettings(
            llm=LLM(model='openhands/claude-opus-4-5-20251101')
        )
    )

    api_data = settings.get_agent_settings_display()
    assert api_data['llm']['model'] == settings.agent_settings.llm.model


# --- Tests for MarketplaceRegistration ---


class TestMarketplaceRegistration:
    """Tests for MarketplaceRegistration model."""

    def test_basic_registration(self):
        """Test creating a basic marketplace registration."""
        reg = MarketplaceRegistration(
            name='test-marketplace',
            source='github:owner/repo',
        )
        assert reg.name == 'test-marketplace'
        assert reg.source == 'github:owner/repo'
        assert reg.ref is None
        assert reg.repo_path is None
        assert reg.auto_load is False  # Defaults to False, not None

    def test_registration_with_auto_load(self):
        """Test registration with auto_load=True."""
        reg = MarketplaceRegistration(
            name='public',
            source='github:OpenHands/skills',
            auto_load=True,
        )
        assert reg.auto_load

    def test_registration_with_ref(self):
        """Test registration with specific ref."""
        reg = MarketplaceRegistration(
            name='versioned',
            source='github:owner/repo',
            ref='v1.0.0',
        )
        assert reg.ref == 'v1.0.0'

    def test_registration_with_repo_path(self):
        """Test registration with repo_path for monorepos."""
        reg = MarketplaceRegistration(
            name='monorepo-marketplace',
            source='github:acme/monorepo',
            repo_path='marketplaces/internal',
        )
        assert reg.repo_path == 'marketplaces/internal'

    def test_repo_path_validation_rejects_absolute(self):
        """Test that absolute repo_path is rejected."""
        with pytest.raises(ValidationError, match='must be relative'):
            MarketplaceRegistration(
                name='test',
                source='github:owner/repo',
                repo_path='/absolute/path',
            )

    def test_repo_path_validation_rejects_traversal(self):
        """Test that parent directory traversal is rejected."""
        with pytest.raises(ValidationError, match="cannot contain '..'"):
            MarketplaceRegistration(
                name='test',
                source='github:owner/repo',
                repo_path='../escape/path',
            )

    def test_repo_path_validation_rejects_url(self):
        """A repository URL pasted into repo_path is rejected (not a subdir)."""
        with pytest.raises(ValidationError, match='not a URL'):
            MarketplaceRegistration(
                name='test',
                source='github:OpenHands/extensions',
                repo_path='https://github.com/OpenHands/extensions',
            )

    def test_serialization(self):
        """Test that MarketplaceRegistration serializes with standard pydantic."""
        reg = MarketplaceRegistration(
            name='test',
            source='github:owner/repo',
            ref='main',
            repo_path='plugins',
            auto_load=True,
        )
        data = reg.model_dump()
        assert data == {
            'name': 'test',
            'source': 'github:owner/repo',
            'ref': 'main',
            'repo_path': 'plugins',
            'auto_load': True,
            'scope': None,
        }

    def test_serialization_excludes_none_for_wire_payload(self):
        """exclude_none drops unset optional fields (as sent to the agent-server)."""
        reg = MarketplaceRegistration(name='test', source='github:owner/repo')
        assert reg.model_dump(exclude_none=True, exclude={'scope'}) == {
            'name': 'test',
            'source': 'github:owner/repo',
            'auto_load': False,
        }

    # --- Name validation tests ---

    def test_name_validation_rejects_empty(self):
        """Test that empty name is rejected."""
        with pytest.raises(ValidationError, match='name cannot be empty'):
            MarketplaceRegistration(name='', source='github:owner/repo')

    def test_name_validation_rejects_whitespace_only(self):
        """Test that whitespace-only name is rejected."""
        with pytest.raises(ValidationError, match='name cannot be empty'):
            MarketplaceRegistration(name='   ', source='github:owner/repo')

    def test_name_validation_rejects_invalid_chars(self):
        """Test that names with invalid characters are rejected."""
        with pytest.raises(ValidationError, match='must start with a letter'):
            MarketplaceRegistration(name='123-invalid', source='github:owner/repo')

    def test_name_validation_rejects_spaces(self):
        """Test that names with spaces are rejected."""
        with pytest.raises(ValidationError, match='must start with a letter'):
            MarketplaceRegistration(name='invalid name', source='github:owner/repo')

    def test_name_validation_strips_whitespace(self):
        """Test that name whitespace is stripped."""
        reg = MarketplaceRegistration(name='  valid-name  ', source='github:owner/repo')
        assert reg.name == 'valid-name'

    # --- Source validation tests ---

    def test_source_validation_github_format(self):
        """Test valid github:owner/repo source."""
        reg = MarketplaceRegistration(name='test', source='github:my-org/my-repo')
        assert reg.source == 'github:my-org/my-repo'

    def test_source_validation_https_url(self):
        """Test valid HTTPS git URL source."""
        reg = MarketplaceRegistration(
            name='test', source='https://github.com/owner/repo'
        )
        assert reg.source == 'https://github.com/owner/repo'

    def test_source_validation_ssh_url(self):
        """Test valid SSH git URL source."""
        reg = MarketplaceRegistration(name='test', source='git@github.com:owner/repo')
        assert reg.source == 'git@github.com:owner/repo'

    def test_source_validation_bitbucket_dc_personal_https_url(self):
        """Bitbucket DC personal repos live under /scm/~user/ and are valid."""
        source = 'https://bitbucket.dc.example.com/scm/~jane.doe_example.com/skills.git'
        reg = MarketplaceRegistration(name='personal', source=source)
        assert reg.source == source

    def test_source_validation_https_url_with_username(self):
        """Copy-pasted DC clone URLs carry a bare username before the host."""
        source = (
            'https://jane.doe@bitbucket.dc.example.com/scm/~jane.doe_example.com/'
            'skills.git'
        )
        reg = MarketplaceRegistration(name='personal', source=source)
        assert reg.source == source

    def test_source_validation_ssh_url_with_user_port_and_tilde(self):
        """DC SSH clone URLs use ssh://git@host:port/~user/repo.git."""
        source = 'ssh://git@bitbucket.dc.example.com:7999/~jane.doe/skills.git'
        reg = MarketplaceRegistration(name='personal', source=source)
        assert reg.source == source

    def test_source_validation_scp_url_with_tilde(self):
        """SCP-style SSH URLs may target personal (~user) repos."""
        source = 'git@bitbucket.dc.example.com:~jane.doe/skills.git'
        reg = MarketplaceRegistration(name='personal', source=source)
        assert reg.source == source

    def test_source_validation_rejects_embedded_password(self):
        """URLs with user:password credentials stay rejected."""
        with pytest.raises(ValidationError, match='source must be'):
            MarketplaceRegistration(
                name='test',
                source='https://user:secret@host.example.com/scm/proj/repo.git',
            )

    def test_source_validation_rejects_tilde_local_path(self):
        """'~' stays rejected for local paths (no home-dir expansion)."""
        with pytest.raises(ValidationError, match='source must be'):
            MarketplaceRegistration(name='test', source='~/skills')

    def test_source_validation_relative_path(self):
        """Test valid relative local path source."""
        reg = MarketplaceRegistration(name='test', source='local/path/to/skills')
        assert reg.source == 'local/path/to/skills'

    def test_source_validation_rejects_empty(self):
        """Test that empty source is rejected."""
        with pytest.raises(ValidationError, match='source cannot be empty'):
            MarketplaceRegistration(name='test', source='')

    def test_source_validation_rejects_absolute_path(self):
        """Test that absolute local path is rejected."""
        with pytest.raises(ValidationError, match='must be relative'):
            MarketplaceRegistration(name='test', source='/absolute/path')

    def test_source_validation_rejects_parent_traversal(self):
        """Test that parent directory traversal in source is rejected."""
        with pytest.raises(ValidationError, match="cannot contain '..'"):
            MarketplaceRegistration(name='test', source='../escape/path')


# --- Tests for Settings.registered_marketplaces ---


class TestSettingsRegisteredMarketplaces:
    """Tests for registered_marketplaces field in Settings."""

    def test_settings_default_empty_registered_marketplaces(self):
        """Test that Settings defaults to empty registered_marketplaces."""
        settings = Settings()
        assert settings.registered_marketplaces == []

    def test_settings_with_registered_marketplaces(self):
        """Test Settings with registered_marketplaces configured."""
        marketplaces = [
            MarketplaceRegistration(
                name='public',
                source='github:OpenHands/skills',
                auto_load=True,
            ),
            MarketplaceRegistration(
                name='team',
                source='github:acme/plugins',
            ),
        ]
        settings = Settings(registered_marketplaces=marketplaces)

        assert len(settings.registered_marketplaces) == 2
        assert settings.registered_marketplaces[0].name == 'public'
        assert settings.registered_marketplaces[0].auto_load is True
        assert settings.registered_marketplaces[1].name == 'team'
        assert (
            settings.registered_marketplaces[1].auto_load is False
        )  # Defaults to False

    def test_settings_serialization_with_registered_marketplaces(self):
        """Test Settings serialization includes registered_marketplaces."""
        marketplaces = [
            MarketplaceRegistration(
                name='test',
                source='github:owner/repo',
                auto_load=True,
            ),
        ]
        settings = Settings(registered_marketplaces=marketplaces)
        data = settings.model_dump()

        assert 'registered_marketplaces' in data
        assert len(data['registered_marketplaces']) == 1
        assert data['registered_marketplaces'][0]['name'] == 'test'
        assert data['registered_marketplaces'][0]['auto_load']

    def test_settings_from_dict_with_registered_marketplaces(self):
        """Test creating Settings from dict with registered_marketplaces."""
        data = {
            'registered_marketplaces': [
                {
                    'name': 'custom',
                    'source': 'github:custom/repo',
                    'ref': 'v1.0.0',
                    'auto_load': True,
                }
            ]
        }
        settings = Settings.model_validate(data)

        assert len(settings.registered_marketplaces) == 1
        assert settings.registered_marketplaces[0].name == 'custom'
        assert settings.registered_marketplaces[0].ref == 'v1.0.0'

    def test_settings_allows_unique_marketplace_names(self):
        """Test that unique marketplace names are allowed."""
        settings = Settings(
            registered_marketplaces=[
                MarketplaceRegistration(
                    name='first',
                    source='github:owner/repo1',
                ),
                MarketplaceRegistration(
                    name='second',
                    source='github:owner/repo2',
                ),
            ]
        )
        assert len(settings.registered_marketplaces) == 2

    def test_settings_with_mixed_marketplaces(self):
        """Test Settings with marketplace containing all optional fields."""
        settings = Settings(
            registered_marketplaces=[
                MarketplaceRegistration(
                    name='full-featured',
                    source='github:owner/repo',
                    ref='v1.0.0',
                    repo_path='marketplaces/internal',
                    auto_load=True,
                ),
                MarketplaceRegistration(
                    name='minimal',
                    source='github:owner/minimal',
                ),
                MarketplaceRegistration(
                    name='auto-load-only',
                    source='github:owner/auto',
                    auto_load=True,
                ),
            ]
        )
        assert len(settings.registered_marketplaces) == 3

        # Verify full-featured marketplace
        full = settings.registered_marketplaces[0]
        assert full.name == 'full-featured'
        assert full.ref == 'v1.0.0'
        assert full.repo_path == 'marketplaces/internal'
        assert full.auto_load

        # Verify minimal marketplace has None for optional fields and False for auto_load
        minimal = settings.registered_marketplaces[1]
        assert minimal.ref is None
        assert minimal.repo_path is None
        assert minimal.auto_load is False  # Defaults to False, not None

    def test_settings_registered_marketplaces_serialization_roundtrip(self):
        """Test that marketplace data survives serialization roundtrip."""
        original = Settings(
            registered_marketplaces=[
                MarketplaceRegistration(
                    name='test',
                    source='github:owner/repo',
                    ref='main',
                    auto_load=True,
                ),
            ]
        )

        # Serialize to dict
        data = original.model_dump()
        assert 'registered_marketplaces' in data
        assert len(data['registered_marketplaces']) == 1

        # Deserialize back
        restored = Settings.model_validate(data)
        assert len(restored.registered_marketplaces) == 1
        assert restored.registered_marketplaces[0].name == 'test'
        assert restored.registered_marketplaces[0].ref == 'main'
        assert restored.registered_marketplaces[0].auto_load


class TestMarketplaceRegistrationValidationEdgeCases:
    """Edge case tests for MarketplaceRegistration validation."""

    def test_name_with_hyphens_and_underscores(self):
        """Test that names with hyphens and underscores are valid."""
        reg = MarketplaceRegistration(
            name='my_marketplace-name', source='github:owner/repo'
        )
        assert reg.name == 'my_marketplace-name'

    def test_name_with_numbers(self):
        """Test that names with numbers are valid."""
        reg = MarketplaceRegistration(name='marketplace123', source='github:owner/repo')
        assert reg.name == 'marketplace123'

    def test_name_strips_surrounding_whitespace(self):
        """Test that surrounding whitespace is stripped from name."""
        reg = MarketplaceRegistration(name='  trimmed  ', source='github:owner/repo')
        assert reg.name == 'trimmed'

    def test_source_strips_surrounding_whitespace(self):
        """Test that surrounding whitespace is stripped from source."""
        reg = MarketplaceRegistration(name='test', source='  github:owner/repo  ')
        assert reg.source == 'github:owner/repo'

    def test_local_path_with_subdirectories(self):
        """Test that local paths with subdirectories are valid."""
        reg = MarketplaceRegistration(
            name='nested',
            source='marketplaces/team/internal',
        )
        assert reg.source == 'marketplaces/team/internal'

    def test_source_validation_rejects_leading_dot(self):
        """Test that sources starting with dot are rejected (hidden files)."""
        with pytest.raises(ValidationError, match='source must be'):
            MarketplaceRegistration(name='test', source='.hidden/repo')

    def test_source_validation_rejects_double_dot_in_middle(self):
        """Test that sources with .. in middle are rejected."""
        with pytest.raises(ValidationError, match="cannot contain '..'"):
            MarketplaceRegistration(name='test', source='path/../escape')

    def test_source_validation_rejects_git_protocol(self):
        """Test that git:// protocol URLs are valid."""
        reg = MarketplaceRegistration(
            name='test',
            source='git://github.com/owner/repo',
        )
        assert reg.source == 'git://github.com/owner/repo'

    def test_repo_path_validation_rejects_double_dot(self):
        """Test that repo_path with .. is rejected."""
        with pytest.raises(ValidationError, match="cannot contain '..'"):
            MarketplaceRegistration(
                name='test',
                source='github:owner/repo',
                repo_path='path/../escape',
            )


class TestSettingsDuplicateMarketplaceNames:
    """Settings construction and marketplace name handling.

    Uniqueness is enforced on the write paths (not on model construction), so a
    legacy row with duplicate names can never lock a user out of settings.
    """

    def test_settings_tolerates_duplicate_marketplace_names(self):
        """Constructing Settings from duplicate stored names must not raise."""
        # Arrange / Act - duplicate names (e.g. legacy data) must load cleanly.
        settings = Settings(
            registered_marketplaces=[
                MarketplaceRegistration(name='plugins', source='github:owner/repo1'),
                MarketplaceRegistration(name='plugins', source='github:owner/repo2'),
            ]
        )
        # Assert
        assert len(settings.registered_marketplaces) == 2

    def test_settings_allows_same_source_different_names(self):
        """Test that same source with different names is allowed."""
        settings = Settings(
            registered_marketplaces=[
                MarketplaceRegistration(
                    name='plugins-1',
                    source='github:owner/repo',
                ),
                MarketplaceRegistration(
                    name='plugins-2',  # Different name, same source
                    source='github:owner/repo',
                ),
            ]
        )
        assert len(settings.registered_marketplaces) == 2

    def test_settings_allows_empty_marketplaces(self):
        """Test that Settings allows empty registered_marketplaces."""
        settings = Settings(registered_marketplaces=[])
        assert settings.registered_marketplaces == []

    def test_settings_allows_none_marketplaces(self):
        """Test that Settings allows None registered_marketplaces (uses default)."""
        settings = Settings()
        assert settings.registered_marketplaces == []


def test_git_full_clone_defaults_to_false_and_updates():
    settings = Settings()

    assert settings.git_full_clone is False

    settings.update({'git_full_clone': True})

    assert settings.git_full_clone is True
    assert settings.model_dump(mode='json')['git_full_clone'] is True
