from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from storage.user_settings import UserSettings

MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / 'migrations'
    / 'versions'
    / '108_add_agent_settings_to_enterprise_settings.py'
)
spec = spec_from_file_location('migration_108', MIGRATION_PATH)
assert spec is not None and spec.loader is not None
migration_108 = module_from_spec(spec)
spec.loader.exec_module(migration_108)


def test_user_settings_are_split_into_agent_and_conversation_buckets():
    row = {
        'agent': 'CodeActAgent',
        'max_iterations': 42,
        'security_analyzer': 'llm',
        'confirmation_mode': True,
        'llm_model': 'anthropic/claude-sonnet-4-5-20250929',
        'llm_base_url': 'https://api.example.com',
        'enable_default_condenser': False,
        'condenser_max_size': 128,
        'mcp_config': {'mcpServers': {'admin': {'url': 'https://mcp.example.com'}}},
        'agent_settings': {},
        'conversation_settings': {},
    }

    agent_settings = migration_108._build_user_agent_settings(row)
    conversation_settings = migration_108._build_user_conversation_settings(row)

    assert agent_settings == {
        'schema_version': 1,
        'agent': 'CodeActAgent',
        'llm': {
            'model': 'anthropic/claude-sonnet-4-5-20250929',
            'base_url': 'https://api.example.com',
        },
        'condenser': {'enabled': False, 'max_size': 128},
        'mcp_config': {'mcpServers': {'admin': {'url': 'https://mcp.example.com'}}},
    }
    assert conversation_settings == {
        'max_iterations': 42,
        'confirmation_mode': True,
        'security_analyzer': 'llm',
    }


def test_user_settings_normalize_legacy_mcp_config():
    row = {
        'agent': 'CodeActAgent',
        'max_iterations': 42,
        'security_analyzer': 'llm',
        'confirmation_mode': True,
        'llm_model': 'anthropic/claude-sonnet-4-5-20250929',
        'llm_base_url': 'https://api.example.com',
        'enable_default_condenser': False,
        'condenser_max_size': 128,
        'mcp_config': {
            'sse_servers': [],
            'stdio_servers': [],
            'shttp_servers': [
                {'url': 'https://mcp.example.com', 'api_key': None, 'timeout': 60}
            ],
        },
        'agent_settings': {},
        'conversation_settings': {},
    }

    assert migration_108._build_user_agent_settings(row) == {
        'schema_version': 1,
        'agent': 'CodeActAgent',
        'llm': {
            'model': 'anthropic/claude-sonnet-4-5-20250929',
            'base_url': 'https://api.example.com',
        },
        'condenser': {'enabled': False, 'max_size': 128},
        'mcp_config': {
            'mcpServers': {'shttp': {'url': 'https://mcp.example.com', 'timeout': 60}}
        },
    }


def test_org_member_diffs_use_nested_llm_and_conversation_settings():
    row = {
        'max_iterations': 50,
        'llm_model': 'openhands/claude-3',
        'llm_base_url': 'https://proxy.example.com',
        'mcp_config': {'mcpServers': {'admin': {'url': 'https://mcp.example.com'}}},
        'agent_settings_diff': {},
        'conversation_settings_diff': {},
    }

    agent_settings_diff = migration_108._build_org_member_agent_settings_diff(row)
    conversation_settings_diff = (
        migration_108._build_org_member_conversation_settings_diff(row)
    )

    assert agent_settings_diff == {
        'schema_version': 1,
        'llm': {
            'model': 'openhands/claude-3',
            'base_url': 'https://proxy.example.com',
        },
        'mcp_config': {'mcpServers': {'admin': {'url': 'https://mcp.example.com'}}},
    }
    assert conversation_settings_diff == {'max_iterations': 50}


def test_org_member_diffs_normalize_legacy_mcp_config():
    row = {
        'max_iterations': 50,
        'llm_model': 'openhands/claude-3',
        'llm_base_url': 'https://proxy.example.com',
        'mcp_config': {
            'sse_servers': [],
            'stdio_servers': [],
            'shttp_servers': [
                {'url': 'https://mcp.deepwiki.com/mcp', 'api_key': None, 'timeout': 60}
            ],
        },
        'agent_settings_diff': {},
        'conversation_settings_diff': {},
    }

    assert migration_108._build_org_member_agent_settings_diff(row) == {
        'schema_version': 1,
        'llm': {
            'model': 'openhands/claude-3',
            'base_url': 'https://proxy.example.com',
        },
        'mcp_config': {
            'mcpServers': {
                'shttp': {'url': 'https://mcp.deepwiki.com/mcp', 'timeout': 60}
            }
        },
    }


def test_org_settings_are_split_into_agent_and_conversation_buckets():
    row = {
        'agent': 'CodeActAgent',
        'default_max_iterations': 99,
        'security_analyzer': 'auto',
        'confirmation_mode': False,
        'default_llm_model': 'anthropic/claude-3-7-sonnet',
        'default_llm_base_url': 'https://api.example.com',
        'enable_default_condenser': True,
        'condenser_max_size': 256,
        'mcp_config': {'mcpServers': {'org': {'url': 'https://org-mcp.example.com'}}},
        'agent_settings': {},
        'conversation_settings': {},
    }

    agent_settings = migration_108._build_org_agent_settings(row)
    conversation_settings = migration_108._build_org_conversation_settings(row)

    assert agent_settings == {
        'schema_version': 1,
        'agent': 'CodeActAgent',
        'llm': {
            'model': 'anthropic/claude-3-7-sonnet',
            'base_url': 'https://api.example.com',
        },
        'condenser': {'enabled': True, 'max_size': 256},
        'mcp_config': {'mcpServers': {'org': {'url': 'https://org-mcp.example.com'}}},
    }
    assert conversation_settings == {
        'max_iterations': 99,
        'confirmation_mode': False,
        'security_analyzer': 'auto',
    }


def test_downgrade_extracts_legacy_values_from_nested_settings():
    row = {
        'agent_settings': {
            'schema_version': 1,
            'agent': 'CodeActAgent',
            'llm': {
                'model': 'anthropic/claude-sonnet-4-5-20250929',
                'base_url': 'https://api.example.com',
            },
            'condenser': {'enabled': False, 'max_size': 128},
        },
        'conversation_settings': {
            'max_iterations': 42,
            'confirmation_mode': True,
            'security_analyzer': 'llm',
        },
    }

    assert migration_108._legacy_user_settings_values(row) == {
        'agent': 'CodeActAgent',
        'max_iterations': 42,
        'security_analyzer': 'llm',
        'confirmation_mode': True,
        'llm_model': 'anthropic/claude-sonnet-4-5-20250929',
        'llm_base_url': 'https://api.example.com',
        'enable_default_condenser': False,
        'condenser_max_size': 128,
    }


def test_downgrade_restores_legacy_mcp_config_from_sdk_settings():
    row = {
        'agent_settings_diff': {
            'schema_version': 1,
            'mcp_config': {
                'mcpServers': {
                    'sse': {'url': 'https://mcp.example.com', 'transport': 'sse'},
                    'shttp': {
                        'url': 'https://mcp.deepwiki.com/mcp',
                        'timeout': 60,
                    },
                    'deepwiki-stdio': {
                        'command': 'npx',
                        'args': ['-y', 'deepwiki-mcp'],
                        'env': {'A': 'B'},
                    },
                }
            },
        },
        'conversation_settings_diff': {},
    }

    assert migration_108._legacy_org_member_values(row)['mcp_config'] == {
        'sse_servers': [{'url': 'https://mcp.example.com'}],
        'stdio_servers': [
            {
                'name': 'deepwiki-stdio',
                'command': 'npx',
                'args': ['-y', 'deepwiki-mcp'],
                'env': {'A': 'B'},
            }
        ],
        'shttp_servers': [{'url': 'https://mcp.deepwiki.com/mcp', 'timeout': 60}],
    }


def test_migrated_payload_loads_via_user_settings_to_settings():
    row = {
        'agent': 'CodeActAgent',
        'max_iterations': 42,
        'security_analyzer': 'llm',
        'confirmation_mode': True,
        'llm_model': 'anthropic/claude-sonnet-4-5-20250929',
        'llm_base_url': 'https://api.example.com',
        'enable_default_condenser': False,
        'condenser_max_size': 128,
        'mcp_config': {'mcpServers': {'admin': {'url': 'https://mcp.example.com'}}},
        'agent_settings': {},
        'conversation_settings': {},
    }

    user_settings = UserSettings(
        agent_settings=migration_108._build_user_agent_settings(row),
        conversation_settings=migration_108._build_user_conversation_settings(row),
    )

    settings = user_settings.to_settings()

    assert settings.agent_settings.agent == 'CodeActAgent'
    assert settings.agent_settings.llm.model == 'anthropic/claude-sonnet-4-5-20250929'
    assert settings.agent_settings.llm.base_url == 'https://api.example.com'
    assert settings.agent_settings.condenser.enabled is False
    assert settings.agent_settings.condenser.max_size == 128
    assert settings.agent_settings.mcp_config is not None
    assert (
        settings.agent_settings.mcp_config.mcpServers['admin'].url
        == 'https://mcp.example.com'
    )
    assert settings.conversation_settings.max_iterations == 42
    assert settings.conversation_settings.confirmation_mode is True
    assert settings.conversation_settings.security_analyzer == 'llm'
