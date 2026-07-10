import importlib
import json

import pytest
from pydantic import SecretStr

from openhands.app_server.mcp.mcp_config_adapter import mcp_config_server_map
from openhands.app_server.settings.llm_profiles import LLMProfiles
from openhands.app_server.settings.settings_models import Settings


def _migration_module():
    return importlib.import_module(
        'migrations.versions.137_rewrite_settings_secret_storage'
    )


def test_secret_aware_json_reads_legacy_encrypted_blob_and_writes_leaf_encrypted_json():
    from storage.encrypt_utils import (
        SecretAwareJSON,
        decrypt_value,
        encrypt_value,
        get_settings_cipher_context,
    )

    payload = {
        'profiles': {
            'legacy': {
                'model': 'anthropic/claude-sonnet-4-5-20250929',
                'api_key': 'legacy-profile-secret',
            }
        },
        'active': 'legacy',
    }
    column = SecretAwareJSON()

    legacy_blob = encrypt_value(json.dumps(payload))
    loaded = column.process_result_value(legacy_blob, dialect=None)
    assert loaded == payload
    assert (
        LLMProfiles.model_validate(loaded).require('legacy').api_key.get_secret_value()
        == 'legacy-profile-secret'
    )

    stored = column.process_bind_param(payload, dialect=None)
    assert stored is not None
    assert 'legacy-profile-secret' not in stored
    stored_json = json.loads(stored)
    assert (
        stored_json['profiles']['legacy']['model']
        == payload['profiles']['legacy']['model']
    )
    assert stored_json['profiles']['legacy']['api_key'] != 'legacy-profile-secret'
    assert (
        LLMProfiles.model_validate(
            column.process_result_value(stored, dialect=None),
            context=get_settings_cipher_context(),
        )
        .require('legacy')
        .api_key.get_secret_value()
        == 'legacy-profile-secret'
    )

    downgraded_blob = encrypt_value(json.dumps(payload))
    assert json.loads(decrypt_value(downgraded_blob)) == payload


def test_migration_rewrites_legacy_profiles_and_agent_settings_secrets():
    from storage.encrypt_utils import (
        encrypt_value,
        get_settings_cipher,
        get_settings_cipher_context,
    )

    migration = _migration_module()
    legacy_profiles = {
        'profiles': {
            'bedrock': {
                'model': 'bedrock/converse/us.anthropic.claude-sonnet-4-5-20250929-v1:0',
                'api_key': 'profile-api-key',
                'aws_access_key_id': 'profile-aws-access',
                'aws_secret_access_key': 'profile-aws-secret',
                'aws_session_token': 'profile-aws-session',
            }
        },
        'active': 'bedrock',
    }

    migrated_profiles = json.loads(
        migration._profiles_to_json(encrypt_value(json.dumps(legacy_profiles)))
    )
    profile_serialized = json.dumps(migrated_profiles)
    for secret in (
        'profile-api-key',
        'profile-aws-access',
        'profile-aws-secret',
        'profile-aws-session',
    ):
        assert secret not in profile_serialized
    profile = LLMProfiles.model_validate(
        migrated_profiles, context=get_settings_cipher_context()
    ).require('bedrock')
    assert profile.api_key.get_secret_value() == 'profile-api-key'
    assert profile.aws_access_key_id.get_secret_value() == 'profile-aws-access'
    assert profile.aws_secret_access_key.get_secret_value() == 'profile-aws-secret'
    assert profile.aws_session_token.get_secret_value() == 'profile-aws-session'

    legacy_agent_settings = {
        'llm': {
            'model': legacy_profiles['profiles']['bedrock']['model'],
            'aws_access_key_id': 'agent-aws-access',
            'aws_secret_access_key': 'agent-aws-secret',
            'aws_session_token': 'agent-aws-session',
        },
        'verification': {'critic_api_key': 'critic-secret'},
        'agent_context': {'secrets': {'AGENT_TOKEN': 'agent-context-secret'}},
        'mcp_config': {
            'mcpServers': {
                'http': {
                    'url': 'https://mcp.example.com',
                    'transport': 'sse',
                    'headers': {'Authorization': 'Bearer header-secret'},
                },
                'stdio': {
                    'command': 'node',
                    'args': ['server.js'],
                    'env': {'MCP_ENV_TOKEN': 'env-secret'},
                },
            }
        },
    }
    encrypted_agent_settings = migration._encrypt_agent_settings(legacy_agent_settings)
    assert encrypted_agent_settings['schema_version'] == 4
    assert encrypted_agent_settings['agent_kind'] == 'openhands'
    assert 'mcpServers' not in encrypted_agent_settings['mcp_config']
    agent_serialized = json.dumps(encrypted_agent_settings)
    for secret in (
        'agent-aws-access',
        'agent-aws-secret',
        'agent-aws-session',
        'critic-secret',
        'agent-context-secret',
        'header-secret',
        'env-secret',
    ):
        assert secret not in agent_serialized

    loaded = Settings.model_validate(
        {'agent_settings': encrypted_agent_settings},
        context=get_settings_cipher_context(),
    )
    assert loaded.agent_settings.llm.aws_access_key_id.get_secret_value() == (
        'agent-aws-access'
    )
    assert loaded.agent_settings.llm.aws_secret_access_key.get_secret_value() == (
        'agent-aws-secret'
    )
    assert loaded.agent_settings.llm.aws_session_token.get_secret_value() == (
        'agent-aws-session'
    )
    assert loaded.agent_settings.verification.critic_api_key.get_secret_value() == (
        'critic-secret'
    )
    assert loaded.agent_settings.agent_context.secrets == {
        'AGENT_TOKEN': 'agent-context-secret'
    }
    servers = mcp_config_server_map(loaded.agent_settings.mcp_config)
    http_auth = servers['http'].auth.headers['Authorization']
    assert http_auth.get_secret_value() == 'Bearer header-secret'
    assert servers['stdio'].env['MCP_ENV_TOKEN'].get_secret_value() == 'env-secret'

    downgraded = migration._decrypt_agent_settings(encrypted_agent_settings)
    assert downgraded['schema_version'] == 4
    assert downgraded['agent_kind'] == 'openhands'
    assert downgraded['agent_context']['secrets']['AGENT_TOKEN'] == (
        'agent-context-secret'
    )
    assert downgraded['mcp_config']['stdio']['env']['MCP_ENV_TOKEN'] == 'env-secret'

    encrypted_once = get_settings_cipher().encrypt(SecretStr('already-encrypted'))
    assert migration._encrypt_secret_value(encrypted_once) == encrypted_once

    native_agent_settings = {
        'mcp_config': {
            'bearer': {
                'url': 'https://bearer.example.com/mcp',
                'transport': 'sse',
                'auth': {'strategy': 'bearer', 'value': 'native-bearer-secret'},
            },
            'basic': {
                'url': 'https://basic.example.com/mcp',
                'transport': 'sse',
                'auth': {
                    'strategy': 'basic',
                    'username': 'user',
                    'password': 'native-basic-secret',
                },
            },
            'header': {
                'url': 'https://header.example.com/mcp',
                'transport': 'sse',
                'auth': {
                    'strategy': 'header',
                    'headers': {'X-API-Key': 'native-header-secret'},
                },
            },
            'oauth': {
                'url': 'https://oauth.example.com/mcp',
                'transport': 'sse',
                'auth': {
                    'strategy': 'oauth2',
                    'authentication': {
                        'type': 'oauth',
                        'client_secret': 'native-client-secret',
                    },
                    'state': {
                        'tokens': {
                            'access_token': 'native-access-token',
                            'refresh_token': 'native-refresh-token',
                        },
                        'client_info': {'client_secret': 'native-state-client-secret'},
                    },
                },
            },
            'legacy': {
                'url': 'https://legacy.example.com/mcp',
                'transport': 'sse',
                'auth': 'legacy-bearer-secret',
                'api_key': 'legacy-api-key-secret',
                'oauth_credentials': {
                    'mcp-oauth-token': {
                        'entry': {
                            'value': {
                                'access_token': 'legacy-access-token',
                                'refresh_token': 'legacy-refresh-token',
                            }
                        }
                    }
                },
            },
        }
    }
    encrypted_native = migration._encrypt_agent_settings(native_agent_settings)
    native_serialized = json.dumps(encrypted_native)
    for secret in (
        'native-bearer-secret',
        'native-basic-secret',
        'native-header-secret',
        'native-client-secret',
        'native-access-token',
        'native-refresh-token',
        'native-state-client-secret',
        'legacy-bearer-secret',
        'legacy-api-key-secret',
        'legacy-access-token',
        'legacy-refresh-token',
    ):
        assert secret not in native_serialized

    loaded_native = Settings.model_validate(
        {'agent_settings': encrypted_native},
        context=get_settings_cipher_context(),
    )
    native_servers = mcp_config_server_map(loaded_native.agent_settings.mcp_config)
    assert native_servers['bearer'].auth.value.get_secret_value() == (
        'native-bearer-secret'
    )
    assert native_servers['basic'].auth.password.get_secret_value() == (
        'native-basic-secret'
    )
    assert native_servers['header'].auth.headers['X-API-Key'].get_secret_value() == (
        'native-header-secret'
    )
    oauth = native_servers['oauth'].auth
    assert oauth.authentication.client_secret.get_secret_value() == (
        'native-client-secret'
    )
    assert oauth.state.tokens.access_token.get_secret_value() == 'native-access-token'
    assert oauth.state.tokens.refresh_token.get_secret_value() == 'native-refresh-token'
    assert oauth.state.client_info.client_secret.get_secret_value() == (
        'native-state-client-secret'
    )

    assert encrypted_native['schema_version'] == 4
    assert encrypted_native['agent_kind'] == 'openhands'

    downgraded_native = migration._decrypt_agent_settings(encrypted_native)
    legacy_auth = downgraded_native['mcp_config']['legacy']['auth']
    assert legacy_auth == {'strategy': 'bearer', 'value': 'legacy-bearer-secret'}


def test_fernet_looking_plaintext_is_encrypted_and_leniently_readable():
    from storage.encrypt_utils import get_settings_cipher_context

    migration = _migration_module()
    plaintext = 'gAAAA-this-is-not-valid-fernet-ciphertext'

    encrypted = migration._encrypt_secret_value(plaintext)
    assert encrypted != plaintext
    assert migration._decrypt_secret_value(encrypted) == plaintext

    cipher = get_settings_cipher_context()['cipher']
    assert cipher.decrypt(plaintext).get_secret_value() == plaintext
