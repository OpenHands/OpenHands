import importlib
import json
import sqlite3
import uuid

import pytest
from pydantic import SecretStr

from openhands.app_server.mcp.mcp_config_adapter import mcp_config_server_map
from openhands.app_server.settings.llm_profiles import LLMProfiles
from openhands.app_server.settings.settings_models import Settings


def _migration_module():
    return importlib.import_module(
        'migrations.versions.137_rewrite_settings_secret_storage'
    )


def test_migration_agent_settings_validation_runs_sdk_migrations():
    migration = _migration_module()

    migration._validate_agent_settings_payload(
        {
            'schema_version': 1,
            'agent_kind': 'llm',
            'llm': {'model': 'openhands/claude-sonnet-4-5'},
            'mcp_config': {
                'mcpServers': {
                    'secure': {
                        'url': 'https://mcp.example.com/sse',
                        'transport': 'sse',
                        'headers': {'Authorization': 'Bearer validation-token'},
                    }
                }
            },
            'verification': {
                'confirmation_mode': True,
                'security_analyzer': 'legacy',
            },
        },
        'org_member',
        'agent_settings_diff',
        {'org_id': 'org-1', 'user_id': 'user-1'},
    )


def test_migration_agent_settings_validation_fails_fast_with_row_location():
    migration = _migration_module()

    with pytest.raises(ValueError, match='org_member.agent_settings_diff') as exc_info:
        migration._validate_agent_settings_payload(
            {
                'agent_kind': 'openhands',
                'mcp_config': {
                    'broken': {
                        'transport': 'sse',
                    }
                },
            },
            'org_member',
            'agent_settings_diff',
            {'org_id': 'org-1', 'user_id': 'user-1'},
        )

    assert 'org_id=org-1' in str(exc_info.value)
    assert 'user_id=user-1' in str(exc_info.value)


def test_migration_saves_current_schema_without_expanding_empty_member_diff():
    migration = _migration_module()

    assert migration._encrypt_agent_settings_diff({}) == {'schema_version': 4}
    assert migration._decrypt_agent_settings_diff({}) == {'schema_version': 4}


def test_migration_saves_current_schema_for_sparse_member_diff():
    migration = _migration_module()

    migrated = migration._encrypt_agent_settings_diff(
        {
            'schema_version': 1,
            'mcp_config': {
                'mcpServers': {
                    'secure': {
                        'url': 'https://mcp.example.com/sse',
                        'transport': 'sse',
                        'headers': {'Authorization': 'Bearer sparse-token'},
                    }
                }
            },
        }
    )

    assert migrated['schema_version'] == 4
    assert 'agent_kind' not in migrated
    assert 'mcpServers' not in migrated['mcp_config']
    assert 'sparse-token' not in json.dumps(migrated)

    downgraded = migration._decrypt_agent_settings_diff(migrated)
    assert downgraded['schema_version'] == 4
    assert 'agent_kind' not in downgraded
    assert (
        downgraded['mcp_config']['secure']['auth']['headers']['Authorization']
        == 'Bearer sparse-token'
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


@pytest.mark.asyncio
async def test_saas_settings_store_persists_secret_aware_material_in_database(
    tmp_path, monkeypatch
):
    monkeypatch.delenv('DB_HOST', raising=False)
    monkeypatch.setenv('POSTHOG_CLIENT_KEY', 'phc_test_dummy')

    import storage.encrypt_utils as encrypt_utils
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
    from storage.org import Org
    from storage.org_member import OrgMember
    from storage.role import Role
    from storage.saas_settings_store import SaasSettingsStore
    from storage.user import User
    from storage.user_settings import UserSettings

    from openhands.app_server import config as app_config
    from openhands.app_server.config import AppServerConfig
    from openhands.app_server.config_api.config_models import AppMode
    from openhands.app_server.services.db_session_injector import DbSessionInjector
    from openhands.app_server.services.jwt_service import JwtServiceInjector
    from openhands.sdk.llm import LLM

    old_config = app_config._global_config
    old_jwt_service = encrypt_utils._jwt_service
    old_settings_cipher = encrypt_utils._settings_cipher
    old_fernet = encrypt_utils._fernet
    try:
        encrypt_utils._jwt_service = None
        encrypt_utils._settings_cipher = None
        encrypt_utils._fernet = None
        db_injector = DbSessionInjector(persistence_dir=tmp_path, host=None)
        app_config._global_config = AppServerConfig(
            persistence_dir=tmp_path,
            db_session=db_injector,
            jwt=JwtServiceInjector(persistence_dir=tmp_path),
            app_mode=AppMode.SAAS,
            lifespan=None,
        )
        async_engine = await db_injector.get_async_db_engine()
        tables = [
            Role.__table__,
            Org.__table__,
            User.__table__,
            OrgMember.__table__,
            UserSettings.__table__,
        ]
        async with async_engine.begin() as conn:
            await conn.run_sync(Org.metadata.create_all, tables=tables)
        session_maker = async_sessionmaker(
            bind=async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        user_id = str(uuid.uuid4())
        org_id = uuid.uuid4()
        sentinels = {
            'llm_api_key': 'test-llm-secret-sentinel',
            'profile_api_key': 'test-profile-secret-sentinel',
            'mcp_header': 'test-mcp-header-secret-sentinel',
            'mcp_password': 'test-mcp-password-secret-sentinel',
        }
        async with session_maker() as session:
            role = Role(name='owner', rank=1)
            org = Org(
                id=org_id,
                name=f'Test Org {org_id.hex[:8]}',
                org_version=1,
                agent_settings={},
                conversation_settings={},
                enable_proactive_conversation_starters=True,
            )
            user = User(
                id=uuid.UUID(user_id),
                current_org_id=org_id,
                user_consents_to_analytics=False,
                enable_sound_notifications=False,
                git_full_clone=False,
            )
            session.add_all([role, org, user])
            await session.flush()
            session.add(
                OrgMember(
                    org_id=org_id,
                    user_id=uuid.UUID(user_id),
                    role_id=role.id,
                    status='active',
                    llm_api_key='bootstrap-nonsecret-key',
                )
            )
            await session.commit()

        store = SaasSettingsStore(user_id=user_id)
        settings = await store.load()
        assert settings is not None
        settings.update(
            {
                'agent_settings_diff': {
                    'llm': {
                        'model': 'anthropic/claude-sonnet-4-5-20250929',
                        'api_key': sentinels['llm_api_key'],
                    },
                    'mcp_config': {
                        'header-server': {
                            'url': 'https://mcp-header.example.invalid/sse',
                            'transport': 'sse',
                            'auth': {
                                'strategy': 'header',
                                'headers': {'X-API-Key': sentinels['mcp_header']},
                            },
                        },
                        'basic-server': {
                            'url': 'https://mcp-basic.example.invalid/sse',
                            'transport': 'sse',
                            'auth': {
                                'strategy': 'basic',
                                'username': 'sentinel-user',
                                'password': sentinels['mcp_password'],
                            },
                        },
                    },
                }
            }
        )
        settings.llm_profiles.save(
            'sentinel-profile',
            LLM(
                model='anthropic/claude-sonnet-4-5-20250929',
                api_key=SecretStr(sentinels['profile_api_key']),
            ),
            include_secrets=True,
        )
        await store.store(settings)

        db_path = tmp_path / 'openhands.db'
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute(
                'select org.llm_profiles, org_member.agent_settings_diff, '
                'org_member._llm_api_key from org '
                'join org_member on org.id = org_member.org_id'
            ).fetchone()
        finally:
            conn.close()
        assert row is not None
        raw_llm_profiles, raw_member_diff, raw_member_llm_key = row
        raw_combined = json.dumps(row, default=str)
        for sentinel in sentinels.values():
            assert sentinel not in raw_combined
        assert 'https://mcp-header.example.invalid/sse' in raw_combined
        assert 'sentinel-profile' in raw_combined
        assert raw_member_llm_key

        cipher = encrypt_utils.get_settings_cipher()
        profiles = json.loads(raw_llm_profiles)
        profile_key = profiles['profiles']['sentinel-profile']['api_key']
        assert profile_key != sentinels['profile_api_key']
        assert (
            cipher.decrypt(profile_key).get_secret_value()
            == sentinels['profile_api_key']
        )

        member_diff = json.loads(raw_member_diff)
        serialized_member_diff = json.dumps(member_diff)
        assert sentinels['mcp_header'] not in serialized_member_diff
        assert sentinels['mcp_password'] not in serialized_member_diff

        loaded = await store.load()
        assert loaded is not None
        assert (
            loaded.agent_settings.llm.api_key.get_secret_value()
            == sentinels['llm_api_key']
        )
        servers = mcp_config_server_map(loaded.agent_settings.mcp_config)
        assert (
            servers['header-server'].auth.headers['X-API-Key'].get_secret_value()
            == sentinels['mcp_header']
        )
        assert (
            servers['basic-server'].auth.password.get_secret_value()
            == sentinels['mcp_password']
        )
        assert (
            loaded.llm_profiles.require('sentinel-profile').api_key.get_secret_value()
            == sentinels['profile_api_key']
        )
    finally:
        await db_injector.close()
        app_config._global_config = old_config
        encrypt_utils._jwt_service = old_jwt_service
        encrypt_utils._settings_cipher = old_settings_cipher
        encrypt_utils._fernet = old_fernet
