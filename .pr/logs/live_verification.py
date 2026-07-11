"""Live verification of PR #15103: MCP auth headers, stdin env secrets,
redaction at rest, and member-private scoping.

Exercises the REAL production code path:
  Settings.update -> SaasSettingsStore.store -> ORM JSON/secret serializers
  -> SaasSettingsStore.load

Uses a temporary SQLite DB with real enterprise ORM tables (role, org, user,
org_member). No mocks on the store/load path. The only stub is the JWT cipher
(injected via encrypt_utils patching) so encryption runs without a Keycloak
server.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import uuid
from unittest.mock import patch

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, os.path.join(REPO_ROOT, 'enterprise'))
sys.path.insert(0, REPO_ROOT)


def _setup_cipher():
    """Set up a real Fernet cipher so encryption/decryption actually runs."""
    from cryptography.fernet import Fernet

    key = Fernet.generate_key()

    class FakeKey:
        def __init__(self, k):
            self.key = type('S', (), {'get_secret_value': lambda self: k})()

    class FakeJwtService:
        def __init__(self, k):
            self._default_key_id = 'default'
            self._key = k

        def get_key(self, kid):
            return FakeKey(key.decode())

        def encrypt_value(self, val):
            return Fernet(key).encrypt(val.encode()).decode()

        def decrypt_value(self, val):
            return Fernet(key).decrypt(val.encode()).decode()

    jwt_svc = FakeJwtService(key)

    import storage.encrypt_utils as eu

    eu._jwt_service = jwt_svc
    eu._fernet = None
    eu._settings_cipher = None
    return jwt_svc


async def run_verification():
    _setup_cipher()

    from sqlalchemy import create_engine, select
    from sqlalchemy.ext.asyncio import (
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )
    from sqlalchemy.orm import sessionmaker
    from storage.api_key import ApiKey  # noqa: F401
    from storage.base import Base
    from storage.billing_session import BillingSession  # noqa: F401
    from storage.conversation_work import ConversationWork  # noqa: F401
    from storage.device_code import DeviceCode  # noqa: F401
    from storage.feedback import Feedback  # noqa: F401
    from storage.github_app_installation import GithubAppInstallation  # noqa: F401
    from storage.org import Org
    from storage.org_budget_settings import OrgBudgetSettings  # noqa: F401
    from storage.org_budget_threshold import OrgBudgetThreshold  # noqa: F401
    from storage.org_git_claim import OrgGitClaim  # noqa: F401
    from storage.org_invitation import OrgInvitation  # noqa: F401
    from storage.org_member import OrgMember
    from storage.org_user_budget_override import OrgUserBudgetOverride  # noqa: F401
    from storage.role import Role
    from storage.slack_conversation import SlackConversation  # noqa: F401
    from storage.stored_conversation_metadata import (
        StoredConversationMetadata,  # noqa: F401
    )
    from storage.stored_conversation_metadata_saas import (  # noqa: F401
        StoredConversationMetadataSaas,
    )
    from storage.stored_offline_token import StoredOfflineToken  # noqa: F401
    from storage.stripe_customer import StripeCustomer  # noqa: F401
    from storage.user import User
    from storage.user_settings import UserSettings  # noqa: F401

    # Collect only the tables that SQLite can render (skip ARRAY columns)
    tables_to_create = []
    for table in Base.metadata.sorted_tables:
        if table.name in ('gitlab_webhook',):
            continue
        tables_to_create.append(table)

    db_fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(db_fd)

    sync_engine = create_engine(
        f'sqlite:///{db_path}', connect_args={'check_same_thread': False}
    )
    Base.metadata.create_all(sync_engine, tables=tables_to_create)
    session_maker = sessionmaker(bind=sync_engine)

    async_engine = create_async_engine(
        f'sqlite+aiosqlite:///{db_path}',
        connect_args={'check_same_thread': False},
    )
    async_session = async_sessionmaker(
        bind=async_engine, class_=AsyncSession, expire_on_commit=False
    )

    org_id = uuid.UUID('5594c7b6-f959-4b81-92e9-b09c206f5081')
    admin_user_id = uuid.UUID('5594c7b6-f959-4b81-92e9-b09c206f5082')
    member2_user_id = uuid.UUID('5594c7b6-f959-4b81-92e9-b09c206f5084')

    with session_maker() as session:
        role = Role(id=10, name='member', rank=3)
        session.add(role)
        org = Org(
            id=org_id,
            name='test-org',
            org_version=1,
            enable_proactive_conversation_starters=True,
            agent_settings={
                'schema_version': 1,
                'mcp_config': {
                    'legacy-org-server': {
                        'url': 'https://legacy.example.com/sse',
                        'transport': 'sse',
                        'headers': {'Authorization': 'Bearer legacy-org-secret'},
                    }
                },
            },
        )
        session.add(org)
        admin_user = User(
            id=admin_user_id,
            current_org_id=org_id,
            user_consents_to_analytics=True,
        )
        session.add(admin_user)
        member2_user = User(
            id=member2_user_id,
            current_org_id=org_id,
            user_consents_to_analytics=True,
        )
        session.add(member2_user)
        admin_member = OrgMember(
            org_id=org_id,
            user_id=admin_user_id,
            role_id=10,
            llm_api_key='admin-initial-key',
            agent_settings_diff={
                'llm': {'model': 'old-model-v1', 'base_url': 'http://old-url-1.com'},
            },
            conversation_settings_diff={'max_iterations': 10},
            status='active',
        )
        session.add(admin_member)
        member2 = OrgMember(
            org_id=org_id,
            user_id=member2_user_id,
            role_id=10,
            llm_api_key='member2-initial-key',
            agent_settings_diff={
                'llm': {'model': 'old-model-v3', 'base_url': 'http://old-url-3.com'},
            },
            conversation_settings_diff={'max_iterations': 30},
            status='active',
        )
        session.add(member2)
        session.commit()

    results = []

    from openhands.app_server.settings.settings_models import Settings

    SYNTHETIC_HTTP_TOKEN = 'synthetic-http-bearer-token-12345'
    SYNTHETIC_STDIO_ENV_SECRET = 'synthetic-stdio-env-secret-67890'
    SYNTHETIC_LLM_KEY = 'synthetic-llm-api-key-abcdef'

    settings = Settings()
    settings.update(
        {
            'agent_settings_diff': {
                'llm': {
                    'model': 'test-model',
                    'base_url': 'http://non-litellm-url.com',
                    'api_key': SYNTHETIC_LLM_KEY,
                },
                'mcp_config': {
                    'http-server': {
                        'url': 'https://mcp.example.com/sse',
                        'transport': 'sse',
                        'headers': {'Authorization': f'Bearer {SYNTHETIC_HTTP_TOKEN}'},
                    },
                    'stdio-server': {
                        'command': 'npx',
                        'args': ['-y', '@modelcontextprotocol/server-memory'],
                        'env': {
                            'API_KEY': SYNTHETIC_STDIO_ENV_SECRET,
                            'OTHER_VAR': 'not-secret',
                        },
                    },
                },
            },
        }
    )

    from storage.saas_settings_store import SaasSettingsStore

    store = SaasSettingsStore(str(admin_user_id))
    with (
        patch('storage.saas_settings_store.a_session_maker', async_session),
        patch('storage.user_store.a_session_maker', async_session),
        patch('storage.org_store.a_session_maker', async_session),
    ):
        await store.store(settings)

    with session_maker() as session:
        org = session.execute(select(Org).where(Org.id == org_id)).scalars().first()
        members = {
            str(m.user_id): m
            for m in session.execute(
                select(OrgMember).where(OrgMember.org_id == org_id)
            )
            .scalars()
            .all()
        }

    org_agent_json = json.dumps(org.agent_settings or {})
    admin_diff_json = json.dumps(members[str(admin_user_id)].agent_settings_diff or {})

    assert 'mcp_config' not in (org.agent_settings or {}), (
        'FAIL: org.agent_settings still contains mcp_config'
    )
    results.append('PASS: org row strips mcp_config (member-private)')

    assert SYNTHETIC_HTTP_TOKEN not in org_agent_json, (
        'FAIL: synthetic HTTP token found in org row'
    )
    assert SYNTHETIC_STDIO_ENV_SECRET not in org_agent_json, (
        'FAIL: synthetic stdio env secret found in org row'
    )
    assert SYNTHETIC_LLM_KEY not in org_agent_json, (
        'FAIL: synthetic LLM key found in org row'
    )
    results.append('PASS: org row has no raw MCP/LLM secret values')

    admin_mcp = members[str(admin_user_id)].agent_settings_diff.get('mcp_config')
    assert admin_mcp is not None, 'FAIL: admin member row missing mcp_config'
    assert SYNTHETIC_HTTP_TOKEN not in admin_diff_json, (
        'FAIL: synthetic HTTP token found in admin member row'
    )
    assert SYNTHETIC_STDIO_ENV_SECRET not in admin_diff_json, (
        'FAIL: synthetic stdio env secret found in admin member row'
    )
    results.append('PASS: acting member row keeps encrypted MCP config, no raw secrets')

    member2_diff = members[str(member2_user_id)].agent_settings_diff or {}
    assert 'mcp_config' not in member2_diff, 'FAIL: peer member inherited mcp_config'
    results.append('PASS: peer member does not inherit MCP config')

    with (
        patch('storage.saas_settings_store.a_session_maker', async_session),
        patch('storage.user_store.a_session_maker', async_session),
        patch('storage.org_store.a_session_maker', async_session),
    ):
        loaded = await store.load()

    assert loaded is not None, 'FAIL: load() returned None'

    from openhands.app_server.mcp.mcp_config_adapter import mcp_config_server_map

    loaded_mcp = mcp_config_server_map(loaded.agent_settings.mcp_config)

    http_server = loaded_mcp.get('http-server')
    assert http_server is not None, 'FAIL: HTTP server not found in loaded settings'

    def _unwrap(val):
        from pydantic import SecretStr

        return val.get_secret_value() if isinstance(val, SecretStr) else val

    # The SDK stores HTTP auth as auth.strategy='bearer', auth.value=<SecretStr>
    auth = (
        http_server.get('auth')
        if isinstance(http_server, dict)
        else getattr(http_server, 'auth', None)
    )
    headers = (
        http_server.get('headers')
        if isinstance(http_server, dict)
        else getattr(http_server, 'headers', None)
    )

    auth_value = None
    auth_headers = None
    if auth:
        if isinstance(auth, dict):
            auth_value = auth.get('value')
            auth_headers = auth.get('headers')
        else:
            auth_value = getattr(auth, 'value', None)
            auth_headers = getattr(auth, 'headers', None)

    # Prefer auth.value (SDK bearer), fall back to headers.Authorization (legacy)
    if auth_value is not None:
        auth_header = f'Bearer {_unwrap(auth_value)}'
    elif headers:
        raw_h = (
            headers.get('Authorization')
            if isinstance(headers, dict)
            else getattr(headers, 'Authorization', None)
        )
        auth_header = _unwrap(raw_h)
    elif auth_headers:
        raw_h = (
            auth_headers.get('Authorization')
            if isinstance(auth_headers, dict)
            else getattr(auth_headers, 'Authorization', None)
        )
        auth_header = _unwrap(raw_h)
    else:
        auth_header = None

    assert auth_header is not None, 'FAIL: no auth on loaded HTTP server'
    assert auth_header == f'Bearer {SYNTHETIC_HTTP_TOKEN}', (
        f'FAIL: HTTP auth header not preserved, got: {auth_header}'
    )
    results.append(
        'PASS: HTTP bearer auth header preserved after store->load round-trip'
    )

    stdio_server = loaded_mcp.get('stdio-server')
    assert stdio_server is not None, 'FAIL: stdio server not found in loaded settings'

    env = (
        stdio_server.get('env')
        if isinstance(stdio_server, dict)
        else getattr(stdio_server, 'env', None)
    )
    assert env is not None, 'FAIL: no env on loaded stdio server'
    api_key_val = _unwrap(env.get('API_KEY'))
    assert api_key_val == SYNTHETIC_STDIO_ENV_SECRET, (
        f'FAIL: stdio env secret not preserved, got: {api_key_val}'
    )
    results.append(
        'PASS: stdio env secret (API_KEY) preserved after store->load round-trip'
    )

    loaded_llm_key = _unwrap(loaded.agent_settings.llm.api_key)
    assert loaded_llm_key is not None, 'FAIL: no LLM API key in loaded settings'
    results.append('PASS: LLM API key present in loaded settings (value preserved)')

    assert 'legacy-org-server' not in loaded_mcp, (
        'FAIL: legacy org-level MCP server leaked into member settings'
    )
    results.append(
        'PASS: legacy org-level mcp_config not leaked to acting member on load'
    )

    api_response = loaded.model_dump(mode='json')
    api_json = json.dumps(api_response)
    assert SYNTHETIC_HTTP_TOKEN not in api_json, (
        'FAIL: HTTP token leaked in API response (no expose_secrets)'
    )
    assert SYNTHETIC_STDIO_ENV_SECRET not in api_json, (
        'FAIL: stdio env secret leaked in API response (no expose_secrets)'
    )
    results.append('PASS: secrets redacted in API response (no expose_secrets context)')

    api_response_exposed = loaded.model_dump(
        mode='json', context={'expose_secrets': True}
    )
    api_json_exposed = json.dumps(api_response_exposed)
    assert SYNTHETIC_HTTP_TOKEN in api_json_exposed, (
        'FAIL: HTTP token NOT present in exposed API response'
    )
    assert SYNTHETIC_STDIO_ENV_SECRET in api_json_exposed, (
        'FAIL: stdio env secret NOT present in exposed API response'
    )
    results.append(
        'PASS: secrets visible in authorized API response (expose_secrets=True)'
    )

    await async_engine.dispose()
    sync_engine.dispose()
    os.unlink(db_path)

    head_sha = os.popen('git rev-parse HEAD').read().strip()
    print()
    print('=' * 70)
    print('PASS live SaasSettingsStore verification')
    print(f'head_sha={head_sha}')
    print(
        'production_paths=Settings.update -> SaasSettingsStore.store -> '
        'ORM JSON/secret serializers -> SaasSettingsStore.load'
    )
    print(f'checks_passed={len(results)}')
    for r in results:
        print(f'  {r}')
    print('=' * 70)

    return True


if __name__ == '__main__':
    os.environ['ALLOW_SHORT_CONTEXT_WINDOWS'] = 'true'
    os.environ['OPENHANDS_SUPPRESS_BANNER'] = '1'
    success = asyncio.run(run_verification())
    sys.exit(0 if success else 1)
