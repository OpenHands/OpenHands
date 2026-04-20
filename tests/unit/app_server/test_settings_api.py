import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from pydantic import SecretStr

from openhands.integrations.provider import ProviderToken, ProviderType
from openhands.integrations.service_types import UserGitInfo
from openhands.sdk.llm import LLM
from openhands.sdk.settings import (
    AgentSettings,
    ConversationSettings,
    VerificationSettings,
)
from openhands.server.app import app
from openhands.server.user_auth.user_auth import UserAuth
from openhands.storage.data_models.secrets import Secrets
from openhands.storage.data_models.settings import Settings
from openhands.storage.memory import InMemoryFileStore
from openhands.storage.secrets.secrets_store import SecretsStore
from openhands.storage.settings.file_settings_store import FileSettingsStore
from openhands.storage.settings.settings_store import SettingsStore

_EXPOSE = {'expose_secrets': True}


def _dump(settings: Settings) -> dict:
    """Dump settings to a JSON-compatible dict with secrets exposed."""
    return settings.model_dump(mode='json', context=_EXPOSE, exclude_unset=True)


class MockUserAuth(UserAuth):
    """Mock implementation of UserAuth for testing."""

    def __init__(self):
        self._settings = None
        self._settings_store = MagicMock()
        self._settings_store.load = AsyncMock(return_value=None)
        self._settings_store.store = AsyncMock()

    async def get_user_id(self) -> str | None:
        return 'test-user'

    async def get_user_email(self) -> str | None:
        return 'test-email@whatever.com'

    async def get_access_token(self) -> SecretStr | None:
        return SecretStr('test-token')

    async def get_provider_tokens(
        self,
    ) -> dict[ProviderType, ProviderToken] | None:  # noqa: E501
        return None

    async def get_user_settings_store(self) -> SettingsStore | None:
        return self._settings_store

    async def get_secrets_store(self) -> SecretsStore | None:
        return None

    async def get_secrets(self) -> Secrets | None:
        return None

    async def get_mcp_api_key(self) -> str | None:
        return None

    async def get_user_git_info(self) -> UserGitInfo | None:
        return None

    @classmethod
    async def get_instance(cls, request: Request) -> UserAuth:
        return MockUserAuth()

    @classmethod
    async def get_for_user(cls, user_id: str) -> UserAuth:
        return MockUserAuth()


@pytest.fixture
def test_client():
    # Create a test client
    with (
        patch.dict(
            os.environ,
            {'SESSION_API_KEY': '', 'ALLOW_SHORT_CONTEXT_WINDOWS': 'true'},
            clear=False,
        ),
        patch('openhands.app_server.utils.dependencies._SESSION_API_KEY', None),
        patch(
            'openhands.server.user_auth.user_auth.UserAuth.get_instance',
            return_value=MockUserAuth(),
        ),
        patch(
            'openhands.storage.settings.file_settings_store.FileSettingsStore.get_instance',
            AsyncMock(return_value=FileSettingsStore(InMemoryFileStore())),
        ),
    ):
        client = TestClient(app)
        yield client


def test_get_agent_settings_schema_includes_critic_verification_fields(test_client):
    response = test_client.get('/api/v1/settings/agent-schema')

    assert response.status_code == 200
    schema = response.json()
    section_keys = [s['key'] for s in schema['sections']]
    assert 'verification' in section_keys
    section = next(s for s in schema['sections'] if s['key'] == 'verification')
    field_keys = [f['key'] for f in section['fields']]
    assert 'verification.critic_enabled' in field_keys
    assert 'confirmation_mode' not in field_keys
    assert 'security_analyzer' not in field_keys


def test_get_conversation_settings_schema_endpoint(test_client):
    response = test_client.get('/api/v1/settings/conversation-schema')

    assert response.status_code == 200
    schema = response.json()
    assert schema['model_name'] == 'ConversationSettings'
    section_keys = [s['key'] for s in schema['sections']]
    assert section_keys == ['general', 'verification']
    verification_section = next(
        s for s in schema['sections'] if s['key'] == 'verification'
    )
    field_keys = [f['key'] for f in verification_section['fields']]
    assert 'confirmation_mode' in field_keys
    assert 'security_analyzer' in field_keys


@pytest.mark.asyncio
async def test_settings_api_endpoints(test_client):
    """Test that the settings API endpoints work with the new auth system."""
    settings = Settings(
        language='en',
        remote_runtime_resource_factor=2,
        agent_settings=AgentSettings(
            agent='test-agent',
            llm=LLM(
                model='test-model',
                api_key=SecretStr('test-key'),
                base_url='https://test.com',
                timeout=123,
                litellm_extra_body={'metadata': {'tier': 'pro'}},
            ),
            verification=VerificationSettings(
                critic_enabled=True,
                critic_mode='all_actions',
                enable_iterative_refinement=True,
                critic_threshold=0.7,
                max_refinement_iterations=4,
            ),
        ),
        conversation_settings=ConversationSettings(
            max_iterations=100,
            confirmation_mode=True,
            security_analyzer='llm',
        ),
    )

    # Make the POST request to store settings (V1 endpoint)
    response = test_client.post('/api/v1/settings', json=_dump(settings))

    # We're not checking the exact response, just that it doesn't error
    assert response.status_code == 200

    # Test the GET settings endpoint (V1 endpoint)
    response = test_client.get('/api/v1/settings')
    assert response.status_code == 200
    response_data = response.json()
    assert 'agent_settings_schema' not in response_data
    vals = response_data['agent_settings']
    assert vals['llm']['model'] == 'test-model'
    assert vals['llm']['timeout'] == 123
    assert vals['llm']['litellm_extra_body'] == {'metadata': {'tier': 'pro'}}
    assert vals['verification']['critic_enabled'] is True
    assert vals['verification']['critic_mode'] == 'all_actions'
    assert vals['verification']['enable_iterative_refinement'] is True
    assert vals['verification']['critic_threshold'] == 0.7
    assert vals['verification']['max_refinement_iterations'] == 4
    cs = response_data['conversation_settings']
    assert cs['confirmation_mode'] is True
    assert cs['security_analyzer'] == 'llm'
    assert cs['max_iterations'] == 100
    # V1 API sets api_key to None for security and uses llm_api_key_set flag instead
    assert vals['llm']['api_key'] is None
    assert response_data['llm_api_key_set'] is True

    # Test updating with partial settings — legacy flat fields should preserve existing
    partial_settings = {
        'language': 'fr',
        'llm_model': None,
        'llm_api_key': None,
    }

    response = test_client.post('/api/v1/settings', json=partial_settings)
    assert response.status_code == 200

    response = test_client.get('/api/v1/settings')
    assert response.status_code == 200
    assert response.json()['agent_settings']['llm']['timeout'] == 123


@pytest.mark.asyncio
async def test_saving_settings_with_frozen_secrets_store(test_client):
    """Regression: POSTing settings must not fail with `secrets_store`.

    See https://github.com/OpenHands/OpenHands/issues/13306.
    """
    payload = _dump(
        Settings(
            language='en',
            agent_settings=AgentSettings(llm=LLM(model='gpt-4')),
        )
    )
    # Inject an extra key the API should ignore gracefully
    payload['secrets_store'] = {'provider_tokens': {}}
    response = test_client.post('/api/v1/settings', json=payload)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_search_api_key_explicit_clear(test_client):
    """Explicit empty search_api_key payloads should clear the stored secret."""
    response = test_client.post(
        '/api/v1/settings',
        json=_dump(
            Settings(
                search_api_key='initial-secret-key',
                agent_settings=AgentSettings(llm=LLM(model='gpt-4')),
            )
        ),
    )
    assert response.status_code == 200

    response = test_client.get('/api/v1/settings')
    assert response.status_code == 200
    assert response.json()['search_api_key_set'] is True

    response = test_client.post(
        '/api/v1/settings',
        json=_dump(
            Settings(
                search_api_key='',
                agent_settings=AgentSettings(llm=LLM(model='claude-3-opus')),
            )
        ),
    )
    assert response.status_code == 200

    response = test_client.get('/api/v1/settings')
    assert response.status_code == 200
    assert response.json()['search_api_key_set'] is False
    assert response.json()['agent_settings']['llm']['model'] == 'claude-3-opus'


@pytest.mark.asyncio
async def test_disabled_skills_persistence(test_client):
    """Test that disabled_skills can be saved and retrieved via the settings API."""
    response = test_client.post(
        '/api/v1/settings',
        json=_dump(
            Settings(
                disabled_skills=['skill_a', 'skill_b'],
                agent_settings=AgentSettings(llm=LLM(model='test-model')),
            )
        ),
    )
    assert response.status_code == 200

    response = test_client.get('/api/v1/settings')
    assert response.status_code == 200
    data = response.json()
    assert data['disabled_skills'] == ['skill_a', 'skill_b']

    response = test_client.post(
        '/api/v1/settings',
        json=_dump(Settings(disabled_skills=['skill_c'])),
    )
    assert response.status_code == 200

    response = test_client.get('/api/v1/settings')
    assert response.status_code == 200
    data = response.json()
    assert data['disabled_skills'] == ['skill_c']

    response = test_client.post(
        '/api/v1/settings',
        json=_dump(Settings(disabled_skills=[])),
    )
    assert response.status_code == 200

    response = test_client.get('/api/v1/settings')
    assert response.status_code == 200
    data = response.json()
    assert data['disabled_skills'] == []
