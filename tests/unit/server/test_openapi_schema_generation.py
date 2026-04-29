import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from pydantic import SecretStr

from openhands.app_server.file_store.memory import InMemoryFileStore
from openhands.app_server.integrations.provider import ProviderToken, ProviderType
from openhands.app_server.integrations.service_types import UserGitInfo
from openhands.app_server.secrets.secrets_models import Secrets
from openhands.app_server.secrets.secrets_store import SecretsStore
from openhands.app_server.settings.file_settings_store import FileSettingsStore
from openhands.app_server.settings.settings_store import SettingsStore
from openhands.app_server.user_auth.user_auth import UserAuth
from openhands.server.app import app


class MockUserAuth(UserAuth):
    """Mock implementation of UserAuth for testing"""

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

    async def get_provider_tokens(self) -> dict[ProviderType, ProviderToken] | None:
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
        patch(
            'openhands.app_server.user_auth.user_auth.UserAuth.get_instance',
            return_value=MockUserAuth(),
        ),
        patch(
            'openhands.app_server.settings.file_settings_store.FileSettingsStore.get_instance',
            AsyncMock(return_value=FileSettingsStore(InMemoryFileStore())),
        ),
    ):
        client = TestClient(app)
        yield client


@pytest.mark.asyncio
async def test_openapi_schema_generation(test_client):
    """Test that the OpenAPI schema can be generated without errors.

    This test ensures that the FastAPI app can generate a valid OpenAPI schema,
    which is important for API documentation and client generation.
    """
    # Get the OpenAPI schema from the /openapi.json endpoint
    response = test_client.get('/openapi.json')

    # Check that the response is successful
    assert response.status_code == 200

    # Verify that the response is valid JSON
    schema = response.json()

    # Basic validation of the schema structure
    assert 'openapi' in schema
    assert 'info' in schema
    assert 'paths' in schema

    # Verify that the schema can be serialized to JSON without errors
    # This is a good test for ensuring all types can be properly converted to JSON Schema
    json_str = json.dumps(schema)
    assert json_str is not None

    # Optionally, you can check for specific endpoints if needed
    assert '/api/v1/settings' in schema['paths']
    assert '/health' in schema['paths']
