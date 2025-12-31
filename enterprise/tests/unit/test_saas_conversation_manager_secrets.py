"""Tests for SaasNestedConversationManager custom secrets handling during resume."""

from types import MappingProxyType
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from pydantic import SecretStr
from server.saas_nested_conversation_manager import SaasNestedConversationManager

from openhands.core.config.openhands_config import OpenHandsConfig
from openhands.integrations.provider import CustomSecret
from openhands.server.config.server_config import ServerConfig
from openhands.storage.memory import InMemoryFileStore


class MockHTTPXResponse:
    """Mock httpx.Response that behaves realistically."""

    def __init__(self, status_code: int, json_data: dict | None = None):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = str(json_data) if json_data else ''

    def json(self):
        """Return JSON data."""
        if self._json_data:
            return self._json_data
        raise ValueError('No JSON data')

    def raise_for_status(self):
        """Raise an exception for 4xx/5xx status codes."""
        if self.status_code >= 400:
            # Create a proper mock response for the exception
            mock_response = MagicMock()
            mock_response.status_code = self.status_code
            mock_response.json = self.json
            mock_response.text = self.text

            error = httpx.HTTPStatusError(
                f"Client error '{self.status_code}' for url 'test'",
                request=MagicMock(),
                response=mock_response,
            )
            raise error


@pytest.fixture
def saas_manager():
    """Create a SaasNestedConversationManager instance for testing."""
    manager = SaasNestedConversationManager(
        sio=MagicMock(),
        config=MagicMock(spec=OpenHandsConfig),
        server_config=MagicMock(spec=ServerConfig),
        file_store=MagicMock(spec=InMemoryFileStore),
        event_retrieval=MagicMock(),
    )
    return manager


@pytest.mark.asyncio
async def test_duplicate_secrets_dont_crash_resume(saas_manager):
    """Test that duplicate secrets during resume are handled gracefully."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    # Simulate resume scenario: secret already exists (400)
    mock_response = MockHTTPXResponse(
        400, {'message': 'Secret MY_API_KEY already exists'}
    )

    async def mock_post(*args, **kwargs):
        return mock_response

    mock_client.post = AsyncMock(side_effect=mock_post)

    custom_secrets = MappingProxyType(
        {
            'MY_API_KEY': CustomSecret(
                secret=SecretStr('api_key_value'),
                description='API Key that already exists on resume',
            ),
        }
    )

    # Should not raise despite 400 "already exists" error
    await saas_manager._setup_custom_secrets(
        client=mock_client,
        api_url='https://runtime.example.com',
        custom_secrets=custom_secrets,
    )

    assert mock_client.post.call_count == 1


@pytest.mark.asyncio
async def test_other_400_errors_still_fail(saas_manager):
    """Test that non-duplicate 400 errors are still raised."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    # 400 error but NOT a duplicate
    mock_response = MockHTTPXResponse(400, {'message': 'Invalid secret name format'})

    async def mock_post(*args, **kwargs):
        return mock_response

    mock_client.post = AsyncMock(side_effect=mock_post)

    custom_secrets = MappingProxyType(
        {
            'INVALID!NAME': CustomSecret(
                secret=SecretStr('value'), description='Secret with invalid name'
            ),
        }
    )

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        await saas_manager._setup_custom_secrets(
            client=mock_client,
            api_url='https://runtime.example.com',
            custom_secrets=custom_secrets,
        )

    assert exc_info.value.response.status_code == 400


@pytest.mark.asyncio
async def test_normal_secret_creation_still_works(saas_manager):
    """Test that normal secret creation works correctly."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    # Successful creation
    mock_response = MockHTTPXResponse(200, {'message': 'Secret created'})

    async def mock_post(*args, **kwargs):
        return mock_response

    mock_client.post = AsyncMock(side_effect=mock_post)

    custom_secrets = MappingProxyType(
        {
            'NEW_SECRET': CustomSecret(
                secret=SecretStr('new_value'), description='A new secret'
            ),
        }
    )

    await saas_manager._setup_custom_secrets(
        client=mock_client,
        api_url='https://runtime.example.com',
        custom_secrets=custom_secrets,
    )

    assert mock_client.post.call_count == 1
    call_args = mock_client.post.call_args_list[0]
    assert call_args[1]['json']['name'] == 'NEW_SECRET'
    assert call_args[1]['json']['value'] == 'new_value'


@pytest.mark.asyncio
async def test_handles_empty_secrets_gracefully(saas_manager):
    """Test that empty or missing secrets are handled correctly."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    # Test with None
    await saas_manager._setup_custom_secrets(
        client=mock_client, api_url='https://runtime.example.com', custom_secrets=None
    )
    assert mock_client.post.call_count == 0

    # Test with empty dict
    await saas_manager._setup_custom_secrets(
        client=mock_client,
        api_url='https://runtime.example.com',
        custom_secrets=MappingProxyType({}),
    )
    assert mock_client.post.call_count == 0
