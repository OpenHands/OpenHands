"""Tests for custom secrets setup with fallback handling."""

from types import MappingProxyType
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from pydantic import SecretStr

from openhands.integrations.provider import CustomSecret


class MockResponse:
    """Mock HTTP response."""

    def __init__(self, status_code: int, text: str = ''):
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f'HTTP {self.status_code}',
                request=Mock(),
                response=self,
            )


async def setup_custom_secrets_with_fallback(
    client: httpx.AsyncClient,
    api_url: str,
    custom_secrets: MappingProxyType,
):
    """Simulates the logic in _setup_custom_secrets with error handling."""
    if custom_secrets:
        for key, secret in custom_secrets.items():
            try:
                response = await client.post(
                    f'{api_url}/api/secrets',
                    json={
                        'name': key,
                        'description': secret.description,
                        'value': secret.secret.get_secret_value(),
                    },
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                # If secret already exists (400), log and continue
                if e.response.status_code == 400:
                    continue
                # For other HTTP errors, re-raise
                raise
            except Exception:
                # Log any other unexpected errors but continue
                continue


@pytest.mark.asyncio
async def test_setup_custom_secrets_handles_existing_secret():
    """Test that custom secrets setup handles 400 error when secret already exists."""
    # Create mock client
    mock_client = AsyncMock()

    # Set up the mock to return 400 for the first secret (already exists)
    # and 200 for the second secret (successfully created)
    mock_responses = [
        MockResponse(400, 'Secret already exists'),
        MockResponse(200),
    ]
    mock_client.post = AsyncMock(side_effect=mock_responses)

    # Create custom secrets
    custom_secrets = MappingProxyType(
        {
            'EXISTING_SECRET': CustomSecret(
                secret=SecretStr('value1'), description='Existing secret'
            ),
            'NEW_SECRET': CustomSecret(
                secret=SecretStr('value2'), description='New secret'
            ),
        }
    )

    # This should not raise an exception even though the first secret returns 400
    await setup_custom_secrets_with_fallback(
        mock_client, 'http://test-runtime:60000', custom_secrets
    )

    # Verify both secrets were attempted
    assert mock_client.post.call_count == 2


@pytest.mark.asyncio
async def test_setup_custom_secrets_handles_missing_endpoint():
    """Test that custom secrets setup handles missing endpoint (404) by re-raising."""
    mock_client = AsyncMock()

    # Simulate a 404 error (endpoint doesn't exist in older runtime)
    mock_client.post = AsyncMock(
        side_effect=httpx.HTTPStatusError(
            'HTTP 404', request=Mock(), response=MockResponse(404, 'Not Found')
        )
    )

    custom_secrets = MappingProxyType(
        {
            'TEST_SECRET': CustomSecret(
                secret=SecretStr('value'), description='Test secret'
            ),
        }
    )

    # This should raise since 404 is not a handled case (only 400 is handled)
    with pytest.raises(httpx.HTTPStatusError):
        await setup_custom_secrets_with_fallback(
            mock_client, 'http://test-runtime:60000', custom_secrets
        )


@pytest.mark.asyncio
async def test_setup_custom_secrets_handles_generic_exception():
    """Test that custom secrets setup handles generic exceptions gracefully."""
    mock_client = AsyncMock()

    # Simulate a connection error
    mock_client.post = AsyncMock(side_effect=Exception('Connection refused'))

    custom_secrets = MappingProxyType(
        {
            'TEST_SECRET': CustomSecret(
                secret=SecretStr('value'), description='Test secret'
            ),
        }
    )

    # This should not raise - generic exceptions are caught and logged
    await setup_custom_secrets_with_fallback(
        mock_client, 'http://test-runtime:60000', custom_secrets
    )

    assert mock_client.post.call_count == 1


@pytest.mark.asyncio
async def test_setup_custom_secrets_successful():
    """Test that custom secrets setup works when all secrets are created successfully."""
    mock_client = AsyncMock()

    # Simulate successful creation
    mock_client.post = AsyncMock(return_value=MockResponse(200))

    custom_secrets = MappingProxyType(
        {
            'SECRET1': CustomSecret(secret=SecretStr('value1'), description='Secret 1'),
            'SECRET2': CustomSecret(secret=SecretStr('value2'), description='Secret 2'),
        }
    )

    # This should complete without raising
    await setup_custom_secrets_with_fallback(
        mock_client, 'http://test-runtime:60000', custom_secrets
    )

    # Verify both secrets were created
    assert mock_client.post.call_count == 2
