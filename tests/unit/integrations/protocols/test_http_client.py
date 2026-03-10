"""Unit tests for HTTPClient abstract base class (ABC)."""

from typing import Any
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from pydantic import SecretStr

from openhands.integrations.protocols.http_client import HTTPClient
from openhands.integrations.service_types import (
    AuthenticationError,
    ProviderTimeoutError,
    RateLimitError,
    RequestMethod,
    ResourceNotFoundError,
    UnknownException,
)


class TestableHTTPClient(HTTPClient):
    """Testable concrete implementation of HTTPClient for unit testing."""

    def __init__(self, provider_name: str = 'test-provider'):
        self.token = SecretStr('test-token')
        self.refresh = False
        self.external_auth_id = None
        self.external_auth_token = None
        self.external_token_manager = False
        self.base_domain = None
        self._provider_name = provider_name

    @property
    def provider(self) -> str:
        return self._provider_name

    @provider.setter
    def provider(self, value: str) -> None:
        self._provider_name = value

    async def get_latest_token(self) -> SecretStr | None:
        return self.token

    async def _get_headers(self) -> dict[str, Any]:
        return {'Authorization': f'Bearer {self.token.get_secret_value()}'}

    async def _make_request(
        self,
        url: str,
        params: dict | None = None,
        method: RequestMethod = RequestMethod.GET,
    ):
        # Mock implementation for testing
        return {'test': 'data'}, {}


@pytest.mark.asyncio
class TestHTTPClient:
    """Test cases for HTTPClient ABC."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestableHTTPClient()

    def test_default_attributes(self):
        """Test default attribute values."""
        assert isinstance(self.client.token, SecretStr)
        assert self.client.refresh is False
        assert self.client.external_auth_id is None
        assert self.client.external_auth_token is None
        assert self.client.external_token_manager is False
        assert self.client.base_domain is None

    def test_provider_property(self):
        """Test provider property."""
        assert self.client.provider == 'test-provider'

    def test_has_token_expired_default_implementation(self):
        """Test default _has_token_expired implementation."""
        # The TestableHTTPClient inherits the default implementation from the protocol
        client = TestableHTTPClient()

        assert client._has_token_expired(401) is True
        assert client._has_token_expired(200) is False
        assert client._has_token_expired(404) is False
        assert client._has_token_expired(500) is False

    async def test_execute_request_get(self):
        """Test execute_request with GET method."""
        client = TestableHTTPClient()

        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_client.get.return_value = mock_response

        url = 'https://api.example.com/user'
        headers = {'Authorization': 'Bearer token'}
        params = {'per_page': 10}

        result = await client.execute_request(
            mock_client, url, headers, params, RequestMethod.GET
        )

        assert result == mock_response
        mock_client.get.assert_called_once_with(url, headers=headers, params=params)

    async def test_execute_request_post(self):
        """Test execute_request with POST method."""
        client = TestableHTTPClient()

        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_client.post.return_value = mock_response

        url = 'https://api.example.com/issues'
        headers = {'Authorization': 'Bearer token'}
        params = {'title': 'Test Issue'}

        result = await client.execute_request(
            mock_client, url, headers, params, RequestMethod.POST
        )

        assert result == mock_response
        mock_client.post.assert_called_once_with(url, headers=headers, json=params)

    def test_handle_http_status_error_401(self):
        """Test handling of 401 HTTP status error."""
        client = TestableHTTPClient('github')

        mock_response = Mock()
        mock_response.status_code = 401

        error = httpx.HTTPStatusError(
            message='401 Unauthorized', request=Mock(), response=mock_response
        )

        result = client.handle_http_status_error(error)
        assert isinstance(result, AuthenticationError)
        assert 'Invalid github token' in str(result)

    def test_handle_http_status_error_404(self):
        """Test handling of 404 HTTP status error."""
        client = TestableHTTPClient()
        client.provider = 'gitlab'

        mock_response = Mock()
        mock_response.status_code = 404

        error = httpx.HTTPStatusError(
            message='404 Not Found', request=Mock(), response=mock_response
        )

        result = client.handle_http_status_error(error)
        assert isinstance(result, ResourceNotFoundError)
        assert 'Resource not found on gitlab API' in str(result)

    def test_handle_http_status_error_429(self):
        """Test handling of 429 HTTP status error."""
        client = TestableHTTPClient()
        client.provider = 'bitbucket'

        mock_response = Mock()
        mock_response.status_code = 429

        error = httpx.HTTPStatusError(
            message='429 Too Many Requests', request=Mock(), response=mock_response
        )

        result = client.handle_http_status_error(error)
        assert isinstance(result, RateLimitError)
        assert 'bitbucket API rate limit exceeded' in str(result)

    def test_handle_http_status_error_other(self):
        """Test handling of other HTTP status errors."""
        client = TestableHTTPClient()
        client.provider = 'test-provider'

        mock_response = Mock()
        mock_response.status_code = 500

        error = httpx.HTTPStatusError(
            message='500 Internal Server Error', request=Mock(), response=mock_response
        )

        result = client.handle_http_status_error(error)
        assert isinstance(result, UnknownException)
        assert 'Unknown error' in str(result)

    def test_handle_http_error(self):
        """Test handling of general HTTP errors."""
        client = TestableHTTPClient()
        client.provider = 'test-provider'

        error = httpx.ConnectError('Connection failed')

        result = client.handle_http_error(error)
        assert isinstance(result, UnknownException)
        assert 'HTTP error ConnectError' in str(result)

    def test_handle_http_error_with_different_error_types(self):
        """Test handling of different HTTP error types."""
        client = TestableHTTPClient()
        client.provider = 'test-provider'

        # Test with non-timeout error (should return UnknownException)
        connect_error = httpx.ConnectError('Connection failed')
        result = client.handle_http_error(connect_error)
        assert isinstance(result, UnknownException)
        assert 'HTTP error ConnectError' in str(result)

        # Test with timeout errors (should return ProviderTimeoutError)
        timeout_errors = [
            httpx.TimeoutException('Request timed out'),
            httpx.ReadTimeout('Read timeout'),
            httpx.WriteTimeout('Write timeout'),
        ]

        for error in timeout_errors:
            result = client.handle_http_error(error)
            assert isinstance(result, ProviderTimeoutError)
            assert 'test-provider API request timed out' in str(result)
            assert type(error).__name__ in str(result)

    def test_runtime_checkable(self):
        """Test that HTTPClient is runtime checkable."""
        from openhands.integrations.protocols.http_client import HTTPClient

        # Test that our testable client implements the protocol
        assert isinstance(self.client, HTTPClient)

        # Test that a class without the required methods doesn't implement the protocol
        class IncompleteClient:
            pass

        incomplete = IncompleteClient()
        assert not isinstance(incomplete, HTTPClient)

    def test_protocol_attributes_exist(self):
        """Test that protocol defines expected attributes."""
        client = TestableHTTPClient()

        # Test default attribute values from protocol
        assert hasattr(client, 'token')
        assert hasattr(client, 'refresh')
        assert hasattr(client, 'external_auth_id')
        assert hasattr(client, 'external_auth_token')
        assert hasattr(client, 'external_token_manager')
        assert hasattr(client, 'base_domain')

        # Test TestableHTTPClient values
        assert client.token == SecretStr('test-token')
        assert client.refresh is False
        assert client.external_auth_id is None
        assert client.external_auth_token is None
        assert client.external_token_manager is False
        assert client.base_domain is None

    def test_protocol_methods_exist(self):
        """Test that protocol defines expected methods."""
        client = TestableHTTPClient()

        # Test that methods exist
        assert hasattr(client, 'get_latest_token')
        assert hasattr(client, '_get_headers')
        assert hasattr(client, '_make_request')
        assert hasattr(client, '_has_token_expired')
        assert hasattr(client, 'execute_request')
        assert hasattr(client, 'handle_http_status_error')
        assert hasattr(client, 'handle_http_error')
        assert hasattr(client, 'provider')

    def test_protocol_concrete_methods_work(self):
        """Test that concrete protocol methods work correctly."""
        client = TestableHTTPClient()

        # These methods should work since TestableHTTPClient implements them
        assert client.provider == 'test-provider'

        # Test that the default implementations from the protocol are available
        assert hasattr(client, '_has_token_expired')
        assert hasattr(client, 'execute_request')
        assert hasattr(client, 'handle_http_status_error')
        assert hasattr(client, 'handle_http_error')

    def test_provider_specific_error_messages(self):
        """Test that error messages are provider-specific."""
        providers = ['github', 'gitlab', 'bitbucket']

        for provider in providers:
            client = TestableHTTPClient()
            client.provider = provider

            # Test 401 error
            mock_response = Mock()
            mock_response.status_code = 401
            error = httpx.HTTPStatusError(
                message='401 Unauthorized', request=Mock(), response=mock_response
            )
            result = client.handle_http_status_error(error)
            assert f'Invalid {provider} token' in str(result)

            # Test 404 error
            mock_response.status_code = 404
            error = httpx.HTTPStatusError(
                message='404 Not Found', request=Mock(), response=mock_response
            )
            result = client.handle_http_status_error(error)
            assert f'Resource not found on {provider} API' in str(result)

            # Test 429 error
            mock_response.status_code = 429
            error = httpx.HTTPStatusError(
                message='429 Too Many Requests', request=Mock(), response=mock_response
            )
            result = client.handle_http_status_error(error)
            assert f'{provider} API rate limit exceeded' in str(result)
