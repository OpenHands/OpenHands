"""Tests for HttpxClientInjector.

This module tests the HttpxClientInjector service, focusing on:
- Client reuse within the same request context
- Client isolation between different requests
- Proper client lifecycle management and cleanup
- Timeout configuration
"""

from unittest.mock import MagicMock, patch

import pytest

from openhands.app_server.services.httpx_client_injector import HttpxClientInjector


class MockRequest:
    """Mock FastAPI Request object for testing."""

    def __init__(self):
        self.state = MagicMock()
        # Initialize state without httpx_client to simulate fresh request
        if hasattr(self.state, 'httpx_client'):
            delattr(self.state, 'httpx_client')


class TestHttpxClientInjector:
    """Test cases for HttpxClientInjector."""

    @pytest.fixture
    def injector(self):
        """Create a HttpxClientInjector instance with default settings."""
        return HttpxClientInjector()

    @pytest.fixture
    def injector_with_custom_timeout(self):
        """Create a HttpxClientInjector instance with custom timeout."""
        return HttpxClientInjector(timeout=30)

    @pytest.fixture
    def mock_request(self):
        """Create a mock FastAPI Request object."""
        return MockRequest()

    @pytest.mark.asyncio
    async def test_creates_new_client_for_fresh_request(self, injector, mock_request):
        """Test that a new httpx client is created for a fresh request."""
        with patch('httpx.AsyncClient') as mock_async_client:
            mock_client_instance = MagicMock()
            mock_async_client.return_value = mock_client_instance

            async for client in injector.depends(mock_request):
                # Verify a new client was created
                mock_async_client.assert_called_once_with(timeout=15)
                assert client is mock_client_instance
                # Verify the client was stored in request state
                assert mock_request.state.httpx_client is mock_client_instance
                break  # Only iterate once since it's a generator

    @pytest.mark.asyncio
    async def test_reuses_existing_client_within_same_request(self, injector):
        """Test that the same httpx client is reused within the same request context."""
        request, existing_client = self.mock_request_with_existing_client()

        with patch('httpx.AsyncClient') as mock_async_client:
            async for client in injector.depends(request):
                # Verify no new client was created
                mock_async_client.assert_not_called()
                # Verify the existing client was returned
                assert client is existing_client
                break  # Only iterate once since it's a generator

    def mock_request_with_existing_client(self):
        """Helper method to create a request with existing client."""
        request = MockRequest()
        existing_client = MagicMock()
        request.state.httpx_client = existing_client
        return request, existing_client

    @pytest.mark.asyncio
    async def test_different_requests_get_different_clients(self, injector):
        """Test that different requests get different client instances."""
        request1 = MockRequest()
        request2 = MockRequest()

        with patch('httpx.AsyncClient') as mock_async_client:
            client1_instance = MagicMock()
            client2_instance = MagicMock()
            mock_async_client.side_effect = [client1_instance, client2_instance]

            # Get client for first request
            async for client1 in injector.depends(request1):
                assert client1 is client1_instance
                assert request1.state.httpx_client is client1_instance
                break

            # Get client for second request
            async for client2 in injector.depends(request2):
                assert client2 is client2_instance
                assert request2.state.httpx_client is client2_instance
                break

            # Verify different clients were created
            assert client1_instance is not client2_instance
            assert mock_async_client.call_count == 2

    @pytest.mark.asyncio
    async def test_multiple_calls_same_request_reuse_client(
        self, injector, mock_request
    ):
        """Test that multiple calls within the same request reuse the same client."""
        with patch('httpx.AsyncClient') as mock_async_client:
            mock_client_instance = MagicMock()
            mock_async_client.return_value = mock_client_instance

            # First call creates client
            async for client1 in injector.depends(mock_request):
                assert client1 is mock_client_instance
                break

            # Second call reuses the same client
            async for client2 in injector.depends(mock_request):
                assert client2 is mock_client_instance
                assert client1 is client2
                break

            # Verify only one client was created
            mock_async_client.assert_called_once()

    @pytest.mark.asyncio
    async def test_custom_timeout_applied_to_client(
        self, injector_with_custom_timeout, mock_request
    ):
        """Test that custom timeout is properly applied to the httpx client."""
        with patch('httpx.AsyncClient') as mock_async_client:
            mock_client_instance = MagicMock()
            mock_async_client.return_value = mock_client_instance

            async for client in injector_with_custom_timeout.depends(mock_request):
                # Verify client was created with custom timeout
                mock_async_client.assert_called_once_with(timeout=30)
                assert client is mock_client_instance
                break

    @pytest.mark.asyncio
    async def test_default_timeout_applied_to_client(self, injector, mock_request):
        """Test that default timeout (15) is applied when no custom timeout is specified."""
        with patch('httpx.AsyncClient') as mock_async_client:
            mock_client_instance = MagicMock()
            mock_async_client.return_value = mock_client_instance

            async for client in injector.depends(mock_request):
                # Verify client was created with default timeout
                mock_async_client.assert_called_once_with(timeout=15)
                assert client is mock_client_instance
                break

    @pytest.mark.asyncio
    async def test_client_lifecycle_async_generator(self, injector, mock_request):
        """Test that the client is properly yielded in the async generator."""
        with patch('httpx.AsyncClient') as mock_async_client:
            mock_client_instance = MagicMock()
            mock_async_client.return_value = mock_client_instance

            # Test that resolve returns an async generator
            resolver = injector.depends(mock_request)
            assert hasattr(resolver, '__aiter__')
            assert hasattr(resolver, '__anext__')

            # Test async generator behavior
            async for client in resolver:
                assert client is mock_client_instance
                # Client should be available during iteration
                assert mock_request.state.httpx_client is mock_client_instance
                break

    @pytest.mark.asyncio
    async def test_request_state_persistence(self, injector):
        """Test that the client persists in request state across multiple resolve calls."""
        request = MockRequest()

        with patch('httpx.AsyncClient') as mock_async_client:
            mock_client_instance = MagicMock()
            mock_async_client.return_value = mock_client_instance

            # First resolve call
            async for client1 in injector.depends(request):
                assert hasattr(request.state, 'httpx_client')
                assert request.state.httpx_client is mock_client_instance
                break

            # Second resolve call - should reuse the same client
            async for client2 in injector.depends(request):
                assert client1 is client2
                assert request.state.httpx_client is mock_client_instance
                break

            # Client should still be in request state after iteration
            assert request.state.httpx_client is mock_client_instance
            # Only one client should have been created
            mock_async_client.assert_called_once()

    @pytest.mark.asyncio
    async def test_injector_configuration_validation(self):
        """Test that HttpxClientInjector validates configuration properly."""
        # Test default configuration
        injector = HttpxClientInjector()
        assert injector.timeout == 15

        # Test custom configuration
        injector_custom = HttpxClientInjector(timeout=60)
        assert injector_custom.timeout == 60

        # Test that configuration is used in client creation
        request = MockRequest()
        with patch('httpx.AsyncClient') as mock_async_client:
            mock_client_instance = MagicMock()
            mock_async_client.return_value = mock_client_instance

            async for client in injector_custom.depends(request):
                mock_async_client.assert_called_once_with(timeout=60)
                break

    @pytest.mark.asyncio
    async def test_concurrent_access_same_request(self, injector, mock_request):
        """Test that concurrent access to the same request returns the same client."""
        import asyncio

        with patch('httpx.AsyncClient') as mock_async_client:
            mock_client_instance = MagicMock()
            mock_async_client.return_value = mock_client_instance

            async def get_client():
                async for client in injector.depends(mock_request):
                    return client

            # Run multiple concurrent calls
            clients = await asyncio.gather(get_client(), get_client(), get_client())

            # All should return the same client instance
            assert all(client is mock_client_instance for client in clients)
            # Only one client should have been created
            mock_async_client.assert_called_once()

    @pytest.mark.asyncio
    async def test_client_cleanup_behavior(self, injector, mock_request):
        """Test the current client cleanup behavior.

        Note: The current implementation stores the client in request.state
        but doesn't explicitly close it. In a real FastAPI application,
        the request state is cleaned up when the request ends, but httpx
        clients should ideally be explicitly closed to free resources.

        This test documents the current behavior. For production use,
        consider implementing a cleanup mechanism using FastAPI's
        dependency system or middleware.
        """
        with patch('httpx.AsyncClient') as mock_async_client:
            mock_client_instance = MagicMock()
            mock_client_instance.aclose = MagicMock()
            mock_async_client.return_value = mock_client_instance

            # Get client from injector
            async for client in injector.depends(mock_request):
                assert client is mock_client_instance
                break

            # Verify client is stored in request state
            assert mock_request.state.httpx_client is mock_client_instance

            # Current implementation doesn't call aclose() automatically
            # This documents the current behavior - client cleanup would need
            # to be handled by FastAPI's request lifecycle or middleware
            mock_client_instance.aclose.assert_not_called()

            # In a real scenario, you might want to manually close the client
            # when the request ends, which could be done via middleware:
            # await mock_request.state.httpx_client.aclose()

    def test_injector_is_pydantic_model(self):
        """Test that HttpxClientInjector is properly configured as a Pydantic model."""
        injector = HttpxClientInjector()

        # Test that it's a Pydantic model
        assert hasattr(injector, 'model_fields')
        assert hasattr(injector, 'model_validate')

        # Test field configuration
        assert 'timeout' in injector.model_fields
        timeout_field = injector.model_fields['timeout']
        assert timeout_field.default == 15
        assert timeout_field.description == 'Default timeout on all http requests'

        # Test model validation
        validated = HttpxClientInjector.model_validate({'timeout': 25})
        assert validated.timeout == 25

    @pytest.mark.asyncio
    async def test_request_state_attribute_handling(self, injector):
        """Test proper handling of request state attributes."""
        request = MockRequest()

        # Initially, request state should not have httpx_client
        assert not hasattr(request.state, 'httpx_client')

        with patch('httpx.AsyncClient') as mock_async_client:
            mock_client_instance = MagicMock()
            mock_async_client.return_value = mock_client_instance

            # After first resolve, client should be stored
            async for client in injector.depends(request):
                assert hasattr(request.state, 'httpx_client')
                assert request.state.httpx_client is mock_client_instance
                break

            # Subsequent calls should use the stored client
            async for client in injector.depends(request):
                assert client is mock_client_instance
                break

            # Only one client should have been created
            mock_async_client.assert_called_once()
