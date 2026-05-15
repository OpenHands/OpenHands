import warnings
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openhands.app_server.integrations.service_types import GitService
from openhands.app_server.mcp.mcp_router import get_conversation_link, init_tavily_proxy
from openhands.app_server.types import AppMode


def test_mcp_server_no_stateless_http_deprecation_warning():
    """Test that mcp_server is created without stateless_http deprecation warning.

    This test verifies the fix for the fastmcp deprecation warning:
    'Providing `stateless_http` when creating a server is deprecated.
    Provide it when calling `run` or as a global setting instead.'

    The fix moves the stateless_http parameter from FastMCP() constructor
    to the http_app() method call.
    """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')

        # Import the mcp_server which triggers FastMCP creation
        from openhands.app_server.mcp.mcp_router import mcp_server

        # Check that no deprecation warning about stateless_http was raised
        stateless_http_warnings = [
            warning
            for warning in w
            if issubclass(warning.category, DeprecationWarning)
            and 'stateless_http' in str(warning.message)
        ]

        assert len(stateless_http_warnings) == 0, (
            f'Unexpected stateless_http deprecation warning: {stateless_http_warnings}'
        )

        # Verify mcp_server was created successfully
        assert mcp_server is not None


@pytest.mark.asyncio
async def test_get_conversation_link_non_saas_mode():
    """Test get_conversation_link in non-SAAS mode."""
    # Mock GitService
    mock_service = AsyncMock(spec=GitService)

    # Test with non-SAAS mode
    with patch('openhands.app_server.mcp.mcp_router.get_global_config') as mock_config:
        mock_config.return_value.app_mode = AppMode.OPENHANDS

        # Call the function
        result = await get_conversation_link(
            service=mock_service, conversation_id='test-convo-id', body='Original body'
        )

        # Verify the result
        assert result == 'Original body'
        # Verify that get_user was not called
        mock_service.get_user.assert_not_called()


@pytest.mark.asyncio
async def test_get_conversation_link_saas_mode():
    """Test get_conversation_link in SAAS mode."""
    # Mock GitService and user
    mock_service = AsyncMock(spec=GitService)
    mock_user = AsyncMock()
    mock_user.login = 'testuser'
    mock_service.get_user.return_value = mock_user

    # Test with SAAS mode
    with (
        patch('openhands.app_server.mcp.mcp_router.get_global_config') as mock_config,
        patch(
            'openhands.app_server.mcp.mcp_router.CONVERSATION_URL',
            'https://test.example.com/conversations/{}',
        ),
    ):
        mock_config.return_value.app_mode = AppMode.SAAS

        # Call the function
        result = await get_conversation_link(
            service=mock_service, conversation_id='test-convo-id', body='Original body'
        )

        # Verify the result
        expected_link = '@testuser can click here to [continue refining the PR](https://test.example.com/conversations/test-convo-id)'
        assert result == f'Original body\n\n{expected_link}'

        # Verify that get_user was called
        mock_service.get_user.assert_called_once()


@pytest.mark.asyncio
async def test_get_conversation_link_empty_body():
    """Test get_conversation_link with an empty body."""
    # Mock GitService and user
    mock_service = AsyncMock(spec=GitService)
    mock_user = AsyncMock()
    mock_user.login = 'testuser'
    mock_service.get_user.return_value = mock_user

    # Test with SAAS mode and empty body
    with (
        patch('openhands.app_server.mcp.mcp_router.get_global_config') as mock_config,
        patch(
            'openhands.app_server.mcp.mcp_router.CONVERSATION_URL',
            'https://test.example.com/conversations/{}',
        ),
    ):
        mock_config.return_value.app_mode = AppMode.SAAS

        # Call the function
        result = await get_conversation_link(
            service=mock_service, conversation_id='test-convo-id', body=''
        )

        # Verify the result
        expected_link = '@testuser can click here to [continue refining the PR](https://test.example.com/conversations/test-convo-id)'
        assert result == f'\n\n{expected_link}'

        # Verify that get_user was called
        mock_service.get_user.assert_called_once()


@pytest.mark.asyncio
async def test_get_conversation_link_none_conversation_id():
    """Test get_conversation_link returns body unchanged when conversation_id is None."""
    mock_service = AsyncMock(spec=GitService)

    with patch('openhands.app_server.mcp.mcp_router.get_global_config') as mock_config:
        mock_config.return_value.app_mode = AppMode.SAAS

        body = 'This is the PR body.'

        # Test with None conversation_id
        result = await get_conversation_link(
            service=mock_service, conversation_id=None, body=body
        )
        assert result == body

        # Test with empty string conversation_id
        result = await get_conversation_link(
            service=mock_service, conversation_id='', body=body
        )
        assert result == body

        # Verify get_user was never called (early return)
        mock_service.get_user.assert_not_called()


class TestInitTavilyProxy:
    """Tests for init_tavily_proxy function."""

    def test_init_tavily_proxy_no_api_key(self):
        """Test init_tavily_proxy does nothing when no API key is configured."""
        with (
            patch(
                'openhands.app_server.mcp.mcp_router.get_global_config'
            ) as mock_config,
            patch('openhands.app_server.mcp.mcp_router.logger') as mock_logger,
            patch('openhands.app_server.mcp.mcp_router.Client') as mock_client,
            patch('openhands.app_server.mcp.mcp_router.create_proxy') as mock_proxy,
            patch('openhands.app_server.mcp.mcp_router.mcp_server') as mock_mcp_server,
        ):
            # Configure no API key
            mock_config.return_value.tavily_api_key = None

            # Call the function
            init_tavily_proxy()

            # Verify it logged the skip message
            mock_logger.info.assert_called_once_with(
                'Tavily API key not configured, skipping Tavily MCP proxy'
            )

            # Verify no proxy was created
            mock_client.assert_not_called()
            mock_proxy.assert_not_called()
            mock_mcp_server.mount.assert_not_called()

    def test_init_tavily_proxy_empty_api_key(self):
        """Test init_tavily_proxy does nothing when API key is empty string."""
        with (
            patch(
                'openhands.app_server.mcp.mcp_router.get_global_config'
            ) as mock_config,
            patch('openhands.app_server.mcp.mcp_router.logger') as mock_logger,
            patch('openhands.app_server.mcp.mcp_router.Client') as mock_client,
            patch('openhands.app_server.mcp.mcp_router.create_proxy') as mock_proxy,
            patch('openhands.app_server.mcp.mcp_router.mcp_server') as mock_mcp_server,
        ):
            # Configure empty API key
            mock_config.return_value.tavily_api_key = ''

            # Call the function
            init_tavily_proxy()

            # Verify it logged the skip message
            mock_logger.info.assert_called_once_with(
                'Tavily API key not configured, skipping Tavily MCP proxy'
            )

            # Verify no proxy was created
            mock_client.assert_not_called()
            mock_proxy.assert_not_called()
            mock_mcp_server.mount.assert_not_called()

    def test_init_tavily_proxy_with_api_key(self):
        """Test init_tavily_proxy creates and mounts proxy when API key is configured."""
        with (
            patch(
                'openhands.app_server.mcp.mcp_router.get_global_config'
            ) as mock_config,
            patch('openhands.app_server.mcp.mcp_router.logger') as mock_logger,
            patch('openhands.app_server.mcp.mcp_router.Client') as mock_client,
            patch(
                'openhands.app_server.mcp.mcp_router.StreamableHttpTransport'
            ) as mock_transport,
            patch(
                'openhands.app_server.mcp.mcp_router.create_proxy'
            ) as mock_create_proxy,
            patch('openhands.app_server.mcp.mcp_router.mcp_server') as mock_mcp_server,
        ):
            # Configure API key
            mock_config.return_value.tavily_api_key = 'test-tavily-key'

            # Setup mocks
            mock_transport_instance = MagicMock()
            mock_transport.return_value = mock_transport_instance
            mock_client_instance = MagicMock()
            mock_client.return_value = mock_client_instance
            mock_proxy_server = MagicMock()
            mock_create_proxy.return_value = mock_proxy_server

            # Call the function
            init_tavily_proxy()

            # Verify transport was created with correct URL
            mock_transport.assert_called_once_with(
                url='https://mcp.tavily.com/mcp/?tavilyApiKey=test-tavily-key'
            )

            # Verify client was created with the transport
            mock_client.assert_called_once_with(transport=mock_transport_instance)

            # Verify proxy was created from the client
            mock_create_proxy.assert_called_once_with(mock_client_instance)

            # Verify proxy was mounted with correct namespace
            mock_mcp_server.mount.assert_called_once_with(
                namespace='tavily', server=mock_proxy_server
            )

            # Verify success was logged
            mock_logger.info.assert_called_once_with(
                'Tavily MCP proxy initialized successfully'
            )

    def test_init_tavily_proxy_handles_exception(self):
        """Test init_tavily_proxy handles exceptions gracefully."""
        with (
            patch(
                'openhands.app_server.mcp.mcp_router.get_global_config'
            ) as mock_config,
            patch('openhands.app_server.mcp.mcp_router.logger') as mock_logger,
            patch('openhands.app_server.mcp.mcp_router.Client') as mock_client,
            patch('openhands.app_server.mcp.mcp_router.StreamableHttpTransport'),
            patch('openhands.app_server.mcp.mcp_router.mcp_server') as mock_mcp_server,
        ):
            # Configure API key
            mock_config.return_value.tavily_api_key = 'test-tavily-key'

            # Make Client raise an exception
            mock_client.side_effect = Exception('Connection failed')

            # Call the function - should not raise
            init_tavily_proxy()

            # Verify error was logged
            mock_logger.error.assert_called_once_with(
                'Failed to initialize Tavily MCP proxy: Connection failed'
            )

            # Verify mount was not called
            mock_mcp_server.mount.assert_not_called()
