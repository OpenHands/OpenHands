import warnings
from unittest.mock import AsyncMock, patch

import pytest

from openhands.integrations.service_types import GitService
from openhands.server.routes.mcp import get_conversation_link
from openhands.server.types import AppMode


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
        from openhands.server.routes.mcp import mcp_server

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
    with patch('openhands.server.routes.mcp.server_config') as mock_config:
        mock_config.app_mode = AppMode.OPENHANDS

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
        patch('openhands.server.routes.mcp.server_config') as mock_config,
        patch(
            'openhands.server.routes.mcp.CONVERSATION_URL',
            'https://test.example.com/conversations/{}',
        ),
    ):
        mock_config.app_mode = AppMode.SAAS

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
        patch('openhands.server.routes.mcp.server_config') as mock_config,
        patch(
            'openhands.server.routes.mcp.CONVERSATION_URL',
            'https://test.example.com/conversations/{}',
        ),
    ):
        mock_config.app_mode = AppMode.SAAS

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

    with patch('openhands.server.routes.mcp.server_config') as mock_config:
        mock_config.app_mode = AppMode.SAAS

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
