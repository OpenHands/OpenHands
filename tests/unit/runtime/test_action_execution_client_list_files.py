"""Test that ActionExecutionClient.list_files always sends recursive parameter."""

from unittest.mock import MagicMock, patch

from openhands.runtime.impl.action_execution.action_execution_client import (
    ActionExecutionClient,
)


class TestActionExecutionClientListFiles:
    """Test the list_files method of ActionExecutionClient."""

    @patch(
        'openhands.runtime.impl.action_execution.action_execution_client.send_request'
    )
    def test_list_files_always_includes_recursive_false(self, mock_send_request):
        """Test that recursive=False is explicitly sent in the request body."""
        # Setup
        mock_response = MagicMock()
        mock_response.json.return_value = ['file1.txt', 'file2.txt']
        mock_response.is_closed = True
        mock_send_request.return_value = mock_response

        client = MagicMock(spec=ActionExecutionClient)
        client.session = MagicMock()
        client.action_execution_server_url = 'http://test-server'
        client._send_action_server_request = MagicMock(return_value=mock_response)
        client.log = MagicMock()

        # Call the actual method
        ActionExecutionClient.list_files(client, path='/test', recursive=False)

        # Assert recursive=False was sent
        client._send_action_server_request.assert_called_once_with(
            'POST',
            'http://test-server/list_files',
            json={'path': '/test', 'recursive': False},  # Explicitly False
            timeout=10,
        )

    @patch(
        'openhands.runtime.impl.action_execution.action_execution_client.send_request'
    )
    def test_list_files_always_includes_recursive_true(self, mock_send_request):
        """Test that recursive=True is sent in the request body."""
        # Setup
        mock_response = MagicMock()
        mock_response.json.return_value = ['file1.txt', 'dir1/', 'dir1/file2.txt']
        mock_response.is_closed = True
        mock_send_request.return_value = mock_response

        client = MagicMock(spec=ActionExecutionClient)
        client.session = MagicMock()
        client.action_execution_server_url = 'http://test-server'
        client._send_action_server_request = MagicMock(return_value=mock_response)
        client.log = MagicMock()

        # Call the actual method
        ActionExecutionClient.list_files(client, path='/test', recursive=True)

        # Assert recursive=True was sent
        client._send_action_server_request.assert_called_once_with(
            'POST',
            'http://test-server/list_files',
            json={'path': '/test', 'recursive': True},
            timeout=10,
        )

    @patch(
        'openhands.runtime.impl.action_execution.action_execution_client.send_request'
    )
    def test_list_files_default_recursive_false(self, mock_send_request):
        """Test that when recursive is not specified, it defaults to False and is still sent."""
        # Setup
        mock_response = MagicMock()
        mock_response.json.return_value = ['file1.txt']
        mock_response.is_closed = True
        mock_send_request.return_value = mock_response

        client = MagicMock(spec=ActionExecutionClient)
        client.session = MagicMock()
        client.action_execution_server_url = 'http://test-server'
        client._send_action_server_request = MagicMock(return_value=mock_response)
        client.log = MagicMock()

        # Call without specifying recursive (should default to False)
        ActionExecutionClient.list_files(client, path='/test')

        # Assert recursive=False was sent (default value)
        client._send_action_server_request.assert_called_once_with(
            'POST',
            'http://test-server/list_files',
            json={'path': '/test', 'recursive': False},  # Default False is sent
            timeout=10,
        )
