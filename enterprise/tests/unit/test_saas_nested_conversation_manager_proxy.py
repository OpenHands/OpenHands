"""
Tests for SaasNestedConversationManager proxy methods.

This module tests the proxy functionality that forwards requests
to the nested server with retries and error handling.

Test Coverage:
- _proxy_get_to_nested_server method with retries
- _proxy_post_to_nested_server method with retries
- get_vscode_url proxy method
- get_web_hosts proxy method
- get_microagents proxy method
- send_event_to_conversation with new retry logic
- Error handling and retry behavior
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from enterprise.server.saas_nested_conversation_manager import (
    SaasNestedConversationManager,
    _NESTED_PROXY_MAX_RETRIES,
    _NESTED_PROXY_RETRY_BASE_DELAY,
)


class TestNestedServerProxy:
    """Test suite for nested server proxy methods."""

    @pytest.fixture
    def conversation_manager(self):
        """Create a minimal SaasNestedConversationManager instance for testing."""
        mock_sio = Mock()
        mock_config = Mock()
        mock_config.max_concurrent_conversations = 5
        mock_config.sandbox = Mock()
        mock_config.sandbox.api_key = 'test_api_key'
        mock_server_config = Mock()
        mock_file_store = Mock()

        manager = SaasNestedConversationManager(
            sio=mock_sio,
            config=mock_config,
            server_config=mock_server_config,
            file_store=mock_file_store,
            event_retrieval=Mock(),
        )
        return manager

    @pytest.fixture
    def mock_runtime(self):
        """Create a mock runtime response."""
        return {
            'runtime_id': 'test_runtime_123',
            'session_id': 'test_session_456',
            'session_api_key': 'test_session_api_key_789',
            'status': 'running',
        }

    @pytest.mark.asyncio
    async def test_proxy_get_raises_error_when_runtime_not_found(
        self, conversation_manager
    ):
        """Test that proxy raises error when runtime is not found."""
        sid = 'nonexistent_session'

        with patch.object(
            conversation_manager, '_get_runtime', new_callable=AsyncMock
        ) as mock_get_runtime:
            mock_get_runtime.return_value = None

            with pytest.raises(ValueError, match='no_such_conversation'):
                await conversation_manager._proxy_get_to_nested_server(
                    sid, '/vscode-url'
                )

    @pytest.mark.asyncio
    async def test_proxy_get_raises_error_when_no_session_api_key(
        self, conversation_manager
    ):
        """Test that proxy raises error when session_api_key is missing."""
        sid = 'test_session'
        runtime_without_key = {
            'runtime_id': 'test_runtime',
            'session_id': sid,
            'status': 'running',
        }

        with patch.object(
            conversation_manager, '_get_runtime', new_callable=AsyncMock
        ) as mock_get_runtime:
            mock_get_runtime.return_value = runtime_without_key

            with pytest.raises(ValueError, match='no_session_api_key'):
                await conversation_manager._proxy_get_to_nested_server(
                    sid, '/vscode-url'
                )

    @pytest.mark.asyncio
    async def test_proxy_get_success(self, conversation_manager, mock_runtime):
        """Test successful proxy GET request."""
        sid = 'test_session_456'
        expected_response = {'vscode_url': 'https://vscode.example.com'}

        with patch.object(
            conversation_manager, '_get_runtime', new_callable=AsyncMock
        ) as mock_get_runtime:
            mock_get_runtime.return_value = mock_runtime

            with patch.object(
                conversation_manager,
                '_get_nested_url_for_runtime',
                return_value='https://nested.example.com/api/conversations/test_session_456',
            ):
                with patch('httpx.AsyncClient') as mock_client_class:
                    mock_response = Mock()
                    mock_response.json.return_value = expected_response
                    mock_response.raise_for_status = Mock()

                    mock_client = AsyncMock()
                    mock_client.get.return_value = mock_response
                    mock_client.__aenter__.return_value = mock_client
                    mock_client.__aexit__.return_value = None
                    mock_client_class.return_value = mock_client

                    result = await conversation_manager._proxy_get_to_nested_server(
                        sid, '/vscode-url'
                    )

                    assert result == expected_response
                    mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_proxy_get_retries_on_timeout(self, conversation_manager, mock_runtime):
        """Test that proxy retries on timeout errors."""
        sid = 'test_session_456'
        expected_response = {'vscode_url': 'https://vscode.example.com'}

        with patch.object(
            conversation_manager, '_get_runtime', new_callable=AsyncMock
        ) as mock_get_runtime:
            mock_get_runtime.return_value = mock_runtime

            with patch.object(
                conversation_manager,
                '_get_nested_url_for_runtime',
                return_value='https://nested.example.com/api/conversations/test_session_456',
            ):
                with patch('httpx.AsyncClient') as mock_client_class:
                    # First two calls timeout, third succeeds
                    mock_response = Mock()
                    mock_response.json.return_value = expected_response
                    mock_response.raise_for_status = Mock()

                    mock_client = AsyncMock()
                    mock_client.get.side_effect = [
                        httpx.TimeoutException('timeout'),
                        httpx.TimeoutException('timeout'),
                        mock_response,
                    ]
                    mock_client.__aenter__.return_value = mock_client
                    mock_client.__aexit__.return_value = None
                    mock_client_class.return_value = mock_client

                    with patch('asyncio.sleep', new_callable=AsyncMock):
                        result = await conversation_manager._proxy_get_to_nested_server(
                            sid, '/vscode-url'
                        )

                    assert result == expected_response
                    assert mock_client.get.call_count == 3

    @pytest.mark.asyncio
    async def test_proxy_get_fails_after_max_retries(
        self, conversation_manager, mock_runtime
    ):
        """Test that proxy fails after exhausting retries."""
        sid = 'test_session_456'

        with patch.object(
            conversation_manager, '_get_runtime', new_callable=AsyncMock
        ) as mock_get_runtime:
            mock_get_runtime.return_value = mock_runtime

            with patch.object(
                conversation_manager,
                '_get_nested_url_for_runtime',
                return_value='https://nested.example.com/api/conversations/test_session_456',
            ):
                with patch('httpx.AsyncClient') as mock_client_class:
                    mock_client = AsyncMock()
                    mock_client.get.side_effect = httpx.TimeoutException('timeout')
                    mock_client.__aenter__.return_value = mock_client
                    mock_client.__aexit__.return_value = None
                    mock_client_class.return_value = mock_client

                    with patch('asyncio.sleep', new_callable=AsyncMock):
                        with pytest.raises(ValueError, match='nested_proxy_failed'):
                            await conversation_manager._proxy_get_to_nested_server(
                                sid, '/vscode-url'
                            )

                    assert mock_client.get.call_count == _NESTED_PROXY_MAX_RETRIES

    @pytest.mark.asyncio
    async def test_proxy_get_no_retry_on_client_error(
        self, conversation_manager, mock_runtime
    ):
        """Test that proxy does not retry on 4xx client errors."""
        sid = 'test_session_456'

        with patch.object(
            conversation_manager, '_get_runtime', new_callable=AsyncMock
        ) as mock_get_runtime:
            mock_get_runtime.return_value = mock_runtime

            with patch.object(
                conversation_manager,
                '_get_nested_url_for_runtime',
                return_value='https://nested.example.com/api/conversations/test_session_456',
            ):
                with patch('httpx.AsyncClient') as mock_client_class:
                    mock_response = Mock()
                    mock_response.status_code = 404

                    mock_client = AsyncMock()
                    mock_client.get.side_effect = httpx.HTTPStatusError(
                        'Not Found', request=Mock(), response=mock_response
                    )
                    mock_client.__aenter__.return_value = mock_client
                    mock_client.__aexit__.return_value = None
                    mock_client_class.return_value = mock_client

                    with pytest.raises(ValueError, match='nested_server_error'):
                        await conversation_manager._proxy_get_to_nested_server(
                            sid, '/vscode-url'
                        )

                    # Should only be called once (no retry on 4xx)
                    assert mock_client.get.call_count == 1

    @pytest.mark.asyncio
    async def test_proxy_get_retries_on_server_error(
        self, conversation_manager, mock_runtime
    ):
        """Test that proxy retries on 5xx server errors."""
        sid = 'test_session_456'
        expected_response = {'vscode_url': 'https://vscode.example.com'}

        with patch.object(
            conversation_manager, '_get_runtime', new_callable=AsyncMock
        ) as mock_get_runtime:
            mock_get_runtime.return_value = mock_runtime

            with patch.object(
                conversation_manager,
                '_get_nested_url_for_runtime',
                return_value='https://nested.example.com/api/conversations/test_session_456',
            ):
                with patch('httpx.AsyncClient') as mock_client_class:
                    # First call gets 500, second succeeds
                    mock_error_response = Mock()
                    mock_error_response.status_code = 500

                    mock_success_response = Mock()
                    mock_success_response.json.return_value = expected_response
                    mock_success_response.raise_for_status = Mock()

                    mock_client = AsyncMock()
                    mock_client.get.side_effect = [
                        httpx.HTTPStatusError(
                            'Server Error', request=Mock(), response=mock_error_response
                        ),
                        mock_success_response,
                    ]
                    mock_client.__aenter__.return_value = mock_client
                    mock_client.__aexit__.return_value = None
                    mock_client_class.return_value = mock_client

                    with patch('asyncio.sleep', new_callable=AsyncMock):
                        result = await conversation_manager._proxy_get_to_nested_server(
                            sid, '/vscode-url'
                        )

                    assert result == expected_response
                    assert mock_client.get.call_count == 2


class TestGetVscodeUrl:
    """Test suite for get_vscode_url method."""

    @pytest.fixture
    def conversation_manager(self):
        """Create a minimal SaasNestedConversationManager instance for testing."""
        mock_sio = Mock()
        mock_config = Mock()
        mock_config.max_concurrent_conversations = 5
        mock_server_config = Mock()
        mock_file_store = Mock()

        manager = SaasNestedConversationManager(
            sio=mock_sio,
            config=mock_config,
            server_config=mock_server_config,
            file_store=mock_file_store,
            event_retrieval=Mock(),
        )
        return manager

    @pytest.mark.asyncio
    async def test_get_vscode_url_calls_proxy(self, conversation_manager):
        """Test that get_vscode_url calls the proxy method correctly."""
        sid = 'test_session'
        expected_result = {'vscode_url': 'https://vscode.example.com'}

        with patch.object(
            conversation_manager,
            '_proxy_get_to_nested_server',
            new_callable=AsyncMock,
        ) as mock_proxy:
            mock_proxy.return_value = expected_result

            result = await conversation_manager.get_vscode_url(sid)

            mock_proxy.assert_called_once_with(sid, '/vscode-url')
            assert result == expected_result


class TestGetWebHosts:
    """Test suite for get_web_hosts method."""

    @pytest.fixture
    def conversation_manager(self):
        """Create a minimal SaasNestedConversationManager instance for testing."""
        mock_sio = Mock()
        mock_config = Mock()
        mock_config.max_concurrent_conversations = 5
        mock_server_config = Mock()
        mock_file_store = Mock()

        manager = SaasNestedConversationManager(
            sio=mock_sio,
            config=mock_config,
            server_config=mock_server_config,
            file_store=mock_file_store,
            event_retrieval=Mock(),
        )
        return manager

    @pytest.mark.asyncio
    async def test_get_web_hosts_calls_proxy(self, conversation_manager):
        """Test that get_web_hosts calls the proxy method correctly."""
        sid = 'test_session'
        expected_result = {'hosts': ['https://host1.example.com', 'https://host2.example.com']}

        with patch.object(
            conversation_manager,
            '_proxy_get_to_nested_server',
            new_callable=AsyncMock,
        ) as mock_proxy:
            mock_proxy.return_value = expected_result

            result = await conversation_manager.get_web_hosts(sid)

            mock_proxy.assert_called_once_with(sid, '/web-hosts')
            assert result == expected_result


class TestGetMicroagents:
    """Test suite for get_microagents method."""

    @pytest.fixture
    def conversation_manager(self):
        """Create a minimal SaasNestedConversationManager instance for testing."""
        mock_sio = Mock()
        mock_config = Mock()
        mock_config.max_concurrent_conversations = 5
        mock_server_config = Mock()
        mock_file_store = Mock()

        manager = SaasNestedConversationManager(
            sio=mock_sio,
            config=mock_config,
            server_config=mock_server_config,
            file_store=mock_file_store,
            event_retrieval=Mock(),
        )
        return manager

    @pytest.mark.asyncio
    async def test_get_microagents_calls_proxy(self, conversation_manager):
        """Test that get_microagents calls the proxy method correctly."""
        sid = 'test_session'
        expected_result = {
            'microagents': [
                {'name': 'agent1', 'type': 'repo', 'content': 'test content'}
            ]
        }

        with patch.object(
            conversation_manager,
            '_proxy_get_to_nested_server',
            new_callable=AsyncMock,
        ) as mock_proxy:
            mock_proxy.return_value = expected_result

            result = await conversation_manager.get_microagents(sid)

            mock_proxy.assert_called_once_with(sid, '/microagents')
            assert result == expected_result


class TestSendEventToConversation:
    """Test suite for send_event_to_conversation with retry logic."""

    @pytest.fixture
    def conversation_manager(self):
        """Create a minimal SaasNestedConversationManager instance for testing."""
        mock_sio = Mock()
        mock_config = Mock()
        mock_config.max_concurrent_conversations = 5
        mock_server_config = Mock()
        mock_file_store = Mock()

        manager = SaasNestedConversationManager(
            sio=mock_sio,
            config=mock_config,
            server_config=mock_server_config,
            file_store=mock_file_store,
            event_retrieval=Mock(),
        )
        return manager

    @pytest.mark.asyncio
    async def test_send_event_calls_proxy(self, conversation_manager):
        """Test that send_event_to_conversation calls the proxy method correctly."""
        sid = 'test_session'
        event_data = {'type': 'message', 'content': 'Hello'}

        with patch.object(
            conversation_manager,
            '_proxy_post_to_nested_server',
            new_callable=AsyncMock,
        ) as mock_proxy:
            mock_proxy.return_value = {'success': True}

            await conversation_manager.send_event_to_conversation(sid, event_data)

            mock_proxy.assert_called_once_with(sid, '/events', event_data)


class TestProxyPostToNestedServer:
    """Test suite for _proxy_post_to_nested_server method."""

    @pytest.fixture
    def conversation_manager(self):
        """Create a minimal SaasNestedConversationManager instance for testing."""
        mock_sio = Mock()
        mock_config = Mock()
        mock_config.max_concurrent_conversations = 5
        mock_config.sandbox = Mock()
        mock_config.sandbox.api_key = 'test_api_key'
        mock_server_config = Mock()
        mock_file_store = Mock()

        manager = SaasNestedConversationManager(
            sio=mock_sio,
            config=mock_config,
            server_config=mock_server_config,
            file_store=mock_file_store,
            event_retrieval=Mock(),
        )
        return manager

    @pytest.fixture
    def mock_runtime(self):
        """Create a mock runtime response."""
        return {
            'runtime_id': 'test_runtime_123',
            'session_id': 'test_session_456',
            'session_api_key': 'test_session_api_key_789',
            'status': 'running',
        }

    @pytest.mark.asyncio
    async def test_proxy_post_success(self, conversation_manager, mock_runtime):
        """Test successful proxy POST request."""
        sid = 'test_session_456'
        post_data = {'type': 'message', 'content': 'Hello'}
        expected_response = {'success': True}

        with patch.object(
            conversation_manager, '_get_runtime', new_callable=AsyncMock
        ) as mock_get_runtime:
            mock_get_runtime.return_value = mock_runtime

            with patch.object(
                conversation_manager,
                '_get_nested_url_for_runtime',
                return_value='https://nested.example.com/api/conversations/test_session_456',
            ):
                with patch('httpx.AsyncClient') as mock_client_class:
                    mock_response = Mock()
                    mock_response.json.return_value = expected_response
                    mock_response.raise_for_status = Mock()

                    mock_client = AsyncMock()
                    mock_client.post.return_value = mock_response
                    mock_client.__aenter__.return_value = mock_client
                    mock_client.__aexit__.return_value = None
                    mock_client_class.return_value = mock_client

                    result = await conversation_manager._proxy_post_to_nested_server(
                        sid, '/events', post_data
                    )

                    assert result == expected_response
                    mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_proxy_post_retries_on_timeout(
        self, conversation_manager, mock_runtime
    ):
        """Test that proxy POST retries on timeout errors."""
        sid = 'test_session_456'
        post_data = {'type': 'message', 'content': 'Hello'}
        expected_response = {'success': True}

        with patch.object(
            conversation_manager, '_get_runtime', new_callable=AsyncMock
        ) as mock_get_runtime:
            mock_get_runtime.return_value = mock_runtime

            with patch.object(
                conversation_manager,
                '_get_nested_url_for_runtime',
                return_value='https://nested.example.com/api/conversations/test_session_456',
            ):
                with patch('httpx.AsyncClient') as mock_client_class:
                    mock_response = Mock()
                    mock_response.json.return_value = expected_response
                    mock_response.raise_for_status = Mock()

                    mock_client = AsyncMock()
                    mock_client.post.side_effect = [
                        httpx.TimeoutException('timeout'),
                        mock_response,
                    ]
                    mock_client.__aenter__.return_value = mock_client
                    mock_client.__aexit__.return_value = None
                    mock_client_class.return_value = mock_client

                    with patch('asyncio.sleep', new_callable=AsyncMock):
                        result = await conversation_manager._proxy_post_to_nested_server(
                            sid, '/events', post_data
                        )

                    assert result == expected_response
                    assert mock_client.post.call_count == 2
