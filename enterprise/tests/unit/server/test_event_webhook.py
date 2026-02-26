"""Unit tests for event_webhook.py"""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import BackgroundTasks, HTTPException, Request, status
from server.routes.event_webhook import (
    BatchMethod,
    BatchOperation,
    _get_session_api_key,
    _get_user_id,
    _parse_conversation_id_and_subpath,
    _process_batch_operations_background,
    on_batch_write,
    on_delete,
    on_write,
)
from server.utils.conversation_callback_utils import (
    process_event,
    update_conversation_metadata,
)
from storage.stored_conversation_metadata import StoredConversationMetadata

from openhands.events.observation.agent import AgentStateChangedObservation


class TestParseConversationIdAndSubpath:
    """Test the _parse_conversation_id_and_subpath function."""

    def test_valid_path_with_metadata(self):
        """Test parsing a valid path with metadata.json."""
        path = 'sessions/conv-123/metadata.json'
        conversation_id, subpath = _parse_conversation_id_and_subpath(path)
        assert conversation_id == 'conv-123'
        assert subpath == 'metadata.json'

    def test_valid_path_with_events(self):
        """Test parsing a valid path with events."""
        path = 'sessions/conv-456/events/event-1.json'
        conversation_id, subpath = _parse_conversation_id_and_subpath(path)
        assert conversation_id == 'conv-456'
        assert subpath == 'events/event-1.json'

    def test_valid_path_with_nested_subpath(self):
        """Test parsing a valid path with nested subpath."""
        path = 'sessions/conv-789/events/subfolder/event.json'
        conversation_id, subpath = _parse_conversation_id_and_subpath(path)
        assert conversation_id == 'conv-789'
        assert subpath == 'events/subfolder/event.json'

    def test_invalid_path_missing_sessions(self):
        """Test parsing an invalid path that doesn't start with 'sessions'."""
        path = 'invalid/conv-123/metadata.json'
        with pytest.raises(HTTPException) as exc_info:
            _parse_conversation_id_and_subpath(path)
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST

    def test_invalid_path_too_short(self):
        """Test parsing an invalid path that's too short."""
        path = 'sessions'
        with pytest.raises(HTTPException) as exc_info:
            _parse_conversation_id_and_subpath(path)
        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST

    def test_invalid_path_empty_conversation_id(self):
        """Test parsing a path with empty conversation ID."""
        path = 'sessions//metadata.json'
        conversation_id, subpath = _parse_conversation_id_and_subpath(path)
        assert conversation_id == ''
        assert subpath == 'metadata.json'


class TestGetUserId:
    """Test the _get_user_id function."""

    def test_get_user_id_success(self, session_maker_with_minimal_fixtures):
        """Test successfully getting user ID."""
        with patch(
            'server.routes.event_webhook.session_maker',
            session_maker_with_minimal_fixtures,
        ):
            user_id = _get_user_id('mock-conversation-id')
            assert user_id == '5594c7b6-f959-4b81-92e9-b09c206f5081'

    def test_get_user_id_conversation_not_found(self, session_maker):
        """Test getting user ID when conversation doesn't exist."""
        with patch('server.routes.event_webhook.session_maker', session_maker):
            with pytest.raises(AttributeError):
                _get_user_id('nonexistent-conversation-id')


class TestGetSessionApiKey:
    """Test the _get_session_api_key function."""

    @pytest.mark.asyncio
    async def test_get_session_api_key_success(self):
        """Test successfully getting session API key."""
        mock_agent_loop_info = MagicMock()
        mock_agent_loop_info.session_api_key = 'test-api-key'

        with patch('server.routes.event_webhook.conversation_manager') as mock_manager:
            mock_manager.get_agent_loop_info = AsyncMock(
                return_value=[mock_agent_loop_info]
            )

            api_key = await _get_session_api_key(
                '5594c7b6-f959-4b81-92e9-b09c206f5081', 'conv-456'
            )
            assert api_key == 'test-api-key'
            mock_manager.get_agent_loop_info.assert_called_once_with(
                '5594c7b6-f959-4b81-92e9-b09c206f5081', filter_to_sids={'conv-456'}
            )

    @pytest.mark.asyncio
    async def test_get_session_api_key_no_results(self):
        """Test getting session API key when no agent loop info is found."""
        with patch('server.routes.event_webhook.conversation_manager') as mock_manager:
            mock_manager.get_agent_loop_info = AsyncMock(return_value=[])

            with pytest.raises(IndexError):
                await _get_session_api_key(
                    '5594c7b6-f959-4b81-92e9-b09c206f5081', 'conv-456'
                )


class TestProcessEvent:
    """Test the process_event function."""

    @pytest.mark.asyncio
    async def test_process_event_regular_event(
        self, session_maker_with_minimal_fixtures
    ):
        """Test processing a regular event."""
        content = {'type': 'action', 'action': 'run', 'args': {'command': 'ls'}}

        with patch(
            'server.utils.conversation_callback_utils.file_store'
        ) as mock_file_store, patch(
            'server.utils.conversation_callback_utils.event_from_dict'
        ) as mock_event_from_dict, patch(
            'server.utils.conversation_callback_utils.session_maker',
            session_maker_with_minimal_fixtures,
        ):
            mock_event = MagicMock()
            mock_event_from_dict.return_value = mock_event

            await process_event(
                '5594c7b6-f959-4b81-92e9-b09c206f5081',
                'conv-456',
                'events/event-1.json',
                content,
            )

            mock_file_store.write.assert_called_once_with(
                'users/5594c7b6-f959-4b81-92e9-b09c206f5081/conversations/conv-456/events/event-1.json',
                json.dumps(content),
            )
            mock_event_from_dict.assert_called_once_with(content)

    @pytest.mark.asyncio
    async def test_process_event_agent_state_changed(
        self, session_maker_with_minimal_fixtures
    ):
        """Test processing an AgentStateChangedObservation event."""
        content = {'type': 'observation', 'observation': 'agent_state_changed'}

        with patch(
            'server.utils.conversation_callback_utils.file_store'
        ) as mock_file_store, patch(
            'server.utils.conversation_callback_utils.event_from_dict'
        ) as mock_event_from_dict, patch(
            'server.utils.conversation_callback_utils.session_maker',
            session_maker_with_minimal_fixtures,
        ), patch(
            'server.utils.conversation_callback_utils.invoke_conversation_callbacks'
        ) as mock_invoke_callbacks, patch(
            'server.utils.conversation_callback_utils.update_active_working_seconds'
        ) as mock_update_working_seconds, patch(
            'server.utils.conversation_callback_utils.EventStore'
        ) as mock_event_store_class:
            mock_event = MagicMock(spec=AgentStateChangedObservation)
            mock_event.agent_state = (
                'stopped'  # Set a non-RUNNING state to trigger the update
            )
            mock_event_from_dict.return_value = mock_event

            await process_event(
                '5594c7b6-f959-4b81-92e9-b09c206f5081',
                'conv-456',
                'events/event-1.json',
                content,
            )

            mock_file_store.write.assert_called_once()
            mock_event_from_dict.assert_called_once_with(content)
            mock_invoke_callbacks.assert_called_once_with('conv-456', mock_event)
            mock_update_working_seconds.assert_called_once()
            mock_event_store_class.assert_called_once_with(
                'conv-456', mock_file_store, '5594c7b6-f959-4b81-92e9-b09c206f5081'
            )

    @pytest.mark.asyncio
    async def test_process_event_agent_state_changed_running(
        self, session_maker_with_minimal_fixtures
    ):
        """Test processing an AgentStateChangedObservation event with RUNNING state."""
        content = {'type': 'observation', 'observation': 'agent_state_changed'}

        with patch(
            'server.utils.conversation_callback_utils.file_store'
        ) as mock_file_store, patch(
            'server.utils.conversation_callback_utils.event_from_dict'
        ) as mock_event_from_dict, patch(
            'server.utils.conversation_callback_utils.session_maker',
            session_maker_with_minimal_fixtures,
        ), patch(
            'server.utils.conversation_callback_utils.invoke_conversation_callbacks'
        ) as mock_invoke_callbacks, patch(
            'server.utils.conversation_callback_utils.update_active_working_seconds'
        ) as mock_update_working_seconds, patch(
            'server.utils.conversation_callback_utils.EventStore'
        ) as mock_event_store_class:
            mock_event = MagicMock(spec=AgentStateChangedObservation)
            mock_event.agent_state = 'running'  # Set RUNNING state to skip the update
            mock_event_from_dict.return_value = mock_event

            await process_event(
                '5594c7b6-f959-4b81-92e9-b09c206f5081',
                'conv-456',
                'events/event-1.json',
                content,
            )

            mock_file_store.write.assert_called_once()
            mock_event_from_dict.assert_called_once_with(content)
            mock_invoke_callbacks.assert_called_once_with('conv-456', mock_event)
            # update_active_working_seconds should NOT be called when agent is RUNNING
            mock_update_working_seconds.assert_not_called()
            mock_event_store_class.assert_not_called()


class TestUpdateConversationMetadata:
    """Test the _update_conversation_metadata function."""

    def test_update_conversation_metadata_all_fields(
        self, session_maker_with_minimal_fixtures
    ):
        """Test updating conversation metadata with all fields."""
        content = {
            'accumulated_cost': 10.50,
            'prompt_tokens': 1000,
            'completion_tokens': 500,
            'total_tokens': 1500,
        }

        with patch(
            'server.utils.conversation_callback_utils.session_maker',
            session_maker_with_minimal_fixtures,
        ):
            update_conversation_metadata('mock-conversation-id', content)

            # Verify the conversation was updated
            with session_maker_with_minimal_fixtures() as session:
                conversation = (
                    session.query(StoredConversationMetadata)
                    .filter(
                        StoredConversationMetadata.conversation_id
                        == 'mock-conversation-id'
                    )
                    .first()
                )
                assert conversation.accumulated_cost == 10.50
                assert conversation.prompt_tokens == 1000
                assert conversation.completion_tokens == 500
                assert conversation.total_tokens == 1500
                assert isinstance(conversation.last_updated_at, datetime)

    def test_update_conversation_metadata_partial_fields(
        self, session_maker_with_minimal_fixtures
    ):
        """Test updating conversation metadata with only some fields."""
        content = {'accumulated_cost': 15.75, 'prompt_tokens': 2000}

        with patch(
            'server.utils.conversation_callback_utils.session_maker',
            session_maker_with_minimal_fixtures,
        ):
            update_conversation_metadata('mock-conversation-id', content)

            # Verify only specified fields were updated, others remain unchanged
            with session_maker_with_minimal_fixtures() as session:
                conversation = (
                    session.query(StoredConversationMetadata)
                    .filter(
                        StoredConversationMetadata.conversation_id
                        == 'mock-conversation-id'
                    )
                    .first()
                )
                assert conversation.accumulated_cost == 15.75
                assert conversation.prompt_tokens == 2000
                # These should remain as original values from fixtures
                assert conversation.completion_tokens == 250
                assert conversation.total_tokens == 750

    def test_update_conversation_metadata_empty_content(
        self, session_maker_with_minimal_fixtures
    ):
        """Test updating conversation metadata with empty content."""
        content: dict[str, float] = {}

        with patch(
            'server.utils.conversation_callback_utils.session_maker',
            session_maker_with_minimal_fixtures,
        ):
            update_conversation_metadata('mock-conversation-id', content)

            # Verify only last_updated_at was changed
            with session_maker_with_minimal_fixtures() as session:
                conversation = (
                    session.query(StoredConversationMetadata)
                    .filter(
                        StoredConversationMetadata.conversation_id
                        == 'mock-conversation-id'
                    )
                    .first()
                )
                # Original values should remain unchanged
                assert conversation.accumulated_cost == 5.25
                assert conversation.prompt_tokens == 500
                assert conversation.completion_tokens == 250
                assert conversation.total_tokens == 750
                assert isinstance(conversation.last_updated_at, datetime)


class TestOnDelete:
    """Test the on_delete endpoint."""

    @pytest.mark.asyncio
    async def test_on_delete_returns_ok(self):
        """Test that on_delete always returns 200 OK."""
        result = await on_delete('any/path', 'any-api-key')
        assert result.status_code == status.HTTP_200_OK


class TestOnWrite:
    """Test the on_write endpoint."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock request object."""
        request = MagicMock(spec=Request)
        request.json = AsyncMock(return_value={'test': 'data'})
        return request

    @pytest.mark.asyncio
    async def test_on_write_metadata_success(
        self, mock_request, session_maker_with_minimal_fixtures
    ):
        """Test successful metadata update."""
        content = {'accumulated_cost': 20.0}
        mock_request.json.return_value = content

        with patch(
            'server.routes.event_webhook.session_maker',
            session_maker_with_minimal_fixtures,
        ), patch(
            'server.utils.conversation_callback_utils.session_maker',
            session_maker_with_minimal_fixtures,
        ), patch(
            'server.routes.event_webhook._get_session_api_key'
        ) as mock_get_api_key:
            mock_get_api_key.return_value = 'correct-api-key'

            result = await on_write(
                'sessions/mock-conversation-id/metadata.json',
                mock_request,
                'correct-api-key',
            )

            assert result.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_on_write_events_success(
        self, mock_request, session_maker_with_minimal_fixtures
    ):
        """Test successful event processing."""
        content = {'type': 'action', 'action': 'run'}
        mock_request.json.return_value = content

        with patch(
            'server.routes.event_webhook.session_maker',
            session_maker_with_minimal_fixtures,
        ), patch(
            'server.routes.event_webhook._get_session_api_key'
        ) as mock_get_api_key, patch(
            'server.utils.conversation_callback_utils.file_store'
        ) as mock_file_store, patch(
            'server.utils.conversation_callback_utils.event_from_dict'
        ) as mock_event_from_dict:
            mock_get_api_key.return_value = 'correct-api-key'
            mock_event_from_dict.return_value = MagicMock()

            result = await on_write(
                'sessions/mock-conversation-id/events/event-1.json',
                mock_request,
                'correct-api-key',
            )

            assert result.status_code == status.HTTP_200_OK
            mock_file_store.write.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_write_invalid_api_key(
        self, mock_request, session_maker_with_minimal_fixtures
    ):
        """Test request with invalid API key."""
        with patch(
            'server.routes.event_webhook.session_maker',
            session_maker_with_minimal_fixtures,
        ), patch(
            'server.routes.event_webhook._get_session_api_key'
        ) as mock_get_api_key:
            mock_get_api_key.return_value = 'correct-api-key'

            result = await on_write(
                'sessions/mock-conversation-id/metadata.json',
                mock_request,
                'wrong-api-key',
            )

            assert result.status_code == status.HTTP_403_FORBIDDEN

    @pytest.mark.asyncio
    async def test_on_write_invalid_path(self, mock_request):
        """Test request with invalid path."""
        with pytest.raises(HTTPException) as excinfo:
            await on_write('invalid/path/format', mock_request, 'any-api-key')
        assert excinfo.value.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.asyncio
    async def test_on_write_unsupported_subpath(
        self, mock_request, session_maker_with_minimal_fixtures
    ):
        """Test request with unsupported subpath."""
        with patch(
            'server.routes.event_webhook.session_maker',
            session_maker_with_minimal_fixtures,
        ), patch(
            'server.routes.event_webhook._get_session_api_key'
        ) as mock_get_api_key:
            mock_get_api_key.return_value = 'correct-api-key'

            result = await on_write(
                'sessions/mock-conversation-id/unsupported.json',
                mock_request,
                'correct-api-key',
            )

            assert result.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.asyncio
    async def test_on_write_invalid_json(self, session_maker_with_minimal_fixtures):
        """Test request with invalid JSON."""
        mock_request = MagicMock(spec=Request)
        mock_request.json = AsyncMock(side_effect=ValueError('Invalid JSON'))

        with patch(
            'server.routes.event_webhook.session_maker',
            session_maker_with_minimal_fixtures,
        ), patch(
            'server.routes.event_webhook._get_session_api_key'
        ) as mock_get_api_key:
            mock_get_api_key.return_value = 'correct-api-key'

            result = await on_write(
                'sessions/mock-conversation-id/metadata.json',
                mock_request,
                'correct-api-key',
            )

            assert result.status_code == status.HTTP_400_BAD_REQUEST


class TestBatchOperation:
    """Test the BatchOperation model."""

    def test_batch_operation_get_content_utf8(self):
        """Test getting content as UTF-8 bytes."""
        op = BatchOperation(
            method=BatchMethod.POST,
            path='sessions/test/metadata.json',
            content='{"test": "data"}',
            encoding=None,
        )
        content = op.get_content()
        assert content == b'{"test": "data"}'

    def test_batch_operation_get_content_base64(self):
        """Test getting content from base64 encoding."""
        import base64

        original_content = '{"test": "data"}'
        encoded_content = base64.b64encode(original_content.encode('utf-8')).decode(
            'ascii'
        )

        op = BatchOperation(
            method=BatchMethod.POST,
            path='sessions/test/metadata.json',
            content=encoded_content,
            encoding='base64',
        )
        content = op.get_content()
        assert content == original_content.encode('utf-8')

    def test_batch_operation_get_content_json(self):
        """Test getting content as JSON."""
        op = BatchOperation(
            method=BatchMethod.POST,
            path='sessions/test/metadata.json',
            content='{"test": "data", "number": 42}',
            encoding=None,
        )
        json_content = op.get_content_json()
        assert json_content == {'test': 'data', 'number': 42}

    def test_batch_operation_get_content_empty_raises_error(self):
        """Test that empty content raises ValueError."""
        op = BatchOperation(
            method=BatchMethod.POST,
            path='sessions/test/metadata.json',
            content=None,
            encoding=None,
        )
        with pytest.raises(ValueError, match='empty_content_in_batch'):
            op.get_content()


class TestOnBatchWrite:
    """Test the on_batch_write endpoint."""

    @pytest.mark.asyncio
    async def test_on_batch_write_returns_accepted(self):
        """Test that on_batch_write returns 202 ACCEPTED and queues background task."""
        batch_ops = [
            BatchOperation(
                method=BatchMethod.POST,
                path='sessions/test-conv/metadata.json',
                content='{"test": "data"}',
            )
        ]

        mock_background_tasks = MagicMock(spec=BackgroundTasks)

        result = await on_batch_write(
            batch_ops=batch_ops,
            background_tasks=mock_background_tasks,
            x_session_api_key='test-api-key',
        )

        # Should return 202 ACCEPTED immediately
        assert result.status_code == status.HTTP_202_ACCEPTED

        # Should have queued the background task
        mock_background_tasks.add_task.assert_called_once_with(
            _process_batch_operations_background,
            batch_ops,
            'test-api-key',
        )


class TestProcessBatchOperationsBackground:
    """Test the _process_batch_operations_background function."""

    @pytest.mark.asyncio
    async def test_process_batch_operations_metadata_success(
        self, session_maker_with_minimal_fixtures
    ):
        """Test successful processing of metadata batch operation."""
        batch_ops = [
            BatchOperation(
                method=BatchMethod.POST,
                path='sessions/mock-conversation-id/metadata.json',
                content='{"accumulated_cost": 15.0}',
            )
        ]

        with patch(
            'server.routes.event_webhook.session_maker',
            session_maker_with_minimal_fixtures,
        ), patch(
            'server.routes.event_webhook._get_session_api_key'
        ) as mock_get_api_key, patch(
            'server.utils.conversation_callback_utils.session_maker',
            session_maker_with_minimal_fixtures,
        ):
            mock_get_api_key.return_value = 'correct-api-key'

            # Should not raise any exceptions
            await _process_batch_operations_background(batch_ops, 'correct-api-key')

            # Verify the conversation metadata was updated
            with session_maker_with_minimal_fixtures() as session:
                conversation = (
                    session.query(StoredConversationMetadata)
                    .filter(
                        StoredConversationMetadata.conversation_id
                        == 'mock-conversation-id'
                    )
                    .first()
                )
                assert conversation.accumulated_cost == 15.0

    @pytest.mark.asyncio
    async def test_process_batch_operations_events_success(
        self, session_maker_with_minimal_fixtures
    ):
        """Test successful processing of events batch operation."""
        batch_ops = [
            BatchOperation(
                method=BatchMethod.POST,
                path='sessions/mock-conversation-id/events/event-1.json',
                content='{"type": "action", "action": "run"}',
            )
        ]

        with patch(
            'server.routes.event_webhook.session_maker',
            session_maker_with_minimal_fixtures,
        ), patch(
            'server.routes.event_webhook._get_session_api_key'
        ) as mock_get_api_key, patch(
            'server.utils.conversation_callback_utils.file_store'
        ) as mock_file_store, patch(
            'server.utils.conversation_callback_utils.event_from_dict'
        ) as mock_event_from_dict:
            mock_get_api_key.return_value = 'correct-api-key'
            mock_event_from_dict.return_value = MagicMock()

            await _process_batch_operations_background(batch_ops, 'correct-api-key')

            # Verify file_store.write was called
            mock_file_store.write.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_batch_operations_auth_failure_continues(
        self, session_maker_with_minimal_fixtures
    ):
        """Test that auth failure for one operation doesn't stop others."""
        batch_ops = [
            BatchOperation(
                method=BatchMethod.POST,
                path='sessions/conv-1/metadata.json',
                content='{"test": "data1"}',
            ),
            BatchOperation(
                method=BatchMethod.POST,
                path='sessions/conv-2/metadata.json',
                content='{"test": "data2"}',
            ),
        ]

        with patch(
            'server.routes.event_webhook.session_maker',
            session_maker_with_minimal_fixtures,
        ), patch(
            'server.routes.event_webhook._get_session_api_key'
        ) as mock_get_api_key, patch(
            'server.utils.conversation_callback_utils.session_maker',
            session_maker_with_minimal_fixtures,
        ):
            # First call succeeds, second fails
            mock_get_api_key.side_effect = ['correct-api-key', 'wrong-api-key']

            # Should not raise exceptions, just log errors
            await _process_batch_operations_background(batch_ops, 'correct-api-key')

    @pytest.mark.asyncio
    async def test_process_batch_operations_invalid_method_skipped(
        self, session_maker_with_minimal_fixtures
    ):
        """Test that invalid methods are skipped with logging."""
        batch_ops = [
            BatchOperation(
                method=BatchMethod.DELETE,  # Not supported
                path='sessions/mock-conversation-id/metadata.json',
                content='{"test": "data"}',
            )
        ]

        with patch('server.routes.event_webhook.logger') as mock_logger:
            await _process_batch_operations_background(batch_ops, 'test-api-key')

            # Should log the invalid operation
            mock_logger.info.assert_called_once_with(
                'invalid_operation_in_batch_webhook',
                extra={
                    'method': 'BatchMethod.DELETE',
                    'path': 'sessions/mock-conversation-id/metadata.json',
                },
            )

    @pytest.mark.asyncio
    async def test_process_batch_operations_exception_handling(self):
        """Test that exceptions in individual operations are handled gracefully."""
        batch_ops = [
            BatchOperation(
                method=BatchMethod.POST,
                path='invalid-path',  # This will cause an exception
                content='{"test": "data"}',
            )
        ]

        with patch('server.routes.event_webhook.logger') as mock_logger:
            # Should not raise exceptions
            await _process_batch_operations_background(batch_ops, 'test-api-key')

            # Should log the error with exception type and message in the log message
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args
            log_message = call_args[0][0]
            assert log_message.startswith('error_processing_batch_operation:')
            assert call_args[1]['extra']['path'] == 'invalid-path'
            assert call_args[1]['extra']['method'] == 'BatchMethod.POST'
            assert call_args[1]['exc_info'] is True
