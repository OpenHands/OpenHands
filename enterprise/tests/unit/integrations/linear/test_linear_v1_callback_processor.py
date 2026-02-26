"""Tests for LinearV1CallbackProcessor."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from openhands.app_server.event_callback.event_callback_result_models import (
    EventCallbackResultStatus,
)

from integrations.linear.linear_v1_callback_processor import (
    LinearV1CallbackProcessor,
)


@pytest.fixture
def processor():
    return LinearV1CallbackProcessor(
        linear_view_data={
            'issue_id': 'abc-123',
            'issue_key': 'LIN-123',
            'workspace_name': 'my-workspace',
        },
        should_request_summary=True,
    )


@pytest.fixture
def conversation_id():
    return uuid4()


@pytest.fixture
def mock_callback():
    cb = MagicMock()
    cb.id = uuid4()
    return cb


@pytest.fixture
def mock_finished_event():
    event = MagicMock()
    event.__class__.__name__ = 'ConversationStateUpdateEvent'
    event.key = 'execution_status'
    event.value = 'finished'
    event.id = uuid4()
    return event


class TestLinearV1CallbackProcessorInit:
    """Test processor initialization."""

    def test_default_fields(self):
        proc = LinearV1CallbackProcessor()
        assert proc.linear_view_data == {}
        assert proc.should_request_summary is True

    def test_custom_fields(self, processor):
        assert processor.linear_view_data['issue_id'] == 'abc-123'
        assert processor.linear_view_data['issue_key'] == 'LIN-123'
        assert processor.linear_view_data['workspace_name'] == 'my-workspace'
        assert processor.should_request_summary is True


class TestCallFiltering:
    """Test event filtering logic."""

    @pytest.mark.asyncio
    async def test_ignores_non_conversation_state_event(
        self, processor, conversation_id, mock_callback
    ):
        """Non-ConversationStateUpdateEvent should be ignored."""
        event = MagicMock()
        with patch(
            'integrations.linear.linear_v1_callback_processor.isinstance',
            return_value=False,
        ):
            result = await processor(conversation_id, mock_callback, event)
        assert result is None

    @pytest.mark.asyncio
    async def test_ignores_non_finished_status(
        self, processor, conversation_id, mock_callback
    ):
        """Events with execution_status != 'finished' should be ignored."""
        event = MagicMock()
        event.key = 'execution_status'
        event.value = 'running'

        with patch(
            'integrations.linear.linear_v1_callback_processor.isinstance',
            return_value=True,
        ):
            result = await processor(conversation_id, mock_callback, event)
        assert result is None

    @pytest.mark.asyncio
    async def test_ignores_non_execution_status_key(
        self, processor, conversation_id, mock_callback
    ):
        """Events with a different key should be ignored."""
        event = MagicMock()
        event.key = 'some_other_key'
        event.value = 'finished'

        with patch(
            'integrations.linear.linear_v1_callback_processor.isinstance',
            return_value=True,
        ):
            result = await processor(conversation_id, mock_callback, event)
        assert result is None

    @pytest.mark.asyncio
    async def test_skips_when_should_request_summary_false(
        self, conversation_id, mock_callback
    ):
        """When should_request_summary is False, return None."""
        proc = LinearV1CallbackProcessor(
            linear_view_data={
                'issue_id': 'abc-123',
                'issue_key': 'LIN-123',
                'workspace_name': 'my-workspace',
            },
            should_request_summary=False,
        )
        event = MagicMock()
        event.key = 'execution_status'
        event.value = 'finished'

        with patch(
            'integrations.linear.linear_v1_callback_processor.isinstance',
            return_value=True,
        ):
            result = await proc(conversation_id, mock_callback, event)
        assert result is None


class TestSummaryFlow:
    """Test the summary request and posting flow."""

    @pytest.mark.asyncio
    async def test_successful_summary(
        self, processor, conversation_id, mock_callback
    ):
        """Happy path: summary is requested and posted to Linear."""
        event = MagicMock()
        event.key = 'execution_status'
        event.value = 'finished'
        event.id = uuid4()

        with (
            patch(
                'integrations.linear.linear_v1_callback_processor.isinstance',
                return_value=True,
            ),
            patch.object(
                processor,
                '_request_summary',
                new_callable=AsyncMock,
                return_value='Summary text',
            ),
            patch.object(
                processor,
                '_post_summary_to_linear',
                new_callable=AsyncMock,
            ) as mock_post,
        ):
            result = await processor(conversation_id, mock_callback, event)

        assert result is not None
        assert result.status == EventCallbackResultStatus.SUCCESS
        assert result.detail == 'Summary text'
        mock_post.assert_called_once_with('Summary text')
        assert processor.should_request_summary is False

    @pytest.mark.asyncio
    async def test_summary_sets_should_request_to_false(
        self, processor, conversation_id, mock_callback
    ):
        """After processing, should_request_summary should be False."""
        event = MagicMock()
        event.key = 'execution_status'
        event.value = 'finished'
        event.id = uuid4()

        with (
            patch(
                'integrations.linear.linear_v1_callback_processor.isinstance',
                return_value=True,
            ),
            patch.object(
                processor,
                '_request_summary',
                new_callable=AsyncMock,
                return_value='Summary',
            ),
            patch.object(
                processor,
                '_post_summary_to_linear',
                new_callable=AsyncMock,
            ),
        ):
            await processor(conversation_id, mock_callback, event)

        assert processor.should_request_summary is False

    @pytest.mark.asyncio
    async def test_error_returns_error_result(
        self, processor, conversation_id, mock_callback
    ):
        """When summary request fails, return error result and attempt to post error."""
        event = MagicMock()
        event.key = 'execution_status'
        event.value = 'finished'
        event.id = uuid4()

        with (
            patch(
                'integrations.linear.linear_v1_callback_processor.isinstance',
                return_value=True,
            ),
            patch.object(
                processor,
                '_request_summary',
                new_callable=AsyncMock,
                side_effect=RuntimeError('Test error'),
            ),
            patch.object(
                processor,
                '_post_summary_to_linear',
                new_callable=AsyncMock,
            ) as mock_post,
        ):
            result = await processor(conversation_id, mock_callback, event)

        assert result is not None
        assert result.status == EventCallbackResultStatus.ERROR
        assert 'Test error' in result.detail
        # Should have attempted to post error message to Linear
        mock_post.assert_called_once()
        assert 'error' in mock_post.call_args[0][0].lower()


class TestPostSummaryToLinear:
    """Test the _post_summary_to_linear method."""

    @pytest.mark.asyncio
    async def test_missing_issue_id_raises(self):
        """Missing issue_id should raise RuntimeError."""
        proc = LinearV1CallbackProcessor(
            linear_view_data={
                'workspace_name': 'my-workspace',
            },
        )
        with pytest.raises(RuntimeError, match='Missing required Linear view data'):
            await proc._post_summary_to_linear('summary')

    @pytest.mark.asyncio
    async def test_missing_workspace_name_raises(self):
        """Missing workspace_name should raise RuntimeError."""
        proc = LinearV1CallbackProcessor(
            linear_view_data={
                'issue_id': 'abc-123',
            },
        )
        with pytest.raises(RuntimeError, match='Missing required Linear view data'):
            await proc._post_summary_to_linear('summary')
