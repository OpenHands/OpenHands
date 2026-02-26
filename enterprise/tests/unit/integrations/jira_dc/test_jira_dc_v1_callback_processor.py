"""Tests for JiraDcV1CallbackProcessor."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from integrations.jira_dc.jira_dc_v1_callback_processor import (
    JiraDcV1CallbackProcessor,
)

from openhands.app_server.event_callback.event_callback_result_models import (
    EventCallbackResultStatus,
)


@pytest.fixture
def processor():
    return JiraDcV1CallbackProcessor(
        jira_dc_view_data={
            'issue_key': 'TEST-123',
            'workspace_name': 'jira.company.com',
            'base_api_url': 'https://jira.company.com',
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


class TestJiraDcV1CallbackProcessorInit:
    """Test processor initialization."""

    def test_default_fields(self):
        proc = JiraDcV1CallbackProcessor()
        assert proc.jira_dc_view_data == {}
        assert proc.should_request_summary is True

    def test_custom_fields(self, processor):
        assert processor.jira_dc_view_data['issue_key'] == 'TEST-123'
        assert processor.jira_dc_view_data['workspace_name'] == 'jira.company.com'
        assert processor.should_request_summary is True


class TestCallFiltering:
    """Test event filtering logic."""

    @pytest.mark.asyncio
    async def test_ignores_non_conversation_state_event(
        self, processor, conversation_id, mock_callback
    ):
        """Non-ConversationStateUpdateEvent should be ignored."""
        event = MagicMock()
        # Make isinstance check fail
        with patch(
            'integrations.jira_dc.jira_dc_v1_callback_processor.isinstance',
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
            'integrations.jira_dc.jira_dc_v1_callback_processor.isinstance',
            return_value=True,
        ):
            result = await processor(conversation_id, mock_callback, event)
        assert result is None

    @pytest.mark.asyncio
    async def test_ignores_non_execution_status_key(
        self, processor, conversation_id, mock_callback
    ):
        """Events with key != 'execution_status' should be ignored."""
        event = MagicMock()
        event.key = 'some_other_key'
        event.value = 'finished'

        with patch(
            'integrations.jira_dc.jira_dc_v1_callback_processor.isinstance',
            return_value=True,
        ):
            result = await processor(conversation_id, mock_callback, event)
        assert result is None


class TestShouldRequestSummary:
    """Test the should_request_summary flag."""

    @pytest.mark.asyncio
    async def test_skips_when_flag_false(
        self, processor, conversation_id, mock_callback
    ):
        """Should skip processing when should_request_summary is False."""
        processor.should_request_summary = False
        event = MagicMock()
        event.key = 'execution_status'
        event.value = 'finished'

        with patch(
            'integrations.jira_dc.jira_dc_v1_callback_processor.isinstance',
            return_value=True,
        ):
            result = await processor(conversation_id, mock_callback, event)
        assert result is None

    @pytest.mark.asyncio
    async def test_flag_set_false_after_processing(
        self, processor, conversation_id, mock_callback
    ):
        """Flag should be set to False after first processing attempt."""
        event = MagicMock()
        event.key = 'execution_status'
        event.value = 'finished'
        event.id = uuid4()

        with (
            patch(
                'integrations.jira_dc.jira_dc_v1_callback_processor.isinstance',
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
                '_post_summary_to_jira_dc',
                new_callable=AsyncMock,
            ),
        ):
            await processor(conversation_id, mock_callback, event)

        assert processor.should_request_summary is False


class TestSuccessfulProcessing:
    """Test successful summary request and posting."""

    @pytest.mark.asyncio
    async def test_success_returns_result(
        self, processor, conversation_id, mock_callback
    ):
        """Successful processing should return SUCCESS result."""
        event = MagicMock()
        event.key = 'execution_status'
        event.value = 'finished'
        event.id = uuid4()

        with (
            patch(
                'integrations.jira_dc.jira_dc_v1_callback_processor.isinstance',
                return_value=True,
            ),
            patch.object(
                processor,
                '_request_summary',
                new_callable=AsyncMock,
                return_value='**Summary** text',
            ),
            patch.object(
                processor,
                '_post_summary_to_jira_dc',
                new_callable=AsyncMock,
            ) as mock_post,
            patch(
                'integrations.jira_dc.jira_dc_v1_callback_processor.markdown_to_jira_markup',
                return_value='*Summary* text',
            ),
        ):
            result = await processor(conversation_id, mock_callback, event)

        assert result is not None
        assert result.status == EventCallbackResultStatus.SUCCESS
        assert result.conversation_id == conversation_id
        assert result.detail == '*Summary* text'
        mock_post.assert_called_once_with('*Summary* text')

    @pytest.mark.asyncio
    async def test_markdown_converted_to_jira_markup(
        self, processor, conversation_id, mock_callback
    ):
        """Markdown should be converted to Jira markup before posting."""
        event = MagicMock()
        event.key = 'execution_status'
        event.value = 'finished'
        event.id = uuid4()

        with (
            patch(
                'integrations.jira_dc.jira_dc_v1_callback_processor.isinstance',
                return_value=True,
            ),
            patch.object(
                processor,
                '_request_summary',
                new_callable=AsyncMock,
                return_value='**bold** text',
            ),
            patch.object(
                processor,
                '_post_summary_to_jira_dc',
                new_callable=AsyncMock,
            ),
            patch(
                'integrations.jira_dc.jira_dc_v1_callback_processor.markdown_to_jira_markup',
            ) as mock_convert,
        ):
            mock_convert.return_value = '*bold* text'
            await processor(conversation_id, mock_callback, event)

        mock_convert.assert_called_once_with('**bold** text')


class TestErrorHandling:
    """Test error handling in callback processing."""

    @pytest.mark.asyncio
    async def test_error_returns_error_result(
        self, processor, conversation_id, mock_callback
    ):
        """Errors should return ERROR result."""
        event = MagicMock()
        event.key = 'execution_status'
        event.value = 'finished'
        event.id = uuid4()

        with (
            patch(
                'integrations.jira_dc.jira_dc_v1_callback_processor.isinstance',
                return_value=True,
            ),
            patch.object(
                processor,
                '_request_summary',
                new_callable=AsyncMock,
                side_effect=Exception('Summary failed'),
            ),
            patch.object(
                processor,
                '_post_summary_to_jira_dc',
                new_callable=AsyncMock,
            ),
        ):
            result = await processor(conversation_id, mock_callback, event)

        assert result is not None
        assert result.status == EventCallbackResultStatus.ERROR
        assert 'Summary failed' in result.detail

    @pytest.mark.asyncio
    async def test_error_tries_to_post_error_message(
        self, processor, conversation_id, mock_callback
    ):
        """On error, should attempt to post error message to Jira DC."""
        event = MagicMock()
        event.key = 'execution_status'
        event.value = 'finished'
        event.id = uuid4()

        with (
            patch(
                'integrations.jira_dc.jira_dc_v1_callback_processor.isinstance',
                return_value=True,
            ),
            patch.object(
                processor,
                '_request_summary',
                new_callable=AsyncMock,
                side_effect=Exception('Summary failed'),
            ),
            patch.object(
                processor,
                '_post_summary_to_jira_dc',
                new_callable=AsyncMock,
            ) as mock_post,
        ):
            await processor(conversation_id, mock_callback, event)

        # Should have tried to post error message
        mock_post.assert_called_once()
        call_arg = mock_post.call_args[0][0]
        assert 'error' in call_arg.lower() or 'Error' in call_arg

    @pytest.mark.asyncio
    async def test_error_posting_error_message_is_handled(
        self, processor, conversation_id, mock_callback
    ):
        """If posting error message also fails, should still return result."""
        event = MagicMock()
        event.key = 'execution_status'
        event.value = 'finished'
        event.id = uuid4()

        with (
            patch(
                'integrations.jira_dc.jira_dc_v1_callback_processor.isinstance',
                return_value=True,
            ),
            patch.object(
                processor,
                '_request_summary',
                new_callable=AsyncMock,
                side_effect=Exception('Summary failed'),
            ),
            patch.object(
                processor,
                '_post_summary_to_jira_dc',
                new_callable=AsyncMock,
                side_effect=Exception('Post also failed'),
            ),
        ):
            result = await processor(conversation_id, mock_callback, event)

        assert result is not None
        assert result.status == EventCallbackResultStatus.ERROR


class TestPostSummaryToJiraDc:
    """Test _post_summary_to_jira_dc helper."""

    @pytest.mark.asyncio
    async def test_missing_view_data_raises(self):
        """Should raise if required view data is missing."""
        proc = JiraDcV1CallbackProcessor(
            jira_dc_view_data={},
        )
        with pytest.raises(RuntimeError, match='Missing required Jira DC view data'):
            await proc._post_summary_to_jira_dc('test summary')

    @pytest.mark.asyncio
    async def test_workspace_not_found_raises(self, processor):
        """Should raise if workspace is not found."""
        with patch(
            'integrations.jira_dc.jira_dc_v1_callback_processor.JiraDcIntegrationStore'
        ) as mock_store_cls:
            mock_store = MagicMock()
            mock_store.get_workspace_by_name = AsyncMock(return_value=None)
            mock_store_cls.get_instance.return_value = mock_store

            with pytest.raises(RuntimeError, match='not found'):
                await processor._post_summary_to_jira_dc('test summary')

    @pytest.mark.asyncio
    async def test_inactive_workspace_raises(self, processor):
        """Should raise if workspace is inactive."""
        mock_workspace = MagicMock(status='inactive')

        with patch(
            'integrations.jira_dc.jira_dc_v1_callback_processor.JiraDcIntegrationStore'
        ) as mock_store_cls:
            mock_store = MagicMock()
            mock_store.get_workspace_by_name = AsyncMock(return_value=mock_workspace)
            mock_store_cls.get_instance.return_value = mock_store

            with pytest.raises(RuntimeError, match='not active'):
                await processor._post_summary_to_jira_dc('test summary')

    @pytest.mark.asyncio
    async def test_successful_post(self, processor):
        """Should successfully post comment to Jira DC."""
        mock_workspace = MagicMock(status='active', svc_acc_api_key='encrypted_key')

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with (
            patch(
                'integrations.jira_dc.jira_dc_v1_callback_processor.JiraDcIntegrationStore'
            ) as mock_store_cls,
            patch(
                'integrations.jira_dc.jira_dc_v1_callback_processor.TokenManager'
            ) as mock_tm_cls,
            patch('httpx.AsyncClient') as mock_client_cls,
        ):
            mock_store = MagicMock()
            mock_store.get_workspace_by_name = AsyncMock(return_value=mock_workspace)
            mock_store_cls.get_instance.return_value = mock_store

            mock_tm = MagicMock()
            mock_tm.decrypt_text.return_value = 'decrypted_key'
            mock_tm_cls.return_value = mock_tm

            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value.__aenter__ = AsyncMock(
                return_value=mock_client
            )
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await processor._post_summary_to_jira_dc('Test summary')

            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert 'TEST-123' in call_args[0][0]
            assert call_args[1]['json'] == {'body': 'Test summary'}
