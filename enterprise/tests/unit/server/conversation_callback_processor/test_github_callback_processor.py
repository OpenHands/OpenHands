"""Tests for the GithubCallbackProcessor.

Covers:
- Event filtering (ignores irrelevant agent states)
- Summary instruction flow
- Summary extraction and sending to GitHub
- Error handling
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from server.conversation_callback_processor.github_callback_processor import (
    GithubCallbackProcessor,
)
from storage.conversation_callback import CallbackStatus

from openhands.core.schema.agent import AgentState
from openhands.events.observation.agent import AgentStateChangedObservation


@pytest.fixture
def github_view():
    """Create a mock GithubViewType."""
    view = MagicMock()
    view.full_repo_name = 'test-owner/test-repo'
    view.issue_number = 42
    view.installation_id = 12345
    return view


@pytest.fixture
def processor(github_view):
    """Create a GithubCallbackProcessor instance."""
    return GithubCallbackProcessor(
        github_view=github_view,
        send_summary_instruction=True,
    )


@pytest.fixture
def callback():
    """Create a mock ConversationCallback."""
    cb = MagicMock()
    cb.conversation_id = 'conv-123'
    cb.status = CallbackStatus.PENDING
    cb.set_processor = MagicMock()
    return cb


# ---------------------------------------------------------------------------
# Event filtering tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_ignores_running_state(processor, callback):
    """Processor should ignore RUNNING state."""
    observation = AgentStateChangedObservation(
        agent_state=AgentState.RUNNING, content=''
    )

    with patch(
        'server.conversation_callback_processor.github_callback_processor.conversation_manager'
    ) as mock_conv_manager:
        await processor(callback, observation)
        mock_conv_manager.send_event_to_conversation.assert_not_called()


@pytest.mark.asyncio
async def test_call_ignores_paused_state(processor, callback):
    """Processor should ignore PAUSED state."""
    observation = AgentStateChangedObservation(
        agent_state=AgentState.PAUSED, content=''
    )

    with patch(
        'server.conversation_callback_processor.github_callback_processor.conversation_manager'
    ) as mock_conv_manager:
        await processor(callback, observation)
        mock_conv_manager.send_event_to_conversation.assert_not_called()


@pytest.mark.asyncio
async def test_call_ignores_error_state(processor, callback):
    """Processor should ignore ERROR state."""
    observation = AgentStateChangedObservation(
        agent_state=AgentState.ERROR, content=''
    )

    with patch(
        'server.conversation_callback_processor.github_callback_processor.conversation_manager'
    ) as mock_conv_manager:
        await processor(callback, observation)
        mock_conv_manager.send_event_to_conversation.assert_not_called()


@pytest.mark.asyncio
async def test_call_processes_finished_state(processor, callback):
    """Processor should process FINISHED state."""
    observation = AgentStateChangedObservation(
        agent_state=AgentState.FINISHED, content=''
    )

    with (
        patch(
            'server.conversation_callback_processor.github_callback_processor.conversation_manager'
        ) as mock_conv_manager,
        patch(
            'server.conversation_callback_processor.github_callback_processor.get_summary_instruction',
            return_value='Summarize this.',
        ),
    ):
        mock_conv_manager.send_event_to_conversation = AsyncMock()
        await processor(callback, observation)
        mock_conv_manager.send_event_to_conversation.assert_called_once()


@pytest.mark.asyncio
async def test_call_processes_awaiting_user_input_state(processor, callback):
    """Processor should process AWAITING_USER_INPUT state."""
    observation = AgentStateChangedObservation(
        agent_state=AgentState.AWAITING_USER_INPUT, content=''
    )

    with (
        patch(
            'server.conversation_callback_processor.github_callback_processor.conversation_manager'
        ) as mock_conv_manager,
        patch(
            'server.conversation_callback_processor.github_callback_processor.get_summary_instruction',
            return_value='Summarize this.',
        ),
    ):
        mock_conv_manager.send_event_to_conversation = AsyncMock()
        await processor(callback, observation)
        mock_conv_manager.send_event_to_conversation.assert_called_once()


# ---------------------------------------------------------------------------
# Summary instruction flow tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_sends_summary_instruction_first(processor, callback):
    """Processor should send summary instruction on first call."""
    observation = AgentStateChangedObservation(
        agent_state=AgentState.FINISHED, content=''
    )

    assert processor.send_summary_instruction is True

    with (
        patch(
            'server.conversation_callback_processor.github_callback_processor.conversation_manager'
        ) as mock_conv_manager,
        patch(
            'server.conversation_callback_processor.github_callback_processor.get_summary_instruction',
            return_value='Please provide a summary.',
        ),
    ):
        mock_conv_manager.send_event_to_conversation = AsyncMock()
        await processor(callback, observation)

        mock_conv_manager.send_event_to_conversation.assert_called_once()
        # Verify the call was for the summary instruction
        call_args = mock_conv_manager.send_event_to_conversation.call_args
        assert call_args[0][0] == 'conv-123'
        # Processor state should be updated
        assert processor.send_summary_instruction is False
        callback.set_processor.assert_called_once_with(processor)


@pytest.mark.asyncio
async def test_call_extracts_and_sends_summary_on_second_call(processor, callback):
    """Processor should extract and send summary when send_summary_instruction is False."""
    # Simulate second call where summary instruction was already sent
    processor.send_summary_instruction = False

    observation = AgentStateChangedObservation(
        agent_state=AgentState.FINISHED, content=''
    )

    with (
        patch(
            'server.conversation_callback_processor.github_callback_processor.conversation_manager'
        ),
        patch(
            'server.conversation_callback_processor.github_callback_processor.extract_summary_from_conversation_manager',
            new_callable=AsyncMock,
            return_value='This is the summary.',
        ) as mock_extract,
        patch.object(
            processor,
            '_send_message_to_github',
            new_callable=AsyncMock,
        ) as mock_send,
        patch('asyncio.create_task') as mock_create_task,
    ):
        await processor(callback, observation)

        mock_extract.assert_called_once()
        mock_create_task.assert_called_once()
        assert callback.status == CallbackStatus.COMPLETED


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_sets_error_status_on_exception(processor, callback):
    """Processor should set ERROR status when exception occurs."""
    processor.send_summary_instruction = False

    observation = AgentStateChangedObservation(
        agent_state=AgentState.FINISHED, content=''
    )

    with (
        patch(
            'server.conversation_callback_processor.github_callback_processor.conversation_manager'
        ),
        patch(
            'server.conversation_callback_processor.github_callback_processor.extract_summary_from_conversation_manager',
            new_callable=AsyncMock,
            side_effect=RuntimeError('Failed to extract summary'),
        ),
    ):
        await processor(callback, observation)

        assert callback.status == CallbackStatus.ERROR


# ---------------------------------------------------------------------------
# _send_message_to_github tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_message_to_github_success(processor, github_view):
    """_send_message_to_github should send message via GithubManager."""
    with (
        patch(
            'server.conversation_callback_processor.github_callback_processor.TokenManager'
        ) as mock_token_manager_cls,
        patch(
            'server.conversation_callback_processor.github_callback_processor.GithubManager'
        ) as mock_github_manager_cls,
        patch(
            'integrations.github.data_collector.GitHubDataCollector'
        ) as mock_data_collector_cls,
    ):
        mock_token_manager = MagicMock()
        mock_token_manager_cls.return_value = mock_token_manager

        mock_data_collector = MagicMock()
        mock_data_collector_cls.return_value = mock_data_collector

        mock_github_manager = MagicMock()
        mock_github_manager.send_message = AsyncMock()
        mock_github_manager_cls.return_value = mock_github_manager

        await processor._send_message_to_github('Test summary message')

        mock_github_manager.send_message.assert_called_once_with(
            'Test summary message', github_view
        )


@pytest.mark.asyncio
async def test_send_message_to_github_handles_exception(processor):
    """_send_message_to_github should catch and log exceptions."""
    with (
        patch(
            'server.conversation_callback_processor.github_callback_processor.TokenManager'
        ) as mock_token_manager_cls,
        patch(
            'server.conversation_callback_processor.github_callback_processor.GithubManager'
        ) as mock_github_manager_cls,
    ):
        mock_token_manager_cls.return_value = MagicMock()
        mock_github_manager_cls.side_effect = RuntimeError('GitHub API error')

        # Should not raise
        await processor._send_message_to_github('Test message')


# ---------------------------------------------------------------------------
# Callback state update tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_updates_callback_updated_at_on_instruction(processor, callback):
    """Processor should update callback.updated_at when sending summary instruction."""
    observation = AgentStateChangedObservation(
        agent_state=AgentState.FINISHED, content=''
    )

    original_time = callback.updated_at

    with (
        patch(
            'server.conversation_callback_processor.github_callback_processor.conversation_manager'
        ) as mock_conv_manager,
        patch(
            'server.conversation_callback_processor.github_callback_processor.get_summary_instruction',
            return_value='Summarize.',
        ),
    ):
        mock_conv_manager.send_event_to_conversation = AsyncMock()
        await processor(callback, observation)

        # updated_at should be set
        assert callback.updated_at != original_time
        assert isinstance(callback.updated_at, datetime)


@pytest.mark.asyncio
async def test_call_updates_callback_updated_at_on_completion(processor, callback):
    """Processor should update callback.updated_at when completing."""
    processor.send_summary_instruction = False

    observation = AgentStateChangedObservation(
        agent_state=AgentState.FINISHED, content=''
    )

    with (
        patch(
            'server.conversation_callback_processor.github_callback_processor.conversation_manager'
        ),
        patch(
            'server.conversation_callback_processor.github_callback_processor.extract_summary_from_conversation_manager',
            new_callable=AsyncMock,
            return_value='Summary',
        ),
        patch('asyncio.create_task'),
    ):
        await processor(callback, observation)

        assert callback.status == CallbackStatus.COMPLETED
        assert isinstance(callback.updated_at, datetime)
