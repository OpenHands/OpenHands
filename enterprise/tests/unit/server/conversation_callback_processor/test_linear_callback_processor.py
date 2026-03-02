from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from server.conversation_callback_processor.linear_callback_processor import (
    LinearCallbackProcessor,
)

from openhands.core.schema.agent import AgentState
from openhands.events.action import MessageAction
from openhands.events.observation.agent import AgentStateChangedObservation


@pytest.fixture
def processor():
    processor = LinearCallbackProcessor(
        issue_id='TEST-123',
        issue_key='TEST-123',
        workspace_name='test-workspace',
    )
    return processor


@pytest.mark.asyncio
@patch(
    'server.conversation_callback_processor.linear_callback_processor.linear_manager'
)
async def test_send_comment_to_linear_success(mock_linear_manager, processor):
    # Setup
    mock_workspace = MagicMock(status='active', svc_acc_api_key='encrypted_key')
    mock_linear_manager.integration_store.get_workspace_by_name = AsyncMock(
        return_value=mock_workspace
    )
    mock_linear_manager.token_manager.decrypt_text.return_value = 'decrypted_key'
    mock_linear_manager.send_message = AsyncMock()

    # Action
    await processor._send_comment_to_linear('This is a summary.')

    # Assert
    mock_linear_manager.integration_store.get_workspace_by_name.assert_called_once_with(
        'test-workspace'
    )
    mock_linear_manager.send_message.assert_called_once()


@pytest.mark.asyncio
async def test_call_ignores_irrelevant_state(processor):
    callback = MagicMock()
    observation = AgentStateChangedObservation(
        agent_state=AgentState.RUNNING, content=''
    )

    with patch(
        'server.conversation_callback_processor.linear_callback_processor.conversation_manager'
    ) as mock_conv_manager:
        await processor(callback, observation)
        mock_conv_manager.send_event_to_conversation.assert_not_called()


@pytest.mark.asyncio
@patch(
    'server.conversation_callback_processor.linear_callback_processor.get_summary_instruction',
    return_value='Summarize this.',
)
@patch(
    'server.conversation_callback_processor.linear_callback_processor.get_last_user_msg_from_conversation_manager',
    new_callable=AsyncMock,
)
@patch(
    'server.conversation_callback_processor.linear_callback_processor.conversation_manager',
    new_callable=AsyncMock,
)
async def test_call_sends_summary_instruction(
    mock_conv_manager, mock_get_last_msg, mock_get_summary_instruction, processor
):
    callback = MagicMock(conversation_id='conv1')
    observation = AgentStateChangedObservation(
        agent_state=AgentState.FINISHED, content=''
    )
    mock_get_last_msg.return_value = [
        MessageAction(content='Not a summary instruction')
    ]

    await processor(callback, observation)

    mock_conv_manager.send_event_to_conversation.assert_called_once()
    call_args = mock_conv_manager.send_event_to_conversation.call_args[0]
    assert call_args[0] == 'conv1'
    assert call_args[1]['action'] == 'message'
    assert call_args[1]['args']['content'] == 'Summarize this.'


@pytest.mark.asyncio
@patch(
    'server.conversation_callback_processor.linear_callback_processor.linear_manager'
)
@patch(
    'server.conversation_callback_processor.linear_callback_processor.extract_summary_from_conversation_manager',
    new_callable=AsyncMock,
)
@patch(
    'server.conversation_callback_processor.linear_callback_processor.get_last_user_msg_from_conversation_manager',
    new_callable=AsyncMock,
)
@patch(
    'server.conversation_callback_processor.linear_callback_processor.get_summary_instruction',
    return_value='Summarize this.',
)
async def test_call_sends_summary_to_linear(
    mock_get_summary_instruction,
    mock_get_last_msg,
    mock_extract_summary,
    mock_linear_manager,
    processor,
):
    callback = MagicMock(conversation_id='conv1')
    observation = AgentStateChangedObservation(
        agent_state=AgentState.AWAITING_USER_INPUT, content=''
    )
    mock_get_last_msg.return_value = [MessageAction(content='Summarize this.')]
    mock_extract_summary.return_value = 'Extracted summary.'
    mock_workspace = MagicMock(status='active', svc_acc_api_key='encrypted_key')
    mock_linear_manager.integration_store.get_workspace_by_name = AsyncMock(
        return_value=mock_workspace
    )
    mock_linear_manager.send_message = AsyncMock()

    with patch(
        'server.conversation_callback_processor.linear_callback_processor.asyncio.create_task'
    ) as mock_create_task, patch(
        'server.conversation_callback_processor.linear_callback_processor.conversation_manager'
    ) as mock_conv_manager:
        await processor(callback, observation)
        mock_create_task.assert_called_once()
        # To ensure the coro is awaited in test
        await mock_create_task.call_args[0][0]

    mock_extract_summary.assert_called_once_with(mock_conv_manager, 'conv1')
    mock_linear_manager.send_message.assert_called_once()


@pytest.mark.asyncio
@patch(
    'server.conversation_callback_processor.linear_callback_processor.linear_manager'
)
async def test_send_comment_to_linear_workspace_not_found(
    mock_linear_manager, processor
):
    """Test behavior when workspace is not found"""
    # Setup
    mock_linear_manager.integration_store.get_workspace_by_name = AsyncMock(
        return_value=None
    )

    # Action
    await processor._send_comment_to_linear('This is a summary.')

    # Assert
    mock_linear_manager.integration_store.get_workspace_by_name.assert_called_once_with(
        'test-workspace'
    )
    # Should not attempt to send message when workspace not found
    mock_linear_manager.send_message.assert_not_called()


@pytest.mark.asyncio
@patch(
    'server.conversation_callback_processor.linear_callback_processor.linear_manager'
)
async def test_send_comment_to_linear_inactive_workspace(
    mock_linear_manager, processor
):
    """Test behavior when workspace is inactive"""
    # Setup
    mock_workspace = MagicMock(status='inactive', svc_acc_api_key='encrypted_key')
    mock_linear_manager.integration_store.get_workspace_by_name = AsyncMock(
        return_value=mock_workspace
    )

    # Action
    await processor._send_comment_to_linear('This is a summary.')

    # Assert
    # Should not attempt to send message when workspace is inactive
    mock_linear_manager.send_message.assert_not_called()


@pytest.mark.asyncio
@patch(
    'server.conversation_callback_processor.linear_callback_processor.linear_manager'
)
async def test_send_comment_to_linear_api_error(mock_linear_manager, processor):
    """Test behavior when API call fails"""
    # Setup
    mock_workspace = MagicMock(status='active', svc_acc_api_key='encrypted_key')
    mock_linear_manager.integration_store.get_workspace_by_name = AsyncMock(
        return_value=mock_workspace
    )
    mock_linear_manager.token_manager.decrypt_text.return_value = 'decrypted_key'
    mock_linear_manager.send_message = AsyncMock(side_effect=Exception('API Error'))

    # Action - should not raise exception, but handle it gracefully
    await processor._send_comment_to_linear('This is a summary.')

    # Assert
    mock_linear_manager.send_message.assert_called_once()


# Test with various agent states
@pytest.mark.asyncio
@pytest.mark.parametrize(
    'agent_state',
    [
        AgentState.LOADING,
        AgentState.RUNNING,
        AgentState.PAUSED,
        AgentState.STOPPED,
        AgentState.ERROR,
    ],
)
async def test_call_ignores_irrelevant_states(processor, agent_state):
    """Test that processor ignores irrelevant agent states"""
    callback = MagicMock()
    observation = AgentStateChangedObservation(agent_state=agent_state, content='')

    with patch(
        'server.conversation_callback_processor.linear_callback_processor.conversation_manager'
    ) as mock_conv_manager:
        await processor(callback, observation)
        mock_conv_manager.send_event_to_conversation.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'agent_state',
    [
        AgentState.AWAITING_USER_INPUT,
        AgentState.FINISHED,
    ],
)
async def test_call_processes_relevant_states(processor, agent_state):
    """Test that processor handles relevant agent states"""
    callback = MagicMock(conversation_id='conv1')
    observation = AgentStateChangedObservation(agent_state=agent_state, content='')

    with patch(
        'server.conversation_callback_processor.linear_callback_processor.get_summary_instruction',
        return_value='Summarize this.',
    ), patch(
        'server.conversation_callback_processor.linear_callback_processor.get_last_user_msg_from_conversation_manager',
        new_callable=AsyncMock,
        return_value=[MessageAction(content='Not a summary instruction')],
    ), patch(
        'server.conversation_callback_processor.linear_callback_processor.conversation_manager',
        new_callable=AsyncMock,
    ) as mock_conv_manager:
        await processor(callback, observation)
        mock_conv_manager.send_event_to_conversation.assert_called_once()


# Test empty last messages
@pytest.mark.asyncio
@patch(
    'server.conversation_callback_processor.linear_callback_processor.get_summary_instruction',
    return_value='Summarize this.',
)
@patch(
    'server.conversation_callback_processor.linear_callback_processor.get_last_user_msg_from_conversation_manager',
    new_callable=AsyncMock,
)
@patch(
    'server.conversation_callback_processor.linear_callback_processor.conversation_manager',
    new_callable=AsyncMock,
)
async def test_call_handles_empty_last_messages(
    mock_conv_manager, mock_get_last_msg, mock_get_summary_instruction, processor
):
    """Test behavior when there are no last user messages"""
    callback = MagicMock(conversation_id='conv1')
    observation = AgentStateChangedObservation(
        agent_state=AgentState.FINISHED, content=''
    )
    mock_get_last_msg.return_value = []  # Empty list

    await processor(callback, observation)

    # Should send summary instruction when no previous messages
    mock_conv_manager.send_event_to_conversation.assert_called_once()


# Test exception handling in main callback
@pytest.mark.asyncio
@patch(
    'server.conversation_callback_processor.linear_callback_processor.get_summary_instruction',
    side_effect=Exception('Unexpected error'),
)
async def test_call_handles_exceptions_gracefully(
    mock_get_summary_instruction, processor
):
    """Test that exceptions in callback processing are handled gracefully"""
    callback = MagicMock(conversation_id='conv1')
    observation = AgentStateChangedObservation(
        agent_state=AgentState.FINISHED, content=''
    )

    # Should not raise exception
    await processor(callback, observation)


# Test correct message construction
@pytest.mark.asyncio
@patch(
    'server.conversation_callback_processor.linear_callback_processor.linear_manager'
)
async def test_send_comment_to_linear_message_construction(
    mock_linear_manager, processor
):
    """Test that outgoing message is constructed correctly"""
    # Setup
    mock_workspace = MagicMock(
        status='active', svc_acc_api_key='encrypted_key', id='workspace_123'
    )
    mock_linear_manager.integration_store.get_workspace_by_name = AsyncMock(
        return_value=mock_workspace
    )
    mock_linear_manager.token_manager.decrypt_text.return_value = 'decrypted_key'
    mock_linear_manager.send_message = AsyncMock()

    test_message = 'This is a test summary message.'

    # Action
    await processor._send_comment_to_linear(test_message)

    # Assert - send_message now receives the string directly
    mock_linear_manager.send_message.assert_called_once_with(
        test_message,
        'TEST-123',  # issue_id
        'decrypted_key',  # api_key
    )


# Test asyncio.create_task usage
@pytest.mark.asyncio
@patch(
    'server.conversation_callback_processor.linear_callback_processor.linear_manager'
)
@patch(
    'server.conversation_callback_processor.linear_callback_processor.extract_summary_from_conversation_manager',
    new_callable=AsyncMock,
)
@patch(
    'server.conversation_callback_processor.linear_callback_processor.get_last_user_msg_from_conversation_manager',
    new_callable=AsyncMock,
)
@patch(
    'server.conversation_callback_processor.linear_callback_processor.get_summary_instruction',
    return_value='Summarize this.',
)
async def test_call_creates_background_task_for_sending(
    mock_get_summary_instruction,
    mock_get_last_msg,
    mock_extract_summary,
    mock_linear_manager,
    processor,
):
    """Test that summary sending is done in background task"""
    callback = MagicMock(conversation_id='conv1')
    observation = AgentStateChangedObservation(
        agent_state=AgentState.AWAITING_USER_INPUT, content=''
    )
    mock_get_last_msg.return_value = [MessageAction(content='Summarize this.')]
    mock_extract_summary.return_value = 'Extracted summary.'
    mock_workspace = MagicMock(status='active', svc_acc_api_key='encrypted_key')
    mock_linear_manager.integration_store.get_workspace_by_name = AsyncMock(
        return_value=mock_workspace
    )
    mock_linear_manager.send_message = AsyncMock()

    with patch(
        'server.conversation_callback_processor.linear_callback_processor.asyncio.create_task'
    ) as mock_create_task, patch(
        'server.conversation_callback_processor.linear_callback_processor.conversation_manager'
    ):
        await processor(callback, observation)

        # Verify that create_task was called
        mock_create_task.assert_called_once()

        # Verify the task is for sending comment
        task_coro = mock_create_task.call_args[0][0]
        assert task_coro.__class__.__name__ == 'coroutine'
