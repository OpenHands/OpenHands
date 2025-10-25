"""Test function calling module."""

import json
from unittest.mock import patch

import pytest
from litellm import ModelResponse

from openhands.agenthub.codeact_agent.function_calling import response_to_actions
from openhands.core.exceptions import FunctionCallValidationError
from openhands.events.action import (
    BrowseInteractiveAction,
    CmdRunAction,
    FileEditAction,
    FileReadAction,
    IPythonRunCellAction,
    MessageAction,
)
from openhands.events.event import FileEditSource, FileReadSource


def create_mock_response(function_name: str, arguments: dict) -> ModelResponse:
    """Helper function to create a mock response with a tool call."""
    return ModelResponse(
        id='mock-id',
        choices=[
            {
                'message': {
                    'tool_calls': [
                        {
                            'function': {
                                'name': function_name,
                                'arguments': json.dumps(arguments),
                            },
                            'id': 'mock-tool-call-id',
                            'type': 'function',
                        }
                    ],
                    'content': None,
                    'role': 'assistant',
                },
                'index': 0,
                'finish_reason': 'tool_calls',
            }
        ],
    )


def test_execute_bash_valid():
    """Test execute_bash with valid arguments."""
    response = create_mock_response(
        'execute_bash', {'command': 'ls', 'is_input': 'false', 'security_risk': 'LOW'}
    )
    actions = response_to_actions(response)
    assert len(actions) == 1
    assert isinstance(actions[0], CmdRunAction)
    assert actions[0].command == 'ls'
    assert actions[0].is_input is False

    # Test with timeout parameter
    with patch.object(CmdRunAction, 'set_hard_timeout') as mock_set_hard_timeout:
        response_with_timeout = create_mock_response(
            'execute_bash',
            {
                'command': 'ls',
                'is_input': 'false',
                'timeout': 30,
                'security_risk': 'LOW',
            },
        )
        actions_with_timeout = response_to_actions(response_with_timeout)

        # Verify set_hard_timeout was called with the correct value
        mock_set_hard_timeout.assert_called_once_with(30.0)

        assert len(actions_with_timeout) == 1
        assert isinstance(actions_with_timeout[0], CmdRunAction)
        assert actions_with_timeout[0].command == 'ls'
        assert actions_with_timeout[0].is_input is False


def test_execute_bash_missing_command():
    """Test execute_bash with missing command argument."""
    response = create_mock_response(
        'execute_bash', {'is_input': 'false', 'security_risk': 'LOW'}
    )
    with pytest.raises(FunctionCallValidationError) as exc_info:
        response_to_actions(response)
    assert 'Missing required argument "command"' in str(exc_info.value)


def test_execute_ipython_cell_valid():
    """Test execute_ipython_cell with valid arguments."""
    response = create_mock_response(
        'execute_ipython_cell', {'code': "print('hello')", 'security_risk': 'LOW'}
    )
    actions = response_to_actions(response)
    assert len(actions) == 1
    assert isinstance(actions[0], IPythonRunCellAction)
    assert actions[0].code == "print('hello')"


def test_execute_ipython_cell_missing_code():
    """Test execute_ipython_cell with missing code argument."""
    response = create_mock_response('execute_ipython_cell', {'security_risk': 'LOW'})
    with pytest.raises(FunctionCallValidationError) as exc_info:
        response_to_actions(response)
    assert 'Missing required argument "code"' in str(exc_info.value)


def test_edit_file_valid():
    """Test edit_file with valid arguments."""
    response = create_mock_response(
        'edit_file',
        {
            'path': '/path/to/file',
            'content': 'file content',
            'start': 1,
            'end': 10,
            'security_risk': 'LOW',
        },
    )
    actions = response_to_actions(response)
    assert len(actions) == 1
    assert isinstance(actions[0], FileEditAction)
    assert actions[0].path == '/path/to/file'
    assert actions[0].content == 'file content'
    assert actions[0].start == 1
    assert actions[0].end == 10


def test_edit_file_missing_required():
    """Test edit_file with missing required arguments."""
    # Missing path
    response = create_mock_response(
        'edit_file', {'content': 'content', 'security_risk': 'LOW'}
    )
    with pytest.raises(FunctionCallValidationError) as exc_info:
        response_to_actions(response)
    assert 'Missing required argument "path"' in str(exc_info.value)

    # Missing content
    response = create_mock_response(
        'edit_file', {'path': '/path/to/file', 'security_risk': 'LOW'}
    )
    with pytest.raises(FunctionCallValidationError) as exc_info:
        response_to_actions(response)
    assert 'Missing required argument "content"' in str(exc_info.value)


def test_str_replace_editor_valid():
    """Test str_replace_editor with valid arguments."""
    # Test view command
    response = create_mock_response(
        'str_replace_editor',
        {'command': 'view', 'path': '/path/to/file', 'security_risk': 'LOW'},
    )
    actions = response_to_actions(response)
    assert len(actions) == 1
    assert isinstance(actions[0], FileReadAction)
    assert actions[0].path == '/path/to/file'
    assert actions[0].impl_source == FileReadSource.OH_ACI

    # Test other commands
    response = create_mock_response(
        'str_replace_editor',
        {
            'command': 'str_replace',
            'path': '/path/to/file',
            'old_str': 'old',
            'new_str': 'new',
            'security_risk': 'LOW',
        },
    )
    actions = response_to_actions(response)
    assert len(actions) == 1
    assert isinstance(actions[0], FileEditAction)
    assert actions[0].path == '/path/to/file'
    assert actions[0].impl_source == FileEditSource.OH_ACI


def test_str_replace_editor_missing_required():
    """Test str_replace_editor with missing required arguments."""
    # Missing command
    response = create_mock_response(
        'str_replace_editor', {'path': '/path/to/file', 'security_risk': 'LOW'}
    )
    with pytest.raises(FunctionCallValidationError) as exc_info:
        response_to_actions(response)
    assert 'Missing required argument "command"' in str(exc_info.value)

    # Missing path
    response = create_mock_response(
        'str_replace_editor', {'command': 'view', 'security_risk': 'LOW'}
    )
    with pytest.raises(FunctionCallValidationError) as exc_info:
        response_to_actions(response)
    assert 'Missing required argument "path"' in str(exc_info.value)


def test_browser_valid():
    """Test browser with valid arguments."""
    response = create_mock_response(
        'browser', {'code': "click('button-1')", 'security_risk': 'LOW'}
    )
    actions = response_to_actions(response)
    assert len(actions) == 1
    assert isinstance(actions[0], BrowseInteractiveAction)
    assert actions[0].browser_actions == "click('button-1')"
    assert actions[0].return_axtree is False  # Default value should be False


def test_browser_missing_code():
    """Test browser with missing code argument."""
    response = create_mock_response('browser', {'security_risk': 'LOW'})
    with pytest.raises(FunctionCallValidationError) as exc_info:
        response_to_actions(response)
    assert 'Missing required argument "code"' in str(exc_info.value)


def test_invalid_json_arguments():
    """Test handling of invalid JSON in arguments."""
    response = ModelResponse(
        id='mock-id',
        choices=[
            {
                'message': {
                    'tool_calls': [
                        {
                            'function': {
                                'name': 'execute_bash',
                                'arguments': 'invalid json',
                            },
                            'id': 'mock-tool-call-id',
                            'type': 'function',
                        }
                    ],
                    'content': None,
                    'role': 'assistant',
                },
                'index': 0,
                'finish_reason': 'tool_calls',
            }
        ],
    )
    with pytest.raises(FunctionCallValidationError) as exc_info:
        response_to_actions(response)
    assert 'Failed to parse tool call arguments' in str(exc_info.value)


def test_unexpected_argument_handling():
    """Test that unexpected arguments in function calls are properly handled.

    This test reproduces issue #8369 Example 4 where an unexpected argument
    (old_str_prefix) causes a TypeError.
    """
    response = create_mock_response(
        'str_replace_editor',
        {
            'command': 'str_replace',
            'path': '/test/file.py',
            'old_str': 'def test():\n    pass',
            'new_str': 'def test():\n    return True',
            'old_str_prefix': 'some prefix',  # Unexpected argument
            'security_risk': 'LOW',
        },
    )

    # Test that the function raises a FunctionCallValidationError
    with pytest.raises(FunctionCallValidationError) as exc_info:
        response_to_actions(response)

    # Verify the error message mentions the unexpected argument
    assert 'old_str_prefix' in str(exc_info.value)
    assert 'Unexpected argument' in str(exc_info.value)


def test_message_action_empty_response_non_grok():
    response = ModelResponse(
        id='mock-id',
        model='gpt-4o',
        choices=[
            {
                'message': {
                    'content': '',
                    'role': 'assistant',
                },
                'index': 0,
                'finish_reason': 'stop',
            }
        ],
    )
    actions = response_to_actions(response)
    assert len(actions) == 1
    assert isinstance(actions[0], MessageAction)
    assert actions[0].content == ''
    assert actions[0].wait_for_response is True


def test_message_action_empty_response_grok():
    response = ModelResponse(
        id='mock-id',
        model='xai/grok-4-0709',
        choices=[
            {
                'message': {
                    'content': '',
                    'role': 'assistant',
                },
                'index': 0,
                'finish_reason': 'stop',
            }
        ],
    )
    actions = response_to_actions(response)
    assert len(actions) == 1
    assert isinstance(actions[0], MessageAction)
    assert actions[0].content == ''
    assert actions[0].wait_for_response is False


def test_message_action_non_empty_response_grok():
    response = ModelResponse(
        id='mock-id',
        model='xai/grok-4-0709',
        choices=[
            {
                'message': {
                    'content': 'Here is my response.',
                    'role': 'assistant',
                },
                'index': 0,
                'finish_reason': 'stop',
            }
        ],
    )
    actions = response_to_actions(response)
    assert len(actions) == 1
    assert isinstance(actions[0], MessageAction)
    assert actions[0].content == 'Here is my response.'
    assert actions[0].wait_for_response is True


@pytest.mark.parametrize(
    'model_name',
    [
        'xai/grok-4-0709',
        'litellm_proxy/xai/grok-4-0709',
        'openrouter/xai/grok-4-0709',
    ],
)
def test_message_action_grok_model_variants(model_name):
    response = ModelResponse(
        id='mock-id',
        model=model_name,
        choices=[
            {
                'message': {
                    'content': '',
                    'role': 'assistant',
                },
                'index': 0,
                'finish_reason': 'stop',
            }
        ],
    )
    actions = response_to_actions(response)
    assert len(actions) == 1
    assert isinstance(actions[0], MessageAction)
    assert actions[0].content == ''
    assert actions[0].wait_for_response is False


@pytest.mark.parametrize(
    'model_name',
    [
        'grok-lite',
        'xai/other-model',
        'gpt-4-grok-mode',
    ],
)
def test_message_action_similar_model_names(model_name):
    response = ModelResponse(
        id='mock-id',
        model=model_name,
        choices=[
            {
                'message': {
                    'content': '',
                    'role': 'assistant',
                },
                'index': 0,
                'finish_reason': 'stop',
            }
        ],
    )
    actions = response_to_actions(response)
    assert len(actions) == 1
    assert isinstance(actions[0], MessageAction)
    assert actions[0].content == ''
    assert actions[0].wait_for_response is True
