import os
import shutil
from unittest.mock import MagicMock, Mock

import pytest
from litellm import ChatCompletionMessageToolCall

from openhands.controller.state.state import State
from openhands.core.config.agent_config import AgentConfig
from openhands.core.message import ImageContent, Message, TextContent
from openhands.events.action import (
    AgentFinishAction,
    CmdRunAction,
    MessageAction,
)
from openhands.events.action.message import SystemMessageAction
from openhands.events.event import (
    Event,
    EventSource,
    FileEditSource,
    FileReadSource,
    RecallType,
)
from openhands.events.observation import CmdOutputObservation
from openhands.events.observation.agent import (
    MicroagentKnowledge,
    RecallObservation,
)
from openhands.events.observation.browse import BrowserOutputObservation
from openhands.events.observation.commands import (
    CmdOutputMetadata,
    IPythonRunCellObservation,
)
from openhands.events.observation.delegate import AgentDelegateObservation
from openhands.events.observation.error import ErrorObservation
from openhands.events.observation.files import FileEditObservation, FileReadObservation
from openhands.events.observation.reject import UserRejectObservation
from openhands.events.tool import ToolCallMetadata
from openhands.memory.conversation_memory import ConversationMemory
from openhands.utils.prompt import PromptManager, RepositoryInfo, RuntimeInfo


@pytest.fixture
def agent_config():
    return AgentConfig(
        enable_prompt_extensions=True,
        enable_som_visual_browsing=True,
        disabled_microagents=['disabled_agent'],
    )


@pytest.fixture
def conversation_memory(agent_config):
    prompt_manager = MagicMock(spec=PromptManager)
    prompt_manager.get_system_message.return_value = 'System message'
    prompt_manager.build_workspace_context.return_value = (
        'Formatted repository and runtime info'
    )

    # Make build_microagent_info return the actual content from the triggered agents
    def build_microagent_info(triggered_agents):
        if not triggered_agents:
            return ''
        return '\n'.join(agent.content for agent in triggered_agents)

    prompt_manager.build_microagent_info.side_effect = build_microagent_info
    return ConversationMemory(agent_config, prompt_manager)


@pytest.fixture
def prompt_dir(tmp_path):
    # Copy contents from "openhands/agenthub/codeact_agent" to the temp directory
    shutil.copytree(
        'openhands/agenthub/codeact_agent/prompts', tmp_path, dirs_exist_ok=True
    )

    # Return the temporary directory path
    return tmp_path


@pytest.fixture
def mock_state():
    state = MagicMock(spec=State)
    state.history = []
    return state


@pytest.fixture
def mock_prompt_manager():
    return MagicMock()


def test_process_events_with_message_action(conversation_memory):
    """Test that MessageAction is processed correctly."""
    # Create a system message action
    system_message = SystemMessageAction(content='System message')
    system_message._source = EventSource.AGENT

    # Create user and assistant messages
    user_message = MessageAction(content='Hello')
    user_message._source = EventSource.USER
    assistant_message = MessageAction(content='Hi there')
    assistant_message._source = EventSource.AGENT

    # Process events
    messages = conversation_memory.process_events(
        condensed_history=[system_message, user_message, assistant_message],
        initial_user_action=user_message,
        max_message_chars=None,
        vision_is_active=False,
    )

    # Check that the messages were processed correctly
    assert len(messages) == 3
    assert messages[0].role == 'system'
    assert messages[0].content[0].text == 'System message'


# Test cases for _ensure_system_message
def test_ensure_system_message_adds_if_missing(conversation_memory):
    """Test that _ensure_system_message adds a system message if none exists."""
    user_message = MessageAction(content='User message')
    user_message._source = EventSource.USER
    events = [user_message]
    conversation_memory._ensure_system_message(events)
    assert len(events) == 2
    assert isinstance(events[0], SystemMessageAction)
    assert events[0].content == 'System message'  # From fixture
    assert isinstance(events[1], MessageAction)  # Original event is still there


def test_ensure_system_message_does_nothing_if_present(conversation_memory):
    """Test that _ensure_system_message does nothing if a system message is already present."""
    original_system_message = SystemMessageAction(content='Existing system message')
    user_message = MessageAction(content='User message')
    user_message._source = EventSource.USER
    events = [
        original_system_message,
        user_message,
    ]
    original_events = list(events)  # Copy before modification
    conversation_memory._ensure_system_message(events)
    assert events == original_events  # List should be unchanged


# Test cases for _ensure_initial_user_message
@pytest.fixture
def initial_user_action():
    msg = MessageAction(content='Initial User Message')
    msg._source = EventSource.USER
    return msg


def test_ensure_initial_user_message_adds_if_only_system(
    conversation_memory, initial_user_action
):
    """Test adding the initial user message when only the system message exists."""
    system_message = SystemMessageAction(content='System')
    system_message._source = EventSource.AGENT
    events = [system_message]
    # Pass empty set for forgotten_event_ids (no events have been condensed)
    conversation_memory._ensure_initial_user_message(events, initial_user_action, set())
    assert len(events) == 2
    assert events[0] == system_message
    assert events[1] == initial_user_action


def test_ensure_initial_user_message_correct_already_present(
    conversation_memory, initial_user_action
):
    """Test that nothing changes if the correct initial user message is at index 1."""
    system_message = SystemMessageAction(content='System')
    agent_message = MessageAction(content='Assistant')
    agent_message._source = EventSource.USER
    events = [
        system_message,
        initial_user_action,
        agent_message,
    ]
    original_events = list(events)
    # Pass empty set for forgotten_event_ids (no events have been condensed)
    conversation_memory._ensure_initial_user_message(events, initial_user_action, set())
    assert events == original_events


def test_ensure_initial_user_message_incorrect_at_index_1(
    conversation_memory, initial_user_action
):
    """Test inserting the correct initial user message when an incorrect message is at index 1."""
    system_message = SystemMessageAction(content='System')
    incorrect_second_message = MessageAction(content='Assistant')
    incorrect_second_message._source = EventSource.AGENT
    events = [system_message, incorrect_second_message]
    # Pass empty set for forgotten_event_ids (no events have been condensed)
    conversation_memory._ensure_initial_user_message(events, initial_user_action, set())
    assert len(events) == 3
    assert events[0] == system_message
    assert events[1] == initial_user_action  # Correct one inserted
    assert events[2] == incorrect_second_message  # Original second message shifted


def test_ensure_initial_user_message_correct_present_later(
    conversation_memory, initial_user_action
):
    """Test inserting the correct initial user message at index 1 even if it exists later."""
    system_message = SystemMessageAction(content='System')
    incorrect_second_message = MessageAction(content='Assistant')
    incorrect_second_message._source = EventSource.AGENT
    # Correct initial message is present, but later in the list
    events = [system_message, incorrect_second_message]
    conversation_memory._ensure_system_message(events)
    # Pass empty set for forgotten_event_ids (no events have been condensed)
    conversation_memory._ensure_initial_user_message(events, initial_user_action, set())
    assert len(events) == 3  # Should still insert at index 1, not remove the later one
    assert events[0] == system_message
    assert events[1] == initial_user_action  # Correct one inserted at index 1
    assert events[2] == incorrect_second_message  # Original second message shifted
    # The duplicate initial_user_action originally at index 2 is now at index 3 (implicitly tested by length and content)


def test_ensure_initial_user_message_different_user_msg_at_index_1(
    conversation_memory, initial_user_action
):
    """Test inserting the correct initial user message when a *different* user message is at index 1."""
    system_message = SystemMessageAction(content='System')
    different_user_message = MessageAction(content='Different User Message')
    different_user_message._source = EventSource.USER
    events = [system_message, different_user_message]
    # Pass empty set for forgotten_event_ids (no events have been condensed)
    conversation_memory._ensure_initial_user_message(events, initial_user_action, set())
    assert len(events) == 2
    assert events[0] == system_message
    assert events[1] == different_user_message  # Original second message remains


def test_ensure_initial_user_message_different_user_msg_at_index_1_and_orphaned_obs(
    conversation_memory, initial_user_action
):
    """Test process_events when an incorrect user message is at index 1 AND
    an orphaned observation (with tool_call_metadata but no matching action) exists.
    Expect: System msg, CORRECT initial user msg, the incorrect user msg (shifted).
            The orphaned observation should be filtered out.
    """
    system_message = SystemMessageAction(content='System')
    different_user_message = MessageAction(content='Different User Message')
    different_user_message._source = EventSource.USER

    # Create an orphaned observation (no matching action/tool call request will exist)
    # Use a dictionary that mimics ModelResponse structure to satisfy Pydantic
    mock_response = {
        'id': 'mock_response_id',
        'choices': [{'message': {'content': None, 'tool_calls': []}}],
        'created': 0,
        'model': '',
        'object': '',
        'usage': {'completion_tokens': 0, 'prompt_tokens': 0, 'total_tokens': 0},
    }
    orphaned_obs = CmdOutputObservation(
        command='orphan_cmd',
        content='Orphaned output',
        command_id=99,
        exit_code=0,
    )
    orphaned_obs.tool_call_metadata = ToolCallMetadata(
        tool_call_id='orphan_call_id',
        function_name='execute_bash',
        model_response=mock_response,
        total_calls_in_response=1,
    )

    # Initial events list: system, wrong user message, orphaned observation
    events = [system_message, different_user_message, orphaned_obs]

    # Call the main process_events method
    messages = conversation_memory.process_events(
        condensed_history=events,
        initial_user_action=initial_user_action,  # Provide the *correct* initial action
        max_message_chars=None,
        vision_is_active=False,
    )

    # Assertions on the final messages list
    assert len(messages) == 2
    # 1. System message should be first
    assert messages[0].role == 'system'
    assert messages[0].content[0].text == 'System'

    # 2. The different user message should be left at index 1
    assert messages[1].role == 'user'
    assert messages[1].content[0].text == different_user_message.content

    # Implicitly assert that the orphaned_obs was filtered out by checking the length (2)


def test_process_events_with_cmd_output_observation(conversation_memory):
    obs = CmdOutputObservation(
        command='echo hello',
        content='Command output',
        metadata=CmdOutputMetadata(
            exit_code=0,
            prefix='[THIS IS PREFIX]',
            suffix='[THIS IS SUFFIX]',
        ),
    )

    initial_user_message = MessageAction(content='Initial user message')
    initial_user_message._source = EventSource.USER
    messages = conversation_memory.process_events(
        condensed_history=[obs],
        initial_user_action=initial_user_message,
        max_message_chars=None,
        vision_is_active=False,
    )

    assert len(messages) == 3  # System + initial user + result
    result = messages[2]  # The actual result is now at index 2
    assert result.role == 'user'
    assert len(result.content) == 1
    assert isinstance(result.content[0], TextContent)
    assert 'Observed result of command executed by user:' in result.content[0].text
    assert '[Command finished with exit code 0]' in result.content[0].text
    assert '[THIS IS PREFIX]' in result.content[0].text
    assert '[THIS IS SUFFIX]' in result.content[0].text


def test_process_events_with_ipython_run_cell_observation(conversation_memory):
    obs = IPythonRunCellObservation(
        code='plt.plot()',
        content='IPython output\n![image](data:image/png;base64,ABC123)',
    )

    initial_user_message = MessageAction(content='Initial user message')
    initial_user_message._source = EventSource.USER
    messages = conversation_memory.process_events(
        condensed_history=[obs],
        initial_user_action=initial_user_message,
        max_message_chars=None,
        vision_is_active=False,
    )

    assert len(messages) == 3  # System + initial user + result
    result = messages[2]  # The actual result is now at index 2
    assert result.role == 'user'
    assert len(result.content) == 1
    assert isinstance(result.content[0], TextContent)
    assert 'IPython output' in result.content[0].text
    assert (
        '![image](data:image/png;base64, ...) already displayed to user'
        in result.content[0].text
    )
    assert 'ABC123' not in result.content[0].text


def test_process_events_with_agent_delegate_observation(conversation_memory):
    obs = AgentDelegateObservation(
        content='Content', outputs={'content': 'Delegated agent output'}
    )

    initial_user_message = MessageAction(content='Initial user message')
    initial_user_message._source = EventSource.USER
    messages = conversation_memory.process_events(
        condensed_history=[obs],
        initial_user_action=initial_user_message,
        max_message_chars=None,
        vision_is_active=False,
    )

    assert len(messages) == 3  # System + initial user + result
    result = messages[2]  # The actual result is now at index 2
    assert result.role == 'user'
    assert len(result.content) == 1
    assert isinstance(result.content[0], TextContent)
    assert 'Delegated agent output' in result.content[0].text


def test_process_events_with_error_observation(conversation_memory):
    obs = ErrorObservation('Error message')

    initial_user_message = MessageAction(content='Initial user message')
    initial_user_message._source = EventSource.USER
    messages = conversation_memory.process_events(
        condensed_history=[obs],
        initial_user_action=initial_user_message,
        max_message_chars=None,
        vision_is_active=False,
    )

    assert len(messages) == 3  # System + initial user + result
    result = messages[2]  # The actual result is now at index 2
    assert result.role == 'user'
    assert len(result.content) == 1
    assert isinstance(result.content[0], TextContent)
    assert 'Error message' in result.content[0].text
    assert 'Error occurred in processing last action' in result.content[0].text


def test_process_events_with_unknown_observation(conversation_memory):
    # Create a mock that inherits from Event but not Action or Observation
    obs = Mock(spec=Event)
    initial_user_message = MessageAction(content='Initial user message')
    initial_user_message._source = EventSource.USER

    with pytest.raises(ValueError, match='Unknown event type'):
        conversation_memory.process_events(
            condensed_history=[obs],
            initial_user_action=initial_user_message,
            max_message_chars=None,
            vision_is_active=False,
        )


def test_process_events_with_file_edit_observation(conversation_memory):
    obs = FileEditObservation(
        path='/test/file.txt',
        prev_exist=True,
        old_content='old content',
        new_content='new content',
        content='diff content',
        impl_source=FileEditSource.LLM_BASED_EDIT,
    )

    initial_user_message = MessageAction(content='Initial user message')
    initial_user_message._source = EventSource.USER
    messages = conversation_memory.process_events(
        condensed_history=[obs],
        initial_user_action=initial_user_message,
        max_message_chars=None,
        vision_is_active=False,
    )

    assert len(messages) == 3  # System + initial user + result
    result = messages[2]  # The actual result is now at index 2
    assert result.role == 'user'
    assert len(result.content) == 1
    assert isinstance(result.content[0], TextContent)
    assert '[Existing file /test/file.txt is edited with' in result.content[0].text


def test_process_events_with_file_read_observation(conversation_memory):
    obs = FileReadObservation(
        path='/test/file.txt',
        content='File content',
        impl_source=FileReadSource.DEFAULT,
    )

    initial_user_message = MessageAction(content='Initial user message')
    initial_user_message._source = EventSource.USER
    messages = conversation_memory.process_events(
        condensed_history=[obs],
        initial_user_action=initial_user_message,
        max_message_chars=None,
        vision_is_active=False,
    )

    assert len(messages) == 3  # System + initial user + result
    result = messages[2]  # The actual result is now at index 2
    assert result.role == 'user'
    assert len(result.content) == 1
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == '\n\nFile content'


def test_process_events_with_browser_output_observation(conversation_memory):
    formatted_content = '[Current URL: http://example.com]\n\n============== BEGIN webpage content ==============\nPage loaded\n============== END webpage content =============='

    obs = BrowserOutputObservation(
        url='http://example.com',
        trigger_by_action='browse',
        screenshot='',
        content=formatted_content,
        error=False,
    )

    initial_user_message = MessageAction(content='Initial user message')
    initial_user_message._source = EventSource.USER
    messages = conversation_memory.process_events(
        condensed_history=[obs],
        initial_user_action=initial_user_message,
        max_message_chars=None,
        vision_is_active=False,
    )

    assert len(messages) == 3  # System + initial user + result
    result = messages[2]  # The actual result is now at index 2
    assert result.role == 'user'
    assert len(result.content) == 1
    assert isinstance(result.content[0], TextContent)
    assert '[Current URL: http://example.com]' in result.content[0].text


def test_process_events_with_user_reject_observation(conversation_memory):
    obs = UserRejectObservation('Action rejected')

    initial_user_message = MessageAction(content='Initial user message')
    initial_user_message._source = EventSource.USER
    messages = conversation_memory.process_events(
        condensed_history=[obs],
        initial_user_action=initial_user_message,
        max_message_chars=None,
        vision_is_active=False,
    )

    assert len(messages) == 3  # System + initial user + result
    result = messages[2]  # The actual result is now at index 2
    assert result.role == 'user'
    assert len(result.content) == 1
    assert isinstance(result.content[0], TextContent)
    assert 'Action rejected' in result.content[0].text
    assert '[Last action has been rejected by the user]' in result.content[0].text


def test_process_events_with_empty_environment_info(conversation_memory):
    """Test that empty environment info observations return an empty list of messages without calling build_workspace_context."""
    # Create a RecallObservation with empty info

    empty_obs = RecallObservation(
        recall_type=RecallType.WORKSPACE_CONTEXT,
        repo_name='',
        repo_directory='',
        repo_instructions='',
        runtime_hosts={},
        additional_agent_instructions='',
        microagent_knowledge=[],
        content='Retrieved environment info',
    )

    initial_user_message = MessageAction(content='Initial user message')
    initial_user_message._source = EventSource.USER
    messages = conversation_memory.process_events(
        condensed_history=[empty_obs],
        initial_user_action=initial_user_message,
        max_message_chars=None,
        vision_is_active=False,
    )

    # Should only contain system message and initial user message
    assert len(messages) == 2

    # Verify that build_workspace_context was NOT called since all input values were empty
    conversation_memory.prompt_manager.build_workspace_context.assert_not_called()


def test_process_events_with_function_calling_observation(conversation_memory):
    mock_response = {
        'id': 'mock_id',
        'total_calls_in_response': 1,
        'choices': [{'message': {'content': 'Task completed'}}],
    }
    obs = CmdOutputObservation(
        command='echo hello',
        content='Command output',
        command_id=1,
        exit_code=0,
    )
    obs.tool_call_metadata = ToolCallMetadata(
        tool_call_id='123',
        function_name='execute_bash',
        model_response=mock_response,
        total_calls_in_response=1,
    )
    # Define initial user action
    initial_user_action = MessageAction(content='Initial user message')
    initial_user_action._source = EventSource.USER
    messages = conversation_memory.process_events(
        condensed_history=[obs],
        initial_user_action=initial_user_action,
        max_message_chars=None,
        vision_is_active=False,
    )

    # No direct message when using function calling
    assert (
        len(messages) == 2
    )  # should be no messages except system message and initial user message


def test_process_events_with_message_action_with_image(conversation_memory):
    action = MessageAction(
        content='Message with image',
        image_urls=['http://example.com/image.jpg'],
    )
    action._source = EventSource.AGENT

    # Define initial user action
    initial_user_action = MessageAction(content='Initial user message')
    initial_user_action._source = EventSource.USER
    messages = conversation_memory.process_events(
        condensed_history=[action],
        initial_user_action=initial_user_action,
        max_message_chars=None,
        vision_is_active=True,
    )

    assert len(messages) == 3
    result = messages[2]
    assert result.role == 'assistant'
    assert len(result.content) == 2
    assert isinstance(result.content[0], TextContent)
    assert isinstance(result.content[1], ImageContent)
    assert result.content[0].text == 'Message with image'
    assert result.content[1].image_urls == ['http://example.com/image.jpg']


def test_process_events_with_user_cmd_action(conversation_memory):
    action = CmdRunAction(command='ls -l')
    action._source = EventSource.USER

    # Define initial user action
    initial_user_action = MessageAction(content='Initial user message')
    initial_user_action._source = EventSource.USER
    messages = conversation_memory.process_events(
        condensed_history=[action],
        initial_user_action=initial_user_action,
        max_message_chars=None,
        vision_is_active=False,
    )

    assert len(messages) == 3
    result = messages[2]
    assert result.role == 'user'
    assert len(result.content) == 1
    assert isinstance(result.content[0], TextContent)
    assert 'User executed the command' in result.content[0].text
    assert 'ls -l' in result.content[0].text


def test_process_events_with_agent_finish_action_with_tool_metadata(
    conversation_memory,
):
    mock_response = {
        'id': 'mock_id',
        'total_calls_in_response': 1,
        'choices': [{'message': {'content': 'Task completed'}}],
    }

    action = AgentFinishAction(thought='Initial thought')
    action._source = EventSource.AGENT
    action.tool_call_metadata = ToolCallMetadata(
        tool_call_id='123',
        function_name='finish',
        model_response=mock_response,
        total_calls_in_response=1,
    )

    # Define initial user action
    initial_user_action = MessageAction(content='Initial user message')
    initial_user_action._source = EventSource.USER
    messages = conversation_memory.process_events(
        condensed_history=[action],
        initial_user_action=initial_user_action,
        max_message_chars=None,
        vision_is_active=False,
    )

    assert len(messages) == 3
    result = messages[2]
    assert result.role == 'assistant'
    assert len(result.content) == 1
    assert isinstance(result.content[0], TextContent)
    assert 'Initial thought\nTask completed' in result.content[0].text


def test_apply_prompt_caching(conversation_memory):
    messages = [
        Message(role='system', content=[TextContent(text='System message')]),
        Message(role='user', content=[TextContent(text='User message')]),
        Message(role='assistant', content=[TextContent(text='Assistant message')]),
        Message(role='user', content=[TextContent(text='Another user message')]),
    ]

    conversation_memory.apply_prompt_caching(messages)

    # System message is hard-coded to be cached always
    assert messages[0].content[0].cache_prompt is True
    assert messages[1].content[0].cache_prompt is False
    assert messages[2].content[0].cache_prompt is False
    # Only the last user message should have cache_prompt=True
    assert messages[3].content[0].cache_prompt is True


def test_process_events_with_environment_microagent_observation(conversation_memory):
    """Test processing a RecallObservation with ENVIRONMENT info type."""
    obs = RecallObservation(
        recall_type=RecallType.WORKSPACE_CONTEXT,
        repo_name='test-repo',
        repo_directory='/path/to/repo',
        repo_instructions='# Test Repository\nThis is a test repository.',
        runtime_hosts={'localhost': 8080},
        content='Retrieved environment info',
    )

    # Define initial user action
    initial_user_action = MessageAction(content='Initial user message')
    initial_user_action._source = EventSource.USER
    messages = conversation_memory.process_events(
        condensed_history=[obs],
        initial_user_action=initial_user_action,
        max_message_chars=None,
        vision_is_active=False,
    )

    assert len(messages) == 3
    result = messages[2]
    assert result.role == 'user'
    assert len(result.content) == 1
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == '\n\nFormatted repository and runtime info'

    # Verify the prompt_manager was called with the correct parameters
    conversation_memory.prompt_manager.build_workspace_context.assert_called_once()
    call_args = conversation_memory.prompt_manager.build_workspace_context.call_args[1]
    assert isinstance(call_args['repository_info'], RepositoryInfo)
    assert call_args['repository_info'].repo_name == 'test-repo'
    assert call_args['repository_info'].repo_directory == '/path/to/repo'
    assert isinstance(call_args['runtime_info'], RuntimeInfo)
    assert call_args['runtime_info'].available_hosts == {'localhost': 8080}
    assert (
        call_args['repo_instructions']
        == '# Test Repository\nThis is a test repository.'
    )


def test_process_events_with_knowledge_microagent_microagent_observation(
    conversation_memory,
):
    """Test processing a RecallObservation with KNOWLEDGE type."""
    microagent_knowledge = [
        MicroagentKnowledge(
            name='test_agent',
            trigger='test',
            content='This is test agent content',
        ),
        MicroagentKnowledge(
            name='another_agent',
            trigger='another',
            content='This is another agent content',
        ),
        MicroagentKnowledge(
            name='disabled_agent',
            trigger='disabled',
            content='This is disabled agent content',
        ),
    ]

    obs = RecallObservation(
        recall_type=RecallType.KNOWLEDGE,
        microagent_knowledge=microagent_knowledge,
        content='Retrieved knowledge from microagents',
    )

    # Define initial user action
    initial_user_action = MessageAction(content='Initial user message')
    initial_user_action._source = EventSource.USER
    messages = conversation_memory.process_events(
        condensed_history=[obs],
        initial_user_action=initial_user_action,
        max_message_chars=None,
        vision_is_active=False,
    )

    assert len(messages) == 3  # System + Initial User + Result
    result = messages[2]  # Result is now at index 2
    assert result.role == 'user'
    assert len(result.content) == 1
    assert isinstance(result.content[0], TextContent)
    # Verify that disabled_agent is filtered out and enabled agents are included
    assert 'This is test agent content' in result.content[0].text
    assert 'This is another agent content' in result.content[0].text
    assert 'This is disabled agent content' not in result.content[0].text

    # Verify the prompt_manager was called with the correct parameters
    conversation_memory.prompt_manager.build_microagent_info.assert_called_once()
    call_args = conversation_memory.prompt_manager.build_microagent_info.call_args[1]

    # Check that disabled_agent was filtered out
    triggered_agents = call_args['triggered_agents']
    assert len(triggered_agents) == 2
    agent_names = [agent.name for agent in triggered_agents]
    assert 'test_agent' in agent_names
    assert 'another_agent' in agent_names
    assert 'disabled_agent' not in agent_names


def test_process_events_with_microagent_observation_extensions_disabled(
    agent_config, conversation_memory
):
    """Test processing a RecallObservation when prompt extensions are disabled."""
    # Modify the agent config to disable prompt extensions
    agent_config.enable_prompt_extensions = False

    obs = RecallObservation(
        recall_type=RecallType.WORKSPACE_CONTEXT,
        repo_name='test-repo',
        repo_directory='/path/to/repo',
        content='Retrieved environment info',
    )

    # Define initial user action
    initial_user_action = MessageAction(content='Initial user message')
    initial_user_action._source = EventSource.USER
    messages = conversation_memory.process_events(
        condensed_history=[obs],
        initial_user_action=initial_user_action,
        max_message_chars=None,
        vision_is_active=False,
    )

    # When prompt extensions are disabled, the RecallObservation should be ignored
    assert len(messages) == 2  # System + Initial User

    # Verify the prompt_manager was not called
    conversation_memory.prompt_manager.build_workspace_context.assert_not_called()
    conversation_memory.prompt_manager.build_microagent_info.assert_not_called()


def test_process_events_with_empty_microagent_knowledge(conversation_memory):
    """Test processing a RecallObservation with empty microagent knowledge."""
    obs = RecallObservation(
        recall_type=RecallType.KNOWLEDGE,
        microagent_knowledge=[],
        content='Retrieved knowledge from microagents',
    )

    # Define initial user action
    initial_user_action = MessageAction(content='Initial user message')
    initial_user_action._source = EventSource.USER
    messages = conversation_memory.process_events(
        condensed_history=[obs],
        initial_user_action=initial_user_action,
        max_message_chars=None,
        vision_is_active=False,
    )

    # The implementation returns an empty string and it doesn't creates a message
    assert len(messages) == 2  # System + Initial User

    # When there are no triggered agents, build_microagent_info is not called
    conversation_memory.prompt_manager.build_microagent_info.assert_not_called()


def test_conversation_memory_processes_microagent_observation(prompt_dir):
    """Test that ConversationMemory processes RecallObservations correctly."""
    # Create a microagent_info.j2 template file
    template_path = os.path.join(prompt_dir, 'microagent_info.j2')
    if not os.path.exists(template_path):
        with open(template_path, 'w') as f:
            f.write("""{% for agent_info in triggered_agents %}
<EXTRA_INFO>
The following information has been included based on a keyword match for "{{ agent_info.trigger_word }}".
It may or may not be relevant to the user's request.

    # Verify the template was correctly rendered
{{ agent_info.content }}
</EXTRA_INFO>
{% endfor %}
""")

    # Create a mock agent config
    agent_config = MagicMock(spec=AgentConfig)
    agent_config.enable_prompt_extensions = True
    agent_config.disabled_microagents = []

    # Create a PromptManager
    prompt_manager = PromptManager(prompt_dir=prompt_dir)

    # Initialize ConversationMemory
    conversation_memory = ConversationMemory(
        config=agent_config, prompt_manager=prompt_manager
    )

    # Create a RecallObservation with microagent knowledge
    microagent_observation = RecallObservation(
        recall_type=RecallType.KNOWLEDGE,
        microagent_knowledge=[
            MicroagentKnowledge(
                name='test_agent',
                trigger='test_trigger',
                content='This is triggered content for testing.',
            )
        ],
        content='Retrieved knowledge from microagents',
    )

    # Process the observation
    messages = conversation_memory._process_observation(
        obs=microagent_observation, tool_call_id_to_message={}, max_message_chars=None
    )

    # Verify the message was created correctly
    assert len(messages) == 1
    message = messages[0]
    assert message.role == 'user'
    assert len(message.content) == 1
    assert isinstance(message.content[0], TextContent)

    expected_text = """<EXTRA_INFO>
The following information has been included based on a keyword match for "test_trigger".
It may or may not be relevant to the user's request.

This is triggered content for testing.
</EXTRA_INFO>"""

    assert message.content[0].text.strip() == expected_text.strip()

    # Clean up
    os.remove(os.path.join(prompt_dir, 'microagent_info.j2'))


def test_conversation_memory_processes_environment_microagent_observation(prompt_dir):
    """Test that ConversationMemory processes environment info RecallObservations correctly."""
    # Create an additional_info.j2 template file
    template_path = os.path.join(prompt_dir, 'additional_info.j2')
    if not os.path.exists(template_path):
        with open(template_path, 'w') as f:
            f.write("""
{% if repository_info %}
<REPOSITORY_INFO>
At the user's request, repository {{ repository_info.repo_name }} has been cloned to directory {{ repository_info.repo_directory }}.
</REPOSITORY_INFO>
{% endif %}

{% if repository_instructions %}
<REPOSITORY_INSTRUCTIONS>
{{ repository_instructions }}
</REPOSITORY_INSTRUCTIONS>
{% endif %}

{% if runtime_info and runtime_info.available_hosts %}
<RUNTIME_INFORMATION>
The user has access to the following hosts for accessing a web application,
each of which has a corresponding port:
{% for host, port in runtime_info.available_hosts.items() %}
* {{ host }} (port {{ port }})
{% endfor %}
</RUNTIME_INFORMATION>
{% endif %}
""")

    # Create a mock agent config
    agent_config = MagicMock(spec=AgentConfig)
    agent_config.enable_prompt_extensions = True

    # Create a PromptManager
    prompt_manager = PromptManager(prompt_dir=prompt_dir)

    # Initialize ConversationMemory
    conversation_memory = ConversationMemory(
        config=agent_config, prompt_manager=prompt_manager
    )

    # Create a RecallObservation with environment info
    microagent_observation = RecallObservation(
        recall_type=RecallType.WORKSPACE_CONTEXT,
        repo_name='owner/repo',
        repo_directory='/workspace/repo',
        repo_instructions='This repository contains important code.',
        runtime_hosts={'example.com': 8080},
        content='Retrieved environment info',
    )

    # Process the observation
    messages = conversation_memory._process_observation(
        obs=microagent_observation, tool_call_id_to_message={}, max_message_chars=None
    )

    # Verify the message was created correctly
    assert len(messages) == 1
    message = messages[0]
    assert message.role == 'user'
    assert len(message.content) == 1
    assert isinstance(message.content[0], TextContent)

    # Check that the message contains the repository info
    assert '<REPOSITORY_INFO>' in message.content[0].text
    assert 'owner/repo' in message.content[0].text
    assert '/workspace/repo' in message.content[0].text

    # Check that the message contains the repository instructions
    assert '<REPOSITORY_INSTRUCTIONS>' in message.content[0].text
    assert 'This repository contains important code.' in message.content[0].text

    # Check that the message contains the runtime info
    assert '<RUNTIME_INFORMATION>' in message.content[0].text
    assert 'example.com (port 8080)' in message.content[0].text


def test_process_events_with_microagent_observation_deduplication(conversation_memory):
    """Test that RecallObservations are properly deduplicated based on agent name.

    The deduplication logic should keep the FIRST occurrence of each microagent
    and filter out later occurrences to avoid redundant information.
    """
    # Create a sequence of RecallObservations with overlapping agents
    obs1 = RecallObservation(
        recall_type=RecallType.KNOWLEDGE,
        microagent_knowledge=[
            MicroagentKnowledge(
                name='python_agent',
                trigger='python',
                content='Python best practices v1',
            ),
            MicroagentKnowledge(
                name='git_agent',
                trigger='git',
                content='Git best practices v1',
            ),
            MicroagentKnowledge(
                name='image_agent',
                trigger='image',
                content='Image best practices v1',
            ),
        ],
        content='First retrieval',
    )

    obs2 = RecallObservation(
        recall_type=RecallType.KNOWLEDGE,
        microagent_knowledge=[
            MicroagentKnowledge(
                name='python_agent',
                trigger='python',
                content='Python best practices v2',
            ),
        ],
        content='Second retrieval',
    )

    obs3 = RecallObservation(
        recall_type=RecallType.KNOWLEDGE,
        microagent_knowledge=[
            MicroagentKnowledge(
                name='git_agent',
                trigger='git',
                content='Git best practices v3',
            ),
        ],
        content='Third retrieval',
    )

    # Define initial user action
    initial_user_action = MessageAction(content='Initial user message')
    initial_user_action._source = EventSource.USER
    messages = conversation_memory.process_events(
        condensed_history=[obs1, obs2, obs3],
        initial_user_action=initial_user_action,
        max_message_chars=None,
        vision_is_active=False,
    )

    # Verify that only the first occurrence of content for each agent is included
    assert len(messages) == 3  # System + Initial User + Result
    # Result is now at index 2
    # First microagent should include all agents since they appear here first
    assert 'Image best practices v1' in messages[2].content[0].text
    assert 'Git best practices v1' in messages[2].content[0].text
    assert 'Python best practices v1' in messages[2].content[0].text


def test_process_events_with_microagent_observation_deduplication_disabled_agents(
    conversation_memory,
):
    """Test that disabled agents are filtered out and deduplication keeps the first occurrence."""
    # Create a sequence of RecallObservations with disabled agents
    obs1 = RecallObservation(
        recall_type=RecallType.KNOWLEDGE,
        microagent_knowledge=[
            MicroagentKnowledge(
                name='disabled_agent',
                trigger='disabled',
                content='Disabled agent content',
            ),
            MicroagentKnowledge(
                name='enabled_agent',
                trigger='enabled',
                content='Enabled agent content v1',
            ),
        ],
        content='First retrieval',
    )

    obs2 = RecallObservation(
        recall_type=RecallType.KNOWLEDGE,
        microagent_knowledge=[
            MicroagentKnowledge(
                name='enabled_agent',
                trigger='enabled',
                content='Enabled agent content v2',
            ),
        ],
        content='Second retrieval',
    )

    # Define initial user action
    initial_user_action = MessageAction(content='Initial user message')
    initial_user_action._source = EventSource.USER
    messages = conversation_memory.process_events(
        condensed_history=[obs1, obs2],
        initial_user_action=initial_user_action,
        max_message_chars=None,
        vision_is_active=False,
    )

    # Verify that disabled agents are filtered out and only the first occurrence of enabled agents is included
    assert len(messages) == 3  # System + Initial User + Result
    # Result is now at index 2
    # First microagent should include enabled_agent but not disabled_agent
    assert 'Disabled agent content' not in messages[2].content[0].text
    assert 'Enabled agent content v1' in messages[2].content[0].text


def test_process_events_with_microagent_observation_deduplication_empty(
    conversation_memory,
):
    """Test that empty RecallObservations are handled correctly."""
    obs = RecallObservation(
        recall_type=RecallType.KNOWLEDGE,
        microagent_knowledge=[],
        content='Empty retrieval',
    )

    # Define initial user action
    initial_user_action = MessageAction(content='Initial user message')
    initial_user_action._source = EventSource.USER
    messages = conversation_memory.process_events(
        condensed_history=[obs],
        initial_user_action=initial_user_action,
        max_message_chars=None,
        vision_is_active=False,
    )

    # Verify that empty RecallObservations are handled gracefully
    assert (
        len(messages) == 2  # System + Initial User
    )  # an empty microagent is not added to Messages
    assert messages[0].role == 'system'
    assert messages[1].role == 'user'  # Initial user message


def test_has_agent_in_earlier_events(conversation_memory):
    """Test the _has_agent_in_earlier_events helper method."""
    # Create test RecallObservations
    obs1 = RecallObservation(
        recall_type=RecallType.KNOWLEDGE,
        microagent_knowledge=[
            MicroagentKnowledge(
                name='agent1',
                trigger='trigger1',
                content='Content 1',
            ),
        ],
        content='First retrieval',
    )

    obs2 = RecallObservation(
        recall_type=RecallType.KNOWLEDGE,
        microagent_knowledge=[
            MicroagentKnowledge(
                name='agent2',
                trigger='trigger2',
                content='Content 2',
            ),
        ],
        content='Second retrieval',
    )

    obs3 = RecallObservation(
        recall_type=RecallType.WORKSPACE_CONTEXT,
        content='Environment info',
    )

    # Create a list with mixed event types
    events = [obs1, MessageAction(content='User message'), obs2, obs3]

    # Test looking for existing agents
    assert conversation_memory._has_agent_in_earlier_events('agent1', 2, events) is True
    assert conversation_memory._has_agent_in_earlier_events('agent1', 3, events) is True
    assert conversation_memory._has_agent_in_earlier_events('agent1', 4, events) is True

    # Test looking for an agent in a later position (should not find it)
    assert (
        conversation_memory._has_agent_in_earlier_events('agent2', 0, events) is False
    )
    assert (
        conversation_memory._has_agent_in_earlier_events('agent2', 1, events) is False
    )

    # Test looking for an agent in a different microagent type (should not find it)
    assert (
        conversation_memory._has_agent_in_earlier_events('non_existent', 3, events)
        is False
    )


class TestFilterUnmatchedToolCalls:
    @pytest.fixture
    def processor(self):
        return ConversationMemory()

    def test_empty_is_unchanged(self):
        assert list(ConversationMemory._filter_unmatched_tool_calls([])) == []

    def test_no_tool_calls_is_unchanged(self):
        messages = [
            Message(role='user', content=[TextContent(text='Hello')]),
            Message(role='assistant', content=[TextContent(text='Hi there')]),
            Message(role='user', content=[TextContent(text='How are you?')]),
        ]
        assert (
            list(ConversationMemory._filter_unmatched_tool_calls(messages)) == messages
        )

    def test_matched_tool_calls_are_unchanged(self):
        messages = [
            Message(role='user', content=[TextContent(text="What's the weather?")]),
            Message(
                role='assistant',
                content=[],
                tool_calls=[
                    ChatCompletionMessageToolCall(
                        id='call_1',
                        type='function',
                        function={'name': 'get_weather', 'arguments': ''},
                    )
                ],
            ),
            Message(
                role='tool',
                tool_call_id='call_1',
                content=[TextContent(text='Sunny, 75°F')],
            ),
            Message(role='assistant', content=[TextContent(text="It's sunny today.")]),
        ]

        # All tool calls have matching responses, should remain unchanged
        assert (
            list(ConversationMemory._filter_unmatched_tool_calls(messages)) == messages
        )

    def test_tool_call_without_response_is_removed(self):
        messages = [
            Message(role='user', content=[TextContent(text='Query')]),
            Message(
                role='tool',
                tool_call_id='missing_call',
                content=[TextContent(text='Response')],
            ),
            Message(role='assistant', content=[TextContent(text='Answer')]),
        ]

        expected_after_filter = [
            Message(role='user', content=[TextContent(text='Query')]),
            Message(role='assistant', content=[TextContent(text='Answer')]),
        ]

        result = list(ConversationMemory._filter_unmatched_tool_calls(messages))
        assert result == expected_after_filter

    def test_tool_response_without_call_is_removed(self):
        messages = [
            Message(role='user', content=[TextContent(text='Query')]),
            Message(
                role='assistant',
                content=[],
                tool_calls=[
                    ChatCompletionMessageToolCall(
                        id='unmatched_call',
                        type='function',
                        function={'name': 'some_function', 'arguments': ''},
                    )
                ],
            ),
            Message(role='assistant', content=[TextContent(text='Answer')]),
        ]

        expected_after_filter = [
            Message(role='user', content=[TextContent(text='Query')]),
            Message(role='assistant', content=[TextContent(text='Answer')]),
        ]

        result = list(ConversationMemory._filter_unmatched_tool_calls(messages))
        assert result == expected_after_filter

    def test_partial_matched_tool_calls_retains_matched(self):
        """When there are both matched and unmatched tools calls in a message, retain the message and only matched calls"""
        messages = [
            Message(role='user', content=[TextContent(text='Get data')]),
            Message(
                role='assistant',
                content=[],
                tool_calls=[
                    ChatCompletionMessageToolCall(
                        id='matched_call',
                        type='function',
                        function={'name': 'function1', 'arguments': ''},
                    ),
                    ChatCompletionMessageToolCall(
                        id='unmatched_call',
                        type='function',
                        function={'name': 'function2', 'arguments': ''},
                    ),
                ],
            ),
            Message(
                role='tool',
                tool_call_id='matched_call',
                content=[TextContent(text='Data')],
            ),
            Message(role='assistant', content=[TextContent(text='Result')]),
        ]

        expected = [
            Message(role='user', content=[TextContent(text='Get data')]),
            # This message should be modified to only include the matched tool call
            Message(
                role='assistant',
                content=[],
                tool_calls=[
                    ChatCompletionMessageToolCall(
                        id='matched_call',
                        type='function',
                        function={'name': 'function1', 'arguments': ''},
                    )
                ],
            ),
            Message(
                role='tool',
                tool_call_id='matched_call',
                content=[TextContent(text='Data')],
            ),
            Message(role='assistant', content=[TextContent(text='Result')]),
        ]

        result = list(ConversationMemory._filter_unmatched_tool_calls(messages))

        # Verify result structure
        assert len(result) == len(expected)
        for i, msg in enumerate(result):
            assert msg == expected[i]


def test_system_message_in_events(conversation_memory):
    """Test that SystemMessageAction in condensed_history is processed correctly."""
    # Create a system message action
    system_message = SystemMessageAction(content='System message', tools=['test_tool'])
    system_message._source = EventSource.AGENT

    # Process events with the system message in condensed_history
    # Define initial user action
    initial_user_action = MessageAction(content='Initial user message')
    initial_user_action._source = EventSource.USER
    messages = conversation_memory.process_events(
        condensed_history=[system_message],
        initial_user_action=initial_user_action,
        max_message_chars=None,
        vision_is_active=False,
    )

    # Check that the system message was processed correctly
    assert len(messages) == 2  # System + Initial User
    assert messages[0].role == 'system'
    assert messages[0].content[0].text == 'System message'
    assert messages[1].role == 'user'  # Initial user message


# Helper function to create mock tool call metadata
def _create_mock_tool_call_metadata(
    tool_call_id: str, function_name: str, response_id: str = 'mock_response_id'
) -> ToolCallMetadata:
    # Use a dictionary that mimics ModelResponse structure to satisfy Pydantic
    mock_response = {
        'id': response_id,
        'choices': [
            {
                'message': {
                    'role': 'assistant',
                    'content': None,  # Content is None for tool calls
                    'tool_calls': [
                        {
                            'id': tool_call_id,
                            'type': 'function',
                            'function': {
                                'name': function_name,
                                'arguments': '{}',
                            },  # Args don't matter for this test
                        }
                    ],
                }
            }
        ],
        'created': 0,
        'model': 'mock_model',
        'object': 'chat.completion',
        'usage': {'completion_tokens': 0, 'prompt_tokens': 0, 'total_tokens': 0},
    }
    return ToolCallMetadata(
        tool_call_id=tool_call_id,
        function_name=function_name,
        model_response=mock_response,
        total_calls_in_response=1,
    )


def test_process_events_partial_history(conversation_memory):
    """Tests process_events with full and partial histories to verify
    _ensure_system_message, _ensure_initial_user_message, and tool call matching logic.
    """
    # --- Define Common Events ---
    system_message = SystemMessageAction(content='System message')
    system_message._source = EventSource.AGENT

    user_message = MessageAction(
        content='Initial user query'
    )  # This is the crucial initial_user_action
    user_message._source = EventSource.USER

    recall_obs = RecallObservation(
        recall_type=RecallType.WORKSPACE_CONTEXT,
        repo_name='test-repo',
        repo_directory='/path/to/repo',
        content='Retrieved environment info',
    )
    recall_obs._source = EventSource.AGENT

    cmd_action = CmdRunAction(command='ls', thought='Running ls')
    cmd_action._source = EventSource.AGENT
    cmd_action.tool_call_metadata = _create_mock_tool_call_metadata(
        tool_call_id='call_ls_1', function_name='execute_bash', response_id='resp_ls_1'
    )

    cmd_obs = CmdOutputObservation(
        command_id=1, command='ls', content='file1.txt\nfile2.py', exit_code=0
    )
    cmd_obs._source = EventSource.AGENT
    cmd_obs.tool_call_metadata = _create_mock_tool_call_metadata(
        tool_call_id='call_ls_1', function_name='execute_bash', response_id='resp_ls_1'
    )

    # --- Scenario 1: Full History ---
    full_history: list[Event] = [
        system_message,
        user_message,  # Correct initial user message at index 1
        recall_obs,
        cmd_action,
        cmd_obs,
    ]
    messages_full = conversation_memory.process_events(
        condensed_history=list(full_history),  # Pass a copy
        initial_user_action=user_message,  # Provide the initial action
        max_message_chars=None,
        vision_is_active=False,
    )

    # Expected: System, User, Recall (formatted), Assistant (tool call), Tool Response
    assert len(messages_full) == 5
    assert messages_full[0].role == 'system'
    assert messages_full[0].content[0].text == 'System message'
    assert messages_full[1].role == 'user'
    assert messages_full[1].content[0].text == 'Initial user query'
    assert messages_full[2].role == 'user'  # Recall obs becomes user message
    assert (
        'Formatted repository and runtime info' in messages_full[2].content[0].text
    )  # From fixture mock
    assert messages_full[3].role == 'assistant'
    assert messages_full[3].tool_calls is not None
    assert len(messages_full[3].tool_calls) == 1
    assert messages_full[3].tool_calls[0].id == 'call_ls_1'
    assert messages_full[4].role == 'tool'
    assert messages_full[4].tool_call_id == 'call_ls_1'
    assert 'file1.txt' in messages_full[4].content[0].text

    # --- Scenario 2: Partial History (Action + Observation) ---
    # Simulates processing only the last action/observation pair
    partial_history_action_obs: list[Event] = [
        cmd_action,
        cmd_obs,
    ]
    messages_partial_action_obs = conversation_memory.process_events(
        condensed_history=list(partial_history_action_obs),  # Pass a copy
        initial_user_action=user_message,  # Provide the initial action
        max_message_chars=None,
        vision_is_active=False,
    )

    # Expected: System (added), Initial User (added), Assistant (tool call), Tool Response
    assert len(messages_partial_action_obs) == 4
    assert (
        messages_partial_action_obs[0].role == 'system'
    )  # Added by _ensure_system_message
    assert messages_partial_action_obs[0].content[0].text == 'System message'
    assert (
        messages_partial_action_obs[1].role == 'user'
    )  # Added by _ensure_initial_user_message
    assert messages_partial_action_obs[1].content[0].text == 'Initial user query'
    assert messages_partial_action_obs[2].role == 'assistant'
    assert messages_partial_action_obs[2].tool_calls is not None
    assert len(messages_partial_action_obs[2].tool_calls) == 1
    assert messages_partial_action_obs[2].tool_calls[0].id == 'call_ls_1'
    assert messages_partial_action_obs[3].role == 'tool'
    assert messages_partial_action_obs[3].tool_call_id == 'call_ls_1'
    assert 'file1.txt' in messages_partial_action_obs[3].content[0].text

    # --- Scenario 3: Partial History (Observation Only) ---
    # Simulates processing only the last observation
    partial_history_obs_only: list[Event] = [
        cmd_obs,
    ]
    messages_partial_obs_only = conversation_memory.process_events(
        condensed_history=list(partial_history_obs_only),  # Pass a copy
        initial_user_action=user_message,  # Provide the initial action
        max_message_chars=None,
        vision_is_active=False,
    )

    # Expected: System (added), Initial User (added).
    # The CmdOutputObservation has tool_call_metadata, but there's no corresponding
    # assistant message (from CmdRunAction) with the matching tool_call.id in the input history.
    # Therefore, _filter_unmatched_tool_calls should remove the tool response message.
    assert len(messages_partial_obs_only) == 2
    assert (
        messages_partial_obs_only[0].role == 'system'
    )  # Added by _ensure_system_message
    assert messages_partial_obs_only[0].content[0].text == 'System message'
    assert (
        messages_partial_obs_only[1].role == 'user'
    )  # Added by _ensure_initial_user_message
    assert messages_partial_obs_only[1].content[0].text == 'Initial user query'


def test_process_ipython_observation_with_vision_enabled(
    agent_config, mock_prompt_manager
):
    """Test that _process_observation correctly handles IPythonRunCellObservation with image_urls when vision is enabled."""
    # Create a ConversationMemory instance
    memory = ConversationMemory(agent_config, mock_prompt_manager)

    # Create an observation with image URLs
    obs = IPythonRunCellObservation(
        content='Test output',
        code="print('test')",
        image_urls=['data:image/png;base64,abc123'],
    )

    # Process the observation with vision enabled
    messages = memory._process_observation(
        obs=obs,
        tool_call_id_to_message={},
        max_message_chars=None,
        vision_is_active=True,
    )

    # Check that the message contains both text and image content
    assert len(messages) == 1
    message = messages[0]
    assert len(message.content) == 2
    assert isinstance(message.content[0], TextContent)
    assert isinstance(message.content[1], ImageContent)
    assert message.content[1].image_urls == ['data:image/png;base64,abc123']


def test_process_ipython_observation_with_vision_disabled(
    agent_config, mock_prompt_manager
):
    """Test that _process_observation correctly handles IPythonRunCellObservation with image_urls when vision is disabled."""
    # Create a ConversationMemory instance
    memory = ConversationMemory(agent_config, mock_prompt_manager)

    # Create an observation with image URLs
    obs = IPythonRunCellObservation(
        content='Test output',
        code="print('test')",
        image_urls=['data:image/png;base64,abc123'],
    )

    # Process the observation with vision disabled
    messages = memory._process_observation(
        obs=obs,
        tool_call_id_to_message={},
        max_message_chars=None,
        vision_is_active=False,
    )

    # Check that the message contains both text and image content
    # (ImageContent is always included, filtering happens at Message serialization level)
    assert len(messages) == 1
    message = messages[0]
    assert len(message.content) == 2
    assert isinstance(message.content[0], TextContent)
    assert isinstance(message.content[1], ImageContent)
    # Check that NO explanatory text about filtered images was added when vision is disabled
    assert 'invalid or empty image(s) were filtered' not in message.content[0].text


def test_ensure_initial_user_message_not_reinserted_when_condensed(
    conversation_memory, initial_user_action
):
    """Test that initial user message is NOT re-inserted when it has been condensed.

    This is a critical test for bug #11910: Old instructions should not be re-executed
    after conversation condensation. If the initial user message has been condensed
    (its ID is in the forgotten_event_ids set), we should NOT re-insert it to prevent
    the LLM from seeing old instructions as fresh commands.
    """
    system_message = SystemMessageAction(content='System')
    system_message._source = EventSource.AGENT

    # Simulate that the initial_user_action has been condensed by adding its ID
    # to the forgotten_event_ids set
    initial_user_action._id = 1  # Assign an ID to the initial user action
    forgotten_event_ids = {1}  # The initial user action's ID is in the forgotten set

    events = [system_message]  # Only system message, no user message

    # Call _ensure_initial_user_message with the condensed event ID
    conversation_memory._ensure_initial_user_message(
        events, initial_user_action, forgotten_event_ids
    )

    # The initial user action should NOT be inserted because it was condensed
    assert len(events) == 1
    assert events[0] == system_message
    # Verify the initial user action was NOT added
    assert initial_user_action not in events


def test_ensure_initial_user_message_reinserted_when_not_condensed(
    conversation_memory, initial_user_action
):
    """Test that initial user message IS re-inserted when it has NOT been condensed.

    This ensures backward compatibility: when no condensation has happened,
    the initial user message should still be inserted as before.
    """
    system_message = SystemMessageAction(content='System')
    system_message._source = EventSource.AGENT

    # The initial user action has NOT been condensed
    initial_user_action._id = 1
    forgotten_event_ids = {5, 10, 15}  # Different IDs, not including the initial action

    events = [system_message]

    # Call _ensure_initial_user_message with non-matching forgotten IDs
    conversation_memory._ensure_initial_user_message(
        events, initial_user_action, forgotten_event_ids
    )

    # The initial user action SHOULD be inserted because it was NOT condensed
    assert len(events) == 2
    assert events[0] == system_message
    assert events[1] == initial_user_action


def test_process_events_does_not_reinsert_condensed_initial_message(
    conversation_memory,
):
    """Test that process_events does not re-insert initial user message when condensed.

    This is an integration test for the full process_events flow, verifying that
    when the initial user message has been condensed, it is not re-inserted into
    the conversation sent to the LLM.
    """
    # Create a system message
    system_message = SystemMessageAction(content='System message')
    system_message._source = EventSource.AGENT
    system_message._id = 0

    # Create the initial user message (will be marked as condensed)
    initial_user_message = MessageAction(content='Do task A, B, and C')
    initial_user_message._source = EventSource.USER
    initial_user_message._id = 1

    # Create a condensation summary observation
    from openhands.events.observation.agent import AgentCondensationObservation

    condensation_summary = AgentCondensationObservation(
        content='Summary: User requested tasks A, B, C. Task A was completed successfully.'
    )
    condensation_summary._id = 2

    # Create a recent user message (not condensed)
    recent_user_message = MessageAction(content='Now continue with task D')
    recent_user_message._source = EventSource.USER
    recent_user_message._id = 3

    # Simulate condensed history: system + summary + recent message
    # The initial user message (id=1) has been condensed/forgotten
    condensed_history = [system_message, condensation_summary, recent_user_message]

    # The initial user message's ID is in the forgotten set
    forgotten_event_ids = {1}

    messages = conversation_memory.process_events(
        condensed_history=condensed_history,
        initial_user_action=initial_user_message,
        forgotten_event_ids=forgotten_event_ids,
        max_message_chars=None,
        vision_is_active=False,
    )

    # Verify the structure of messages
    # Should have: system, condensation summary, recent user message
    # Should NOT have the initial user message "Do task A, B, and C"
    assert len(messages) == 3
    assert messages[0].role == 'system'
    assert messages[0].content[0].text == 'System message'

    # The second message should be the condensation summary, NOT the initial user message
    assert messages[1].role == 'user'
    assert 'Summary: User requested tasks A, B, C' in messages[1].content[0].text

    # The third message should be the recent user message
    assert messages[2].role == 'user'
    assert 'Now continue with task D' in messages[2].content[0].text

    # Critically, the old instruction should NOT appear
    for msg in messages:
        for content in msg.content:
            if hasattr(content, 'text'):
                assert 'Do task A, B, and C' not in content.text
