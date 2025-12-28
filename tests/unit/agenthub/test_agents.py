from typing import Union
from unittest.mock import Mock

import pytest
from litellm import ChatCompletionMessageToolCall

from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.agenthub.codeact_agent.function_calling import (
    response_to_actions as codeact_response_to_actions,
)
from openhands.agenthub.codeact_agent.tools import (
    BrowserTool,
    IPythonTool,
    LLMBasedFileEditTool,
    ThinkTool,
    create_cmd_run_tool,
    create_str_replace_editor_tool,
)
from openhands.agenthub.codeact_agent.tools.browser import (
    _BROWSER_DESCRIPTION,
    _BROWSER_TOOL_DESCRIPTION,
)
from openhands.agenthub.readonly_agent.function_calling import (
    response_to_actions as readonly_response_to_actions,
)
from openhands.agenthub.readonly_agent.readonly_agent import ReadOnlyAgent
from openhands.agenthub.readonly_agent.tools import (
    GlobTool,
    GrepTool,
)
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig, LLMConfig
from openhands.core.config.openhands_config import OpenHandsConfig
from openhands.core.exceptions import FunctionCallNotExistsError
from openhands.core.message import ImageContent, Message, TextContent
from openhands.events.action import (
    CmdRunAction,
    MessageAction,
)
from openhands.events.action.message import SystemMessageAction
from openhands.events.event import EventSource
from openhands.events.observation.commands import (
    CmdOutputObservation,
)
from openhands.events.tool import ToolCallMetadata
from openhands.llm.llm_registry import LLMRegistry
from openhands.memory.condenser import View


@pytest.fixture
def create_llm_registry():
    def _get_registry(llm_config):
        config = OpenHandsConfig()
        config.set_llm_config(llm_config)
        return LLMRegistry(config=config)

    return _get_registry


@pytest.fixture(params=['CodeActAgent', 'ReadOnlyAgent'])
def agent_class(request):
    if request.param == 'CodeActAgent':
        return CodeActAgent
    else:
        from openhands.agenthub.readonly_agent.readonly_agent import ReadOnlyAgent

        return ReadOnlyAgent


@pytest.fixture
def agent(agent_class, create_llm_registry) -> Union[CodeActAgent, ReadOnlyAgent]:
    llm_config = LLMConfig(model='gpt-4o', api_key='test_key')
    config = AgentConfig()
    agent = agent_class(config=config, llm_registry=create_llm_registry(llm_config))
    agent.llm = Mock()
    agent.llm.config = Mock()
    agent.llm.config.max_message_chars = 1000
    return agent


def test_agent_with_default_config_has_default_tools(create_llm_registry):
    llm_config = LLMConfig(model='gpt-4o', api_key='test_key')
    config = AgentConfig()
    codeact_agent = CodeActAgent(
        config=config, llm_registry=create_llm_registry(llm_config)
    )
    assert len(codeact_agent.tools) > 0
    default_tool_names = [tool['function']['name'] for tool in codeact_agent.tools]
    assert {
        'browser',
        'execute_bash',
        'execute_ipython_cell',
        'finish',
        'str_replace_editor',
        'think',
    }.issubset(default_tool_names)


@pytest.fixture
def mock_state() -> State:
    state = Mock(spec=State)
    state.history = []
    state.extra_data = {}

    return state


def test_reset(agent):
    # Add some state
    action = MessageAction(content='test')
    action._source = EventSource.AGENT
    agent.pending_actions.append(action)

    # Create a mock state with initial user message
    mock_state = Mock(spec=State)
    initial_user_message = MessageAction(content='Initial user message')
    initial_user_message._source = EventSource.USER
    mock_state.history = [initial_user_message]

    # Reset
    agent.reset()

    # Verify state is cleared
    assert len(agent.pending_actions) == 0


def test_step_with_pending_actions(agent):
    # Add a pending action
    pending_action = MessageAction(content='test')
    pending_action._source = EventSource.AGENT
    agent.pending_actions.append(pending_action)

    # Create a mock state with initial user message
    mock_state = Mock(spec=State)
    initial_user_message = MessageAction(content='Initial user message')
    initial_user_message._source = EventSource.USER
    mock_state.history = [initial_user_message]

    # Step should return the pending action
    result = agent.step(mock_state)
    assert result == pending_action
    assert len(agent.pending_actions) == 0


def test_cmd_run_tool():
    CmdRunTool = create_cmd_run_tool()
    assert CmdRunTool['type'] == 'function'
    assert CmdRunTool['function']['name'] == 'execute_bash'
    assert 'command' in CmdRunTool['function']['parameters']['properties']
    assert 'security_risk' in CmdRunTool['function']['parameters']['properties']
    assert CmdRunTool['function']['parameters']['required'] == [
        'command',
        'security_risk',
    ]


def test_ipython_tool():
    assert IPythonTool['type'] == 'function'
    assert IPythonTool['function']['name'] == 'execute_ipython_cell'
    assert 'code' in IPythonTool['function']['parameters']['properties']
    assert 'security_risk' in IPythonTool['function']['parameters']['properties']
    assert IPythonTool['function']['parameters']['required'] == [
        'code',
        'security_risk',
    ]


def test_llm_based_file_edit_tool():
    assert LLMBasedFileEditTool['type'] == 'function'
    assert LLMBasedFileEditTool['function']['name'] == 'edit_file'

    properties = LLMBasedFileEditTool['function']['parameters']['properties']
    assert 'path' in properties
    assert 'content' in properties
    assert 'start' in properties
    assert 'end' in properties
    assert 'security_risk' in properties

    assert LLMBasedFileEditTool['function']['parameters']['required'] == [
        'path',
        'content',
        'security_risk',
    ]


def test_str_replace_editor_tool():
    StrReplaceEditorTool = create_str_replace_editor_tool()
    assert StrReplaceEditorTool['type'] == 'function'
    assert StrReplaceEditorTool['function']['name'] == 'str_replace_editor'

    properties = StrReplaceEditorTool['function']['parameters']['properties']
    assert 'command' in properties
    assert 'path' in properties
    assert 'file_text' in properties
    assert 'old_str' in properties
    assert 'new_str' in properties
    assert 'insert_line' in properties
    assert 'security_risk' in properties

    assert StrReplaceEditorTool['function']['parameters']['required'] == [
        'command',
        'path',
        'security_risk',
    ]


def test_browser_tool():
    assert BrowserTool['type'] == 'function'
    assert BrowserTool['function']['name'] == 'browser'
    assert 'code' in BrowserTool['function']['parameters']['properties']
    assert 'security_risk' in BrowserTool['function']['parameters']['properties']
    assert BrowserTool['function']['parameters']['required'] == [
        'code',
        'security_risk',
    ]
    # Check that the description includes all the functions
    description = _BROWSER_TOOL_DESCRIPTION
    assert 'goto(' in description
    assert 'go_back()' in description
    assert 'go_forward()' in description
    assert 'noop(' in description
    assert 'scroll(' in description
    assert 'fill(' in description
    assert 'select_option(' in description
    assert 'click(' in description
    assert 'dblclick(' in description
    assert 'hover(' in description
    assert 'press(' in description
    assert 'focus(' in description
    assert 'clear(' in description
    assert 'drag_and_drop(' in description
    assert 'upload_file(' in description

    # Test BrowserTool definition
    assert BrowserTool['type'] == 'function'
    assert BrowserTool['function']['name'] == 'browser'
    assert BrowserTool['function']['description'] == _BROWSER_DESCRIPTION
    assert BrowserTool['function']['parameters']['type'] == 'object'
    assert 'code' in BrowserTool['function']['parameters']['properties']
    assert BrowserTool['function']['parameters']['required'] == [
        'code',
        'security_risk',
    ]
    assert (
        BrowserTool['function']['parameters']['properties']['code']['type'] == 'string'
    )
    assert 'description' in BrowserTool['function']['parameters']['properties']['code']


def test_response_to_actions_invalid_tool():
    # Test response with invalid tool call
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = 'Invalid tool'
    mock_response.choices[0].message.tool_calls = [Mock()]
    mock_response.choices[0].message.tool_calls[0].id = 'tool_call_10'
    mock_response.choices[0].message.tool_calls[0].function = Mock()
    mock_response.choices[0].message.tool_calls[0].function.name = 'invalid_tool'
    mock_response.choices[0].message.tool_calls[0].function.arguments = '{}'

    with pytest.raises(FunctionCallNotExistsError):
        codeact_response_to_actions(mock_response)
    with pytest.raises(FunctionCallNotExistsError):
        readonly_response_to_actions(mock_response)


def test_step_with_no_pending_actions(mock_state: State, create_llm_registry):
    # Mock the LLM response
    mock_response = Mock()
    mock_response.id = 'mock_id'
    mock_response.total_calls_in_response = 1
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = 'Task completed'
    mock_response.choices[0].message.tool_calls = []

    mock_config = Mock()
    mock_config.model = 'mock_model'

    llm = Mock()
    llm.config = mock_config
    llm.completion = Mock(return_value=mock_response)
    llm.is_function_calling_active = Mock(return_value=True)  # Enable function calling
    llm.is_caching_prompt_active = Mock(return_value=False)
    llm.format_messages_for_llm = Mock(return_value=[])  # Mock message formatting

    # Create agent with mocked LLM
    llm_config = LLMConfig(model='gpt-4o', api_key='test_key')
    config = AgentConfig()
    config.enable_prompt_extensions = False
    agent = CodeActAgent(config=config, llm_registry=create_llm_registry(llm_config))
    # Replace the LLM with our mock after creation
    agent.llm = llm

    # Test step with no pending actions
    mock_state.latest_user_message = None
    mock_state.latest_user_message_id = None
    mock_state.latest_user_message_timestamp = None
    mock_state.latest_user_message_cause = None
    mock_state.latest_user_message_timeout = None
    mock_state.latest_user_message_llm_metrics = None
    mock_state.latest_user_message_tool_call_metadata = None

    # Add initial user message to history
    initial_user_message = MessageAction(content='Initial user message')
    initial_user_message._source = EventSource.USER
    mock_state.history = [initial_user_message]

    # Mock the view returned by condenser
    mock_view = View(events=mock_state.history)
    mock_state.view = mock_view

    action = agent.step(mock_state)
    assert isinstance(action, MessageAction)
    assert action.content == 'Task completed'


@pytest.mark.parametrize('agent_type', ['CodeActAgent', 'ReadOnlyAgent'])
def test_correct_tool_description_loaded_based_on_model_name(
    agent_type, create_llm_registry
):
    """Tests that the simplified tool descriptions are loaded for specific models."""
    o3_mock_config = LLMConfig(model='mock_o3_model', api_key='test_key')
    if agent_type == 'CodeActAgent':
        from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent

        agent_class = CodeActAgent
    else:
        from openhands.agenthub.readonly_agent.readonly_agent import ReadOnlyAgent

        agent_class = ReadOnlyAgent

    agent = agent_class(
        config=AgentConfig(),
        llm_registry=create_llm_registry(o3_mock_config),
    )
    for tool in agent.tools:
        # Assert all descriptions have less than 1024 characters
        assert len(tool['function']['description']) < 1024

    sonnect_mock_config = LLMConfig(model='mock_sonnet_model', api_key='test_key')
    agent = agent_class(
        config=AgentConfig(),
        llm_registry=create_llm_registry(sonnect_mock_config),
    )
    # Assert existence of the detailed tool descriptions that are longer than 1024 characters
    if agent_type == 'CodeActAgent':
        # This only holds for CodeActAgent
        assert any(len(tool['function']['description']) > 1024 for tool in agent.tools)


def test_mismatched_tool_call_events_and_auto_add_system_message(
    agent, mock_state: State
):
    """Tests that the agent can convert mismatched tool call events (i.e., an observation with no corresponding action) into messages.

    This also tests that the system message is automatically added to the event stream if SystemMessageAction is not present.
    """
    tool_call_metadata = Mock(
        spec=ToolCallMetadata,
        model_response=Mock(
            id='model_response_0',
            choices=[
                Mock(
                    message=Mock(
                        role='assistant',
                        content='',
                        tool_calls=[
                            Mock(spec=ChatCompletionMessageToolCall, id='tool_call_0')
                        ],
                    )
                )
            ],
        ),
        tool_call_id='tool_call_0',
        function_name='foo',
    )

    action = CmdRunAction('foo')
    action._source = EventSource.AGENT
    action.tool_call_metadata = tool_call_metadata

    observation = CmdOutputObservation(content='', command_id=0, command='foo')
    observation.tool_call_metadata = tool_call_metadata

    # Add initial user message
    initial_user_message = MessageAction(content='Initial user message')
    initial_user_message._source = EventSource.USER

    # When both events are provided, the agent should get three messages:
    # 1. The system message (added automatically for backward compatibility)
    # 2. The action message
    # 3. The observation message
    mock_state.history = [initial_user_message, action, observation]
    messages = agent._get_messages(mock_state.history, initial_user_message, set())
    assert len(messages) == 4  # System + initial user + action + observation
    assert messages[0].role == 'system'  # First message should be the system message
    assert (
        messages[1].role == 'user'
    )  # Second message should be the initial user message
    assert messages[2].role == 'assistant'  # Third message should be the action
    assert messages[3].role == 'tool'  # Fourth message should be the observation

    # The same should hold if the events are presented out-of-order
    mock_state.history = [initial_user_message, observation, action]
    messages = agent._get_messages(mock_state.history, initial_user_message, set())
    assert len(messages) == 4
    assert messages[0].role == 'system'  # First message should be the system message
    assert (
        messages[1].role == 'user'
    )  # Second message should be the initial user message

    # If only one of the two events is present, then we should just get the system message
    # plus any valid message from the event
    mock_state.history = [initial_user_message, action]
    messages = agent._get_messages(mock_state.history, initial_user_message, set())
    assert (
        len(messages) == 2
    )  # System + initial user message, action is waiting for its observation
    assert messages[0].role == 'system'
    assert messages[1].role == 'user'

    mock_state.history = [initial_user_message, observation]
    messages = agent._get_messages(mock_state.history, initial_user_message, set())
    assert (
        len(messages) == 2
    )  # System + initial user message, observation has no matching action
    assert messages[0].role == 'system'
    assert messages[1].role == 'user'


def test_grep_tool():
    assert GrepTool['type'] == 'function'
    assert GrepTool['function']['name'] == 'grep'

    properties = GrepTool['function']['parameters']['properties']
    assert 'pattern' in properties
    assert 'path' in properties
    assert 'include' in properties

    assert GrepTool['function']['parameters']['required'] == ['pattern']


def test_glob_tool():
    assert GlobTool['type'] == 'function'
    assert GlobTool['function']['name'] == 'glob'

    properties = GlobTool['function']['parameters']['properties']
    assert 'pattern' in properties
    assert 'path' in properties

    assert GlobTool['function']['parameters']['required'] == ['pattern']


def test_think_tool():
    assert ThinkTool['type'] == 'function'
    assert ThinkTool['function']['name'] == 'think'

    properties = ThinkTool['function']['parameters']['properties']
    assert 'thought' in properties

    assert ThinkTool['function']['parameters']['required'] == ['thought']


def test_enhance_messages_adds_newlines_between_consecutive_user_messages(
    agent: CodeActAgent,
):
    """Test that _enhance_messages adds newlines between consecutive user messages."""
    # Create consecutive user messages with various content types
    messages = [
        # First user message with TextContent only
        Message(role='user', content=[TextContent(text='First user message')]),
        # Second user message with TextContent only - should get newlines added
        Message(role='user', content=[TextContent(text='Second user message')]),
        # Assistant message
        Message(role='assistant', content=[TextContent(text='Assistant response')]),
        # Third user message with TextContent only - shouldn't get newlines
        Message(role='user', content=[TextContent(text='Third user message')]),
        # Fourth user message with ImageContent first, TextContent second - should get newlines
        Message(
            role='user',
            content=[
                ImageContent(image_urls=['https://example.com/image.jpg']),
                TextContent(text='Fourth user message with image'),
            ],
        ),
        # Fifth user message with only ImageContent - no TextContent to modify
        Message(
            role='user',
            content=[
                ImageContent(image_urls=['https://example.com/another-image.jpg'])
            ],
        ),
    ]

    # Call _enhance_messages
    enhanced_messages = agent.conversation_memory._apply_user_message_formatting(
        messages
    )

    # Verify newlines were added correctly
    assert enhanced_messages[1].content[0].text.startswith('\n\n')
    assert enhanced_messages[1].content[0].text == '\n\nSecond user message'

    # Third message follows assistant, so shouldn't have newlines
    assert not enhanced_messages[3].content[0].text.startswith('\n\n')
    assert enhanced_messages[3].content[0].text == 'Third user message'

    # Fourth message follows user, so should have newlines in its TextContent
    assert enhanced_messages[4].content[1].text.startswith('\n\n')
    assert enhanced_messages[4].content[1].text == '\n\nFourth user message with image'

    # Fifth message only has ImageContent, no TextContent to modify
    assert len(enhanced_messages[5].content) == 1
    assert isinstance(enhanced_messages[5].content[0], ImageContent)


def test_get_system_message(create_llm_registry):
    """Test that the Agent.get_system_message method returns a SystemMessageAction."""
    # Create a mock agent
    config = AgentConfig()
    llm_config = LLMConfig(model='gpt-4o', api_key='test_key')
    agent = CodeActAgent(config=config, llm_registry=create_llm_registry(llm_config))

    result = agent.get_system_message()

    # Check that the system message was created correctly
    assert isinstance(result, SystemMessageAction)
    assert 'You are OpenHands agent' in result.content
    assert len(result.tools) > 0
    assert any(tool['function']['name'] == 'execute_bash' for tool in result.tools)
    assert result._source == EventSource.AGENT


def test_step_raises_error_if_no_initial_user_message(
    agent: CodeActAgent, mock_state: State
):
    """Tests that step raises ValueError if the initial user message is not found."""
    # Ensure history does NOT contain a user MessageAction
    assistant_message = MessageAction(content='Assistant message')
    assistant_message._source = EventSource.AGENT
    mock_state.history = [assistant_message]
    # Mock the condenser to return the history as is
    agent.condenser = Mock()
    agent.condenser.condensed_history.return_value = View(events=mock_state.history)

    with pytest.raises(ValueError, match='Initial user message not found'):
        agent.step(mock_state)
