import pytest
from litellm import ModelResponse

from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.core.config import AgentConfig, LLMConfig
from openhands.core.config.openhands_config import OpenHandsConfig
from openhands.events.action import MessageAction
from openhands.llm.llm_registry import LLMRegistry


@pytest.fixture
def llm_config():
    return LLMConfig(
        model='claude-3-5-sonnet-20241022',
        api_key='fake',
        caching_prompt=True,
    )


@pytest.fixture
def llm_registry():
    registry = LLMRegistry(config=OpenHandsConfig())
    return registry


@pytest.fixture
def codeact_agent(llm_registry):
    config = AgentConfig()
    agent = CodeActAgent(config, llm_registry)
    return agent


def response_mock(content: str, tool_call_id: str):
    class MockModelResponse:
        def __init__(self, content, tool_call_id):
            self.choices = [
                {
                    'message': {
                        'content': content,
                        'tool_calls': [
                            {
                                'function': {
                                    'id': tool_call_id,
                                    'name': 'execute_bash',
                                    'arguments': '{}',
                                }
                            }
                        ],
                    }
                }
            ]

        def model_dump(self):
            return {'choices': self.choices}

    return ModelResponse(**MockModelResponse(content, tool_call_id).model_dump())


def test_get_messages(codeact_agent: CodeActAgent):
    # Add some events to history
    history = list()
    # Add system message action
    system_message_action = codeact_agent.get_system_message()
    history.append(system_message_action)

    message_action_1 = MessageAction('Initial user message')
    message_action_1._source = 'user'
    history.append(message_action_1)
    message_action_2 = MessageAction('Sure!')
    message_action_2._source = 'agent'
    history.append(message_action_2)
    message_action_3 = MessageAction('Hello, agent!')
    message_action_3._source = 'user'
    history.append(message_action_3)
    message_action_4 = MessageAction('Hello, user!')
    message_action_4._source = 'agent'
    history.append(message_action_4)
    message_action_5 = MessageAction('Laaaaaaaast!')
    message_action_5._source = 'user'
    history.append(message_action_5)

    codeact_agent.reset()
    messages = codeact_agent._get_messages(history, message_action_1, set())

    assert (
        len(messages) == 6
    )  # System, initial user + user message, agent message, last user message
    assert messages[0].role == 'system'  # system message
    assert messages[0].content[0].cache_prompt  # system message should be cached
    assert messages[1].role == 'user'
    assert messages[1].content[0].text.endswith('Initial user message')
    # we add cache breakpoint to only the last user message
    assert not messages[1].content[0].cache_prompt

    assert messages[3].role == 'user'
    assert messages[3].content[0].text == ('Hello, agent!')
    assert not messages[3].content[0].cache_prompt
    assert messages[4].role == 'assistant'
    assert messages[4].content[0].text == 'Hello, user!'
    assert not messages[4].content[0].cache_prompt
    assert messages[5].role == 'user'
    assert messages[5].content[0].text.startswith('Laaaaaaaast!')
    assert messages[5].content[0].cache_prompt


def test_get_messages_prompt_caching(codeact_agent: CodeActAgent):
    history = list()
    # Add system message action
    system_message_action = codeact_agent.get_system_message()
    history.append(system_message_action)

    # Add multiple user and agent messages
    initial_user_message = None  # Keep track of the first user message
    for i in range(15):
        message_action_user = MessageAction(f'User message {i}')
        message_action_user._source = 'user'
        if initial_user_message is None:
            initial_user_message = message_action_user  # Store the first one
        history.append(message_action_user)
        message_action_agent = MessageAction(f'Agent message {i}')
        message_action_agent._source = 'agent'
        history.append(message_action_agent)

    codeact_agent.reset()
    messages = codeact_agent._get_messages(history, initial_user_message, set())

    # Check that only the last two user messages have cache_prompt=True
    cached_user_messages = [
        msg
        for msg in messages
        if msg.role in ('user', 'system') and msg.content[0].cache_prompt
    ]
    assert (
        len(cached_user_messages) == 2
    )  # Including the initial system message + last user message

    # Verify that these are indeed the last user message (from start)
    assert cached_user_messages[0].content[0].text.startswith('You are OpenHands agent')
    assert cached_user_messages[1].content[0].text.startswith('User message 14')
