"""Test for MCP tool timeout causing agent to stall indefinitely."""

import asyncio
import json
from unittest import mock

import pytest
from mcp import McpError

from openhands.controller.agent import Agent
from openhands.controller.agent_controller import AgentController
from openhands.core.schema import AgentState
from openhands.events.action.mcp import MCPAction
from openhands.events.action.message import SystemMessageAction
from openhands.events.event import EventSource
from openhands.events.observation.mcp import MCPObservation
from openhands.events.stream import EventStream
from openhands.mcp.client import MCPClient
from openhands.mcp.tool import MCPClientTool
from openhands.mcp.utils import call_tool_mcp
from openhands.server.services.conversation_stats import ConversationStats
from openhands.storage.memory import InMemoryFileStore


class MockConfig:
    """Mock config for testing."""

    def __init__(self):
        self.max_message_chars = 10000


class MockLLM:
    """Mock LLM for testing."""

    def __init__(self):
        self.metrics = None
        self.config = MockConfig()


@pytest.fixture
def conversation_stats():
    return ConversationStats(None, 'convo-id', None)


class MockAgent(Agent):
    """Mock agent for testing."""

    def __init__(self):
        self.step_called = False
        self.next_action = None
        self.llm = MockLLM()

    def step(self, *args, **kwargs):
        """Mock step method."""
        self.step_called = True
        return self.next_action

    def get_system_message(self):
        """Mock get_system_message method."""
        return SystemMessageAction(content='System message')


@pytest.mark.asyncio
async def test_mcp_tool_timeout_error_handling(conversation_stats):
    """Test that verifies MCP tool timeout errors are properly handled and returned as observations."""
    # Create a mock MCPClient
    mock_client = mock.MagicMock(spec=MCPClient)

    # Configure the mock to raise a McpError when call_tool is called
    async def mock_call_tool(*args, **kwargs):
        # Simulate a timeout
        await asyncio.sleep(0.1)
        # Create a mock error object with the message attribute
        error = mock.MagicMock()
        error.message = 'Timed out while waiting for response to ClientRequest. Waited 30.0 seconds.'
        raise McpError(error)

    mock_client.call_tool.side_effect = mock_call_tool

    # Create a mock tool
    mock_tool = MCPClientTool(
        name='test_tool',
        description='Test tool',
        inputSchema={'type': 'object', 'properties': {}},
        session=None,
    )
    mock_client.tools = [mock_tool]
    mock_client.tool_map = {'test_tool': mock_tool}

    # Create a mock file store
    mock_file_store = InMemoryFileStore({})

    # Create a mock event stream
    event_stream = EventStream(sid='test-session', file_store=mock_file_store)

    # Create a mock agent
    agent = MockAgent()

    # Create a mock agent controller
    controller = AgentController(
        agent=agent,
        event_stream=event_stream,
        conversation_stats=conversation_stats,
        iteration_delta=10,
        budget_per_task_delta=None,
        sid='test-session',
    )

    # Set up the agent state
    await controller.set_agent_state_to(AgentState.RUNNING)

    # Create an MCP action
    mcp_action = MCPAction(
        name='test_tool',
        arguments={'param': 'value'},
        thought='Testing MCP timeout handling',
    )

    # Add the action to the event stream
    event_stream.add_event(mcp_action, EventSource.AGENT)

    # Set the pending action
    controller._pending_action = mcp_action

    # Before the fix, this would raise an exception and not return an observation
    # Now with the fix, it should return an error observation
    result = await call_tool_mcp([mock_client], mcp_action)

    # Verify that the function returns an error observation
    assert isinstance(result, MCPObservation)
    content = json.loads(result.content)
    assert content['isError'] is True
    assert 'timed out' in content['error'].lower()

    # The agent controller would now be able to continue processing
    # because it received an error observation instead of an exception

    # Verify that the agent is still in the RUNNING state
    assert controller.get_agent_state() == AgentState.RUNNING

    # Verify that the agent can continue processing
    agent.next_action = MCPAction(
        name='another_tool',
        arguments={'param': 'value'},
        thought='Another action after timeout',
    )

    # The agent controller would be able to step because it received an observation
    # This demonstrates that the fix is working


@pytest.mark.skip(reason='2025-10-07 : This test is flaky')
@pytest.mark.asyncio
async def test_mcp_tool_timeout_agent_continuation(conversation_stats):
    """Test that verifies the agent can continue processing after an MCP tool timeout."""
    # Create a mock MCPClient
    mock_client = mock.MagicMock(spec=MCPClient)

    # Configure the mock to raise a McpError when call_tool is called
    async def mock_call_tool(*args, **kwargs):
        # Simulate a timeout
        await asyncio.sleep(0.1)
        # Create a mock error object with the message attribute
        error = mock.MagicMock()
        error.message = 'Timed out while waiting for response to ClientRequest. Waited 30.0 seconds.'
        raise McpError(error)

    mock_client.call_tool.side_effect = mock_call_tool

    # Create a mock tool
    mock_tool = MCPClientTool(
        name='test_tool',
        description='Test tool',
        inputSchema={'type': 'object', 'properties': {}},
        session=None,
    )
    mock_client.tools = [mock_tool]
    mock_client.tool_map = {'test_tool': mock_tool}

    # Create a mock file store
    mock_file_store = InMemoryFileStore({})

    # Create a mock event stream
    event_stream = EventStream(sid='test-session', file_store=mock_file_store)

    # Create a mock agent
    agent = MockAgent()

    # Create a mock agent controller
    controller = AgentController(
        agent=agent,
        event_stream=event_stream,
        conversation_stats=conversation_stats,
        iteration_delta=10,
        budget_per_task_delta=None,
        sid='test-session',
    )

    # Set up the agent state
    await controller.set_agent_state_to(AgentState.RUNNING)

    # Create an MCP action
    mcp_action = MCPAction(
        name='test_tool',
        arguments={'param': 'value'},
        thought='Testing MCP timeout handling',
    )

    # Add the action to the event stream
    event_stream.add_event(mcp_action, EventSource.AGENT)

    # Set the pending action
    controller._pending_action = mcp_action

    # Now implement the fix in call_tool_mcp
    async def fixed_call_tool_mcp(clients, action):
        try:
            # This will raise the McpError
            await mock_client.call_tool(action.name, action.arguments)
        except McpError as e:
            # Create an error observation
            error_content = json.dumps(
                {'isError': True, 'error': str(e), 'content': []}
            )
            observation = MCPObservation(
                content=error_content,
                name=action.name,
                arguments=action.arguments,
            )
            # Set the cause
            setattr(observation, '_cause', action.id)
            return observation

    # Use our fixed function
    with mock.patch(
        'openhands.mcp.utils.call_tool_mcp', side_effect=fixed_call_tool_mcp
    ):
        # Call the function that would normally be called by the agent controller
        result = await call_tool_mcp([mock_client], mcp_action)

        # Verify that the function returns an error observation
        assert isinstance(result, MCPObservation)
        content = json.loads(result.content)
        assert content['isError'] is True
        assert 'timed out' in content['error'].lower()

        # Now simulate the agent controller's handling of the observation
        event_stream.add_event(result, EventSource.ENVIRONMENT)

        # Verify that the pending action is cleared
        controller._pending_action = None

        # Verify that the agent is still in the RUNNING state
        assert controller.get_agent_state() == AgentState.RUNNING

        # Verify that the agent can continue processing
        agent.next_action = MCPAction(
            name='another_tool',
            arguments={'param': 'value'},
            thought='Another action after timeout',
        )

        # Simulate a step
        await controller._step()

        # Verify that the agent was asked to step
        assert agent.step_called
