import asyncio
import os
import shutil
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openhands.controller.agent import Agent
from openhands.controller.agent_controller import AgentController
from openhands.core.config import OpenHandsConfig
from openhands.core.main import run_controller
from openhands.core.schema.agent import AgentState
from openhands.events.action.agent import RecallAction
from openhands.events.action.message import MessageAction, SystemMessageAction
from openhands.events.event import EventSource
from openhands.events.observation.agent import (
    RecallObservation,
    RecallType,
)
from openhands.events.serialization.observation import observation_from_dict
from openhands.events.stream import EventStream
from openhands.llm import LLM
from openhands.llm.llm_registry import LLMRegistry
from openhands.llm.metrics import Metrics
from openhands.memory.memory import Memory
from openhands.runtime.impl.action_execution.action_execution_client import (
    ActionExecutionClient,
)
from openhands.server.services.conversation_stats import ConversationStats
from openhands.server.session.agent_session import AgentSession
from openhands.storage.memory import InMemoryFileStore
from openhands.utils.prompt import (
    ConversationInstructions,
    PromptManager,
    RepositoryInfo,
    RuntimeInfo,
)


@pytest.fixture
def file_store():
    """Create a temporary file store for testing."""
    return InMemoryFileStore({})


@pytest.fixture
def mock_llm_registry(file_store):
    """Create a mock LLMRegistry for testing."""
    return MagicMock(spec=LLMRegistry)


@pytest.fixture
def event_stream(file_store):
    """Create a test event stream."""
    return EventStream(sid='test_sid', file_store=file_store)


@pytest.fixture
def memory(event_stream):
    """Create a test memory instance."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    memory = Memory(event_stream, 'test_sid')
    yield memory
    loop.close()


@pytest.fixture
def prompt_dir(tmp_path):
    # Copy contents from "openhands/agenthub/codeact_agent" to the temp directory
    shutil.copytree(
        'openhands/agenthub/codeact_agent/prompts', tmp_path, dirs_exist_ok=True
    )

    # Return the temporary directory path
    return tmp_path


@pytest.fixture
def mock_agent():
    # Create a dummy agent for the controller
    agent = MagicMock(spec=Agent)
    agent.llm = MagicMock(spec=LLM)
    agent.llm.metrics = Metrics()
    agent.llm.config = OpenHandsConfig().get_llm_config()

    # Add a proper system message mock
    system_message = SystemMessageAction(content='Test system message')
    system_message._source = EventSource.AGENT
    system_message._id = -1  # Set invalid ID to avoid the ID check
    agent.get_system_message.return_value = system_message

    agent.config = MagicMock()
    agent.config.enable_mcp = False

    return agent


@pytest.mark.asyncio
async def test_memory_on_event_exception_handling(memory, event_stream, mock_agent):
    """Test that exceptions in Memory.on_event are properly handled via status callback."""
    # Create a mock runtime
    runtime = MagicMock(spec=ActionExecutionClient)
    runtime.event_stream = event_stream

    # Mock Memory method to raise an exception
    with (
        patch.object(
            memory, '_on_workspace_context_recall', side_effect=Exception('Test error')
        ),
        patch('openhands.core.main.create_agent', return_value=mock_agent),
    ):
        state = await run_controller(
            config=OpenHandsConfig(),
            initial_user_action=MessageAction(content='Test message'),
            runtime=runtime,
            sid='test',
            fake_user_response_fn=lambda _: 'repeat',
            memory=memory,
        )

        # Verify that the controller's last error was set
        assert state.iteration_flag.current_value == 0
        assert state.agent_state == AgentState.ERROR
        assert state.last_error == 'Error: Exception'


@pytest.mark.asyncio
async def test_memory_on_workspace_context_recall_exception_handling(
    memory, event_stream, mock_agent
):
    """Test that exceptions in Memory._on_workspace_context_recall are properly handled via status callback."""
    # Create a mock runtime
    runtime = MagicMock(spec=ActionExecutionClient)
    runtime.event_stream = event_stream

    # Mock Memory._on_workspace_context_recall to raise an exception
    with (
        patch.object(
            memory,
            '_find_microagent_knowledge',
            side_effect=Exception('Test error from _find_microagent_knowledge'),
        ),
        patch('openhands.core.main.create_agent', return_value=mock_agent),
    ):
        state = await run_controller(
            config=OpenHandsConfig(),
            initial_user_action=MessageAction(content='Test message'),
            runtime=runtime,
            sid='test',
            fake_user_response_fn=lambda _: 'repeat',
            memory=memory,
        )

        # Verify that the controller's last error was set
        assert state.iteration_flag.current_value == 0
        assert state.agent_state == AgentState.ERROR
        assert state.last_error == 'Error: Exception'


@pytest.mark.asyncio
async def test_memory_with_microagents():
    """Test that Memory loads microagents from the global directory and processes microagent actions.

    This test verifies that:
    1. Memory loads microagents from the global GLOBAL_MICROAGENTS_DIR
    2. When a microagent action with a trigger word is processed, a RecallObservation is created
    """
    # Create a mock event stream
    event_stream = MagicMock(spec=EventStream)

    # Initialize Memory to use the global microagents dir
    memory = Memory(
        event_stream=event_stream,
        sid='test-session',
    )

    # Verify microagents were loaded - at least one microagent should be loaded
    # from the global directory that's in the repo
    assert len(memory.knowledge_microagents) > 0

    # Check for the derived name 'flarglebargle'
    derived_name = 'flarglebargle'
    assert derived_name in memory.knowledge_microagents

    # Create a microagent action with the trigger word
    microagent_action = RecallAction(
        query='Hello, flarglebargle!', recall_type=RecallType.KNOWLEDGE
    )

    # Set the source to USER
    microagent_action._source = EventSource.USER  # type: ignore[attr-defined]

    # Mock the event_stream.add_event method
    added_events = []

    def original_add_event(event, source):
        added_events.append((event, source))

    event_stream.add_event = original_add_event

    # Add the microagent action to the event stream
    event_stream.add_event(microagent_action, EventSource.USER)

    # Clear the events list to only capture new events
    added_events.clear()

    # Process the microagent action
    await memory._on_event(microagent_action)

    # Verify a RecallObservation was added to the event stream
    assert len(added_events) == 1
    observation, source = added_events[0]
    assert isinstance(observation, RecallObservation)
    assert source == EventSource.ENVIRONMENT
    assert observation.recall_type == RecallType.KNOWLEDGE

    # We should have at least one microagent: flarglebargle (triggered by keyword)
    # Note: The default-tools microagent might not be loaded in tests
    assert len(observation.microagent_knowledge) == 1

    # Find the flarglebargle microagent in the list
    flarglebargle_knowledge = None
    for knowledge in observation.microagent_knowledge:
        if knowledge.name == derived_name:
            flarglebargle_knowledge = knowledge
            break

    # Check against the derived name
    assert flarglebargle_knowledge is not None
    assert flarglebargle_knowledge.name == derived_name
    assert flarglebargle_knowledge.trigger == 'flarglebargle'
    assert 'magic word' in flarglebargle_knowledge.content


def test_memory_repository_info(prompt_dir, file_store):
    """Test that Memory adds repository info to RecallObservations."""
    # real event stream
    event_stream = EventStream(sid='test-session', file_store=file_store)

    # Create a test repo microagent first
    repo_microagent_name = 'test_repo_microagent'
    repo_microagent_content = """---
name: test_repo
type: repo
agent: CodeActAgent
---

REPOSITORY INSTRUCTIONS: This is a test repository.
"""

    # Create a temporary repo microagent file
    os.makedirs(os.path.join(prompt_dir, 'micro'), exist_ok=True)
    with open(
        os.path.join(prompt_dir, 'micro', f'{repo_microagent_name}.md'), 'w'
    ) as f:
        f.write(repo_microagent_content)

    # Patch the global microagents directory to use our test directory
    test_microagents_dir = os.path.join(prompt_dir, 'micro')
    with patch('openhands.memory.memory.GLOBAL_MICROAGENTS_DIR', test_microagents_dir):
        # Initialize Memory
        memory = Memory(
            event_stream=event_stream,
            sid='test-session',
        )

        # Set repository info
        memory.set_repository_info('owner/repo', '/workspace/repo')

        # Create and add the first user message
        user_message = MessageAction(content='First user message')
        user_message._source = EventSource.USER  # type: ignore[attr-defined]
        event_stream.add_event(user_message, EventSource.USER)

        # Create and add the microagent action
        microagent_action = RecallAction(
            query='First user message', recall_type=RecallType.WORKSPACE_CONTEXT
        )
        microagent_action._source = EventSource.USER  # type: ignore[attr-defined]
        event_stream.add_event(microagent_action, EventSource.USER)

        # Give it a little time to process
        time.sleep(0.3)

        # Get all events from the stream
        events = list(event_stream.get_events())

        # Find the RecallObservation event
        microagent_obs_events = [
            event for event in events if isinstance(event, RecallObservation)
        ]

        # We should have at least one RecallObservation
        assert len(microagent_obs_events) > 0

        # Get the first RecallObservation
        observation = microagent_obs_events[0]
        assert observation.recall_type == RecallType.WORKSPACE_CONTEXT
        assert observation.repo_name == 'owner/repo'
        assert observation.repo_directory == '/workspace/repo'
        assert 'This is a test repository' in observation.repo_instructions

    # Clean up
    os.remove(os.path.join(prompt_dir, 'micro', f'{repo_microagent_name}.md'))


@pytest.mark.asyncio
async def test_memory_with_agent_microagents():
    """Test that Memory processes microagent based on trigger words from agent messages."""
    # Create a mock event stream
    event_stream = MagicMock(spec=EventStream)

    # Initialize Memory to use the global microagents dir
    memory = Memory(
        event_stream=event_stream,
        sid='test-session',
    )

    # Verify microagents were loaded - at least one microagent should be loaded
    # from the global directory that's in the repo
    assert len(memory.knowledge_microagents) > 0

    # Check for the derived name 'flarglebargle'
    derived_name = 'flarglebargle'
    assert derived_name in memory.knowledge_microagents

    # Create a microagent action with the trigger word
    microagent_action = RecallAction(
        query='Hello, flarglebargle!', recall_type=RecallType.KNOWLEDGE
    )

    # Set the source to AGENT
    microagent_action._source = EventSource.AGENT  # type: ignore[attr-defined]

    # Mock the event_stream.add_event method
    added_events = []

    def original_add_event(event, source):
        added_events.append((event, source))

    event_stream.add_event = original_add_event

    # Add the microagent action to the event stream
    event_stream.add_event(microagent_action, EventSource.AGENT)

    # Clear the events list to only capture new events
    added_events.clear()

    # Process the microagent action
    await memory._on_event(microagent_action)

    # Verify a RecallObservation was added to the event stream
    assert len(added_events) == 1
    observation, source = added_events[0]
    assert isinstance(observation, RecallObservation)
    assert source == EventSource.ENVIRONMENT
    assert observation.recall_type == RecallType.KNOWLEDGE

    # We should have at least one microagent: flarglebargle (triggered by keyword)
    # Note: The default-tools microagent might not be loaded in tests
    assert len(observation.microagent_knowledge) == 1

    # Find the flarglebargle microagent in the list
    flarglebargle_knowledge = None
    for knowledge in observation.microagent_knowledge:
        if knowledge.name == derived_name:
            flarglebargle_knowledge = knowledge
            break

    # Check against the derived name
    assert flarglebargle_knowledge is not None
    assert flarglebargle_knowledge.name == derived_name
    assert flarglebargle_knowledge.trigger == 'flarglebargle'
    assert 'magic word' in flarglebargle_knowledge.content


@pytest.mark.asyncio
async def test_custom_secrets_descriptions():
    """Test that custom_secrets_descriptions are properly stored in memory and included in RecallObservation."""
    # Create a mock event stream
    event_stream = MagicMock(spec=EventStream)

    # Initialize Memory
    memory = Memory(
        event_stream=event_stream,
        sid='test-session',
    )

    # Create a mock runtime with custom secrets descriptions
    mock_runtime = MagicMock()
    mock_runtime.web_hosts = {'test-host.example.com': 8080}
    mock_runtime.additional_agent_instructions = 'Test instructions'

    # Define custom secrets descriptions
    custom_secrets = {
        'API_KEY': 'API key for external service',
        'DATABASE_URL': 'Connection string for the database',
        'SECRET_TOKEN': 'Authentication token for secure operations',
    }

    # Set runtime info with custom secrets
    memory.set_runtime_info(mock_runtime, custom_secrets, '/workspace')

    # Set repository info
    memory.set_repository_info('test-owner/test-repo', '/workspace/test-repo')

    # Create a workspace context recall action
    recall_action = RecallAction(
        query='Initial message', recall_type=RecallType.WORKSPACE_CONTEXT
    )
    recall_action._source = EventSource.USER  # type: ignore[attr-defined]

    # Mock the event_stream.add_event method
    added_events = []

    def mock_add_event(event, source):
        added_events.append((event, source))

    event_stream.add_event = mock_add_event

    # Process the recall action
    await memory._on_event(recall_action)

    # Verify a RecallObservation was added to the event stream
    assert len(added_events) == 1
    observation, source = added_events[0]

    # Verify the observation is a RecallObservation
    assert isinstance(observation, RecallObservation)
    assert source == EventSource.ENVIRONMENT
    assert observation.recall_type == RecallType.WORKSPACE_CONTEXT

    # Verify custom_secrets_descriptions are included in the observation
    assert observation.custom_secrets_descriptions == custom_secrets

    # Verify repository info is included
    assert observation.repo_name == 'test-owner/test-repo'
    assert observation.repo_directory == '/workspace/test-repo'

    # Verify runtime info is included
    assert observation.runtime_hosts == {'test-host.example.com': 8080}
    assert observation.additional_agent_instructions == 'Test instructions'


def test_custom_secrets_descriptions_serialization(prompt_dir):
    """Test that custom_secrets_descriptions are properly serialized in the message for the LLM."""
    # Create a PromptManager with the test prompt directory
    prompt_manager = PromptManager(prompt_dir)

    # Create a RuntimeInfo with custom_secrets_descriptions
    custom_secrets = {
        'API_KEY': 'API key for external service',
        'DATABASE_URL': 'Connection string for the database',
        'SECRET_TOKEN': 'Authentication token for secure operations',
    }
    runtime_info = RuntimeInfo(
        date='2025-05-15',
        available_hosts={'test-host.example.com': 8080},
        additional_agent_instructions='Test instructions',
        custom_secrets_descriptions=custom_secrets,
        working_dir='/workspace',
    )

    # Create a RepositoryInfo
    repository_info = RepositoryInfo(
        repo_name='test-owner/test-repo',
        repo_directory='/workspace/test-repo',
        branch_name='main',
    )

    conversation_instructions = ConversationInstructions(
        content='additional agent context for the task'
    )

    # Build the workspace context message
    workspace_context = prompt_manager.build_workspace_context(
        repository_info=repository_info,
        runtime_info=runtime_info,
        repo_instructions='Test repository instructions',
        conversation_instructions=conversation_instructions,
    )

    # Verify that the workspace context includes the custom_secrets_descriptions
    assert '<CUSTOM_SECRETS>' in workspace_context
    for secret_name, secret_description in custom_secrets.items():
        assert f'**${secret_name}**' in workspace_context
        assert secret_description in workspace_context

    assert '<CONVERSATION_INSTRUCTIONS>' in workspace_context
    assert 'additional agent context for the task' in workspace_context


def test_serialization_deserialization_with_custom_secrets():
    """Test that RecallObservation can be serialized and deserialized with custom_secrets_descriptions."""
    # This simulates an older version of the RecallObservation
    legacy_observation = {
        'message': 'Added workspace context',
        'observation': 'recall',
        'content': 'Test content',
        'extras': {
            'recall_type': 'workspace_context',
            'repo_name': 'test-owner/test-repo',
            'repo_directory': '/workspace/test-repo',
            'repo_instructions': 'Test repository instructions',
            'runtime_hosts': {'test-host.example.com': 8080},
            'additional_agent_instructions': 'Test instructions',
            'date': '2025-05-15',
            'microagent_knowledge': [],  # Intentionally omitting custom_secrets_descriptions
        },
    }

    legacy_observation = observation_from_dict(legacy_observation)

    # Verify that the observation was created successfully
    assert legacy_observation.recall_type == RecallType.WORKSPACE_CONTEXT
    assert legacy_observation.repo_name == 'test-owner/test-repo'
    assert legacy_observation.repo_directory == '/workspace/test-repo'


def test_memory_multiple_repo_microagents(prompt_dir, file_store):
    """Test that Memory loads and concatenates multiple repo microagents correctly."""
    # Create real event stream
    event_stream = EventStream(sid='test-session', file_store=file_store)

    # Create two test repo microagents
    repo_microagent1_name = 'test_repo_microagent1'
    repo_microagent1_content = """---
REPOSITORY INSTRUCTIONS: This is the first test repository.
"""

    repo_microagent2_name = 'test_repo_microagent2'
    repo_microagent2_content = """---
name: test_repo2
type: repo
agent: CodeActAgent
---

REPOSITORY INSTRUCTIONS: This is the second test repository.
"""

    # Create temporary repo microagent files
    os.makedirs(os.path.join(prompt_dir, 'micro'), exist_ok=True)
    with open(
        os.path.join(prompt_dir, 'micro', f'{repo_microagent1_name}.md'), 'w'
    ) as f:
        f.write(repo_microagent1_content)

    with open(
        os.path.join(prompt_dir, 'micro', f'{repo_microagent2_name}.md'), 'w'
    ) as f:
        f.write(repo_microagent2_content)

    # Patch the global microagents directory to use our test directory
    test_microagents_dir = os.path.join(prompt_dir, 'micro')
    with patch('openhands.memory.memory.GLOBAL_MICROAGENTS_DIR', test_microagents_dir):
        # Initialize Memory
        memory = Memory(
            event_stream=event_stream,
            sid='test-session',
        )

        # Set repository info
        memory.set_repository_info('owner/repo', '/workspace/repo')

        # Create and add the first user message
        user_message = MessageAction(content='First user message')
        user_message._source = EventSource.USER  # type: ignore[attr-defined]
        event_stream.add_event(user_message, EventSource.USER)

        # Create and add the microagent action
        microagent_action = RecallAction(
            query='First user message', recall_type=RecallType.WORKSPACE_CONTEXT
        )
        microagent_action._source = EventSource.USER  # type: ignore[attr-defined]
        event_stream.add_event(microagent_action, EventSource.USER)

        # Give it a little time to process
        time.sleep(0.3)

        # Get all events from the stream
        events = list(event_stream.get_events())

        # Find the RecallObservation event
        microagent_obs_events = [
            event for event in events if isinstance(event, RecallObservation)
        ]

        # We should have one RecallObservation
        assert len(microagent_obs_events) > 0

        # Get the first RecallObservation
        observation = microagent_obs_events[0]
        assert observation.recall_type == RecallType.WORKSPACE_CONTEXT
        assert observation.repo_name == 'owner/repo'
        assert observation.repo_directory == '/workspace/repo'
        assert 'This is the first test repository' in observation.repo_instructions
        assert 'This is the second test repository' in observation.repo_instructions

    # Clean up
    os.remove(os.path.join(prompt_dir, 'micro', f'{repo_microagent1_name}.md'))
    os.remove(os.path.join(prompt_dir, 'micro', f'{repo_microagent2_name}.md'))


@pytest.mark.asyncio
async def test_conversation_instructions_plumbed_to_memory(
    mock_agent, event_stream, file_store, mock_llm_registry
):
    # Setup
    session = AgentSession(
        sid='test-session',
        file_store=file_store,
        llm_registry=mock_llm_registry,
        conversation_stats=ConversationStats(file_store, 'test-session', None),
    )

    # Create a mock runtime and set it up
    mock_runtime = MagicMock(spec=ActionExecutionClient)

    # Mock the runtime creation to set up the runtime attribute
    async def mock_create_runtime(*args, **kwargs):
        session.runtime = mock_runtime
        return True

    session._create_runtime = AsyncMock(side_effect=mock_create_runtime)

    # Create a spy on set_initial_state
    class SpyAgentController(AgentController):
        set_initial_state_call_count = 0
        test_initial_state = None

        def set_initial_state(self, *args, state=None, **kwargs):
            self.set_initial_state_call_count += 1
            self.test_initial_state = state
            super().set_initial_state(*args, state=state, **kwargs)

    # Patch AgentController
    with (
        patch(
            'openhands.server.session.agent_session.AgentController', SpyAgentController
        ),
    ):
        await session.start(
            runtime_name='test-runtime',
            config=OpenHandsConfig(),
            agent=mock_agent,
            max_iterations=10,
            conversation_instructions='instructions for conversation',
        )

        # Use the memory instance from the session, not the fixture
        assert (
            session.memory.conversation_instructions.content
            == 'instructions for conversation'
        )
