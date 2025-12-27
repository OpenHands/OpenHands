import asyncio
import json
import os
import signal
import sys
from pathlib import Path
from typing import Callable, Protocol

import openhands.agenthub  # noqa F401 (we import this to get the agents registered)
from openhands.controller.replay import ReplayManager
from openhands.controller.state.state import State
from openhands.core.config import (
    OpenHandsConfig,
    parse_arguments,
    setup_config_from_args,
)
from openhands.core.config.mcp_config import OpenHandsMCPConfigImpl
from openhands.core.logger import openhands_logger as logger
from openhands.core.loop import run_agent_until_done
from openhands.core.schema import AgentState
from openhands.core.setup import (
    create_agent,
    create_controller,
    create_memory,
    create_runtime,
    generate_sid,
    get_provider_tokens,
    initialize_repository_for_runtime,
)
from openhands.events import EventSource, EventStreamSubscriber
from openhands.events.action import MessageAction, NullAction
from openhands.events.action.action import Action
from openhands.events.event import Event
from openhands.events.observation import AgentStateChangedObservation
from openhands.io import read_input, read_task
from openhands.mcp import add_mcp_tools_to_agent
from openhands.memory.memory import Memory
from openhands.runtime.base import Runtime
from openhands.utils.async_utils import call_async_from_sync
from openhands.utils.utils import create_registry_and_conversation_stats


class FakeUserResponseFunc(Protocol):
    def __call__(
        self,
        state: State,
        encapsulate_solution: bool = False,
        try_parse: Callable[[Action | None], str] | None = None,
    ) -> str: ...


async def run_controller(
    config: OpenHandsConfig,
    initial_user_action: Action,
    sid: str | None = None,
    runtime: Runtime | None = None,
    exit_on_message: bool = False,
    fake_user_response_fn: FakeUserResponseFunc | None = None,
    headless_mode: bool = True,
    memory: Memory | None = None,
    conversation_instructions: str | None = None,
) -> State | None:
    """Main coroutine to run the agent controller with task input flexibility.

    It's only used when you launch openhands backend directly via cmdline.

    Args:
        config: The app config.
        initial_user_action: An Action object containing initial user input
        sid: (optional) The session id. IMPORTANT: please don't set this unless you know what you're doing.
            Set it to incompatible value will cause unexpected behavior on RemoteRuntime.
        runtime: (optional) A runtime for the agent to run on.
        exit_on_message: quit if agent asks for a message from user (optional)
        fake_user_response_fn: An optional function that receives the current state
            (could be None) and returns a fake user response.
        headless_mode: Whether the agent is run in headless mode.

    Returns:
        The final state of the agent, or None if an error occurred.

    Raises:
        AssertionError: If initial_user_action is not an Action instance.
        Exception: Various exceptions may be raised during execution and will be logged.

    Notes:
        - State persistence: If config.file_store is set, the agent's state will be
          saved between sessions.
        - Trajectories: If config.trajectories_path is set, execution history will be
          saved as JSON for analysis.
        - Budget control: Execution is limited by config.max_iterations and
          config.max_budget_per_task.

    Example:
        >>> config = load_openhands_config()
        >>> action = MessageAction(content="Write a hello world program")
        >>> state = await run_controller(config=config, initial_user_action=action)
    """
    sid = sid or generate_sid(config)

    llm_registry, conversation_stats, config = create_registry_and_conversation_stats(
        config,
        sid,
        None,
    )

    agent = create_agent(config, llm_registry)

    # when the runtime is created, it will be connected and clone the selected repository
    repo_directory = None
    if runtime is None:
        # In itialize repository if needed
        repo_tokens = get_provider_tokens()
        runtime = create_runtime(
            config,
            llm_registry,
            sid=sid,
            headless_mode=headless_mode,
            agent=agent,
            git_provider_tokens=repo_tokens,
        )
        # Connect to the runtime
        call_async_from_sync(runtime.connect)

        # Initialize repository if needed
        if config.sandbox.selected_repo:
            repo_directory = initialize_repository_for_runtime(
                runtime,
                immutable_provider_tokens=repo_tokens,
                selected_repository=config.sandbox.selected_repo,
            )

    event_stream = runtime.event_stream

    # when memory is created, it will load the microagents from the selected repository
    if memory is None:
        memory = create_memory(
            runtime=runtime,
            event_stream=event_stream,
            sid=sid,
            selected_repository=config.sandbox.selected_repo,
            repo_directory=repo_directory,
            conversation_instructions=conversation_instructions,
            working_dir=str(runtime.workspace_root),
        )

    # Add MCP tools to the agent
    if agent.config.enable_mcp:
        # Add OpenHands' MCP server by default
        _, openhands_mcp_stdio_servers = (
            OpenHandsMCPConfigImpl.create_default_mcp_server_config(
                config.mcp_host, config, None
            )
        )
        runtime.config.mcp.stdio_servers.extend(openhands_mcp_stdio_servers)

        await add_mcp_tools_to_agent(agent, runtime, memory)

    replay_events: list[Event] | None = None
    if config.replay_trajectory_path:
        logger.info('Trajectory replay is enabled')
        assert isinstance(initial_user_action, NullAction)
        replay_events, initial_user_action = load_replay_log(
            config.replay_trajectory_path
        )

    controller, initial_state = create_controller(
        agent, runtime, config, conversation_stats, replay_events=replay_events
    )

    assert isinstance(initial_user_action, Action), (
        f'initial user actions must be an Action, got {type(initial_user_action)}'
    )
    logger.debug(
        f'Agent Controller Initialized: Running agent {agent.name}, model '
        f'{agent.llm.config.model}, with actions: {initial_user_action}'
    )

    # Set up asyncio-safe signal handler for graceful shutdown
    sigint_count = 0
    shutdown_event = asyncio.Event()

    def signal_handler():
        """Handle SIGINT signals for graceful shutdown."""
        nonlocal sigint_count
        sigint_count += 1

        if sigint_count == 1:
            logger.info('Received SIGINT (Ctrl+C). Initiating graceful shutdown...')
            logger.info('Press Ctrl+C again to force immediate exit.')
            shutdown_event.set()
        else:
            logger.info('Received second SIGINT. Forcing immediate exit...')
            sys.exit(1)

    # Register the asyncio signal handler (safer for async contexts)
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, signal_handler)

    # start event is a MessageAction with the task, either resumed or new
    if initial_state is not None and initial_state.last_error:
        # we're resuming the previous session
        event_stream.add_event(
            MessageAction(
                content=(
                    "Let's get back on track. If you experienced errors before, do "
                    'NOT resume your task. Ask me about it.'
                ),
            ),
            EventSource.USER,
        )
    else:
        # init with the provided actions
        event_stream.add_event(initial_user_action, EventSource.USER)

    def on_event(event: Event) -> None:
        if isinstance(event, AgentStateChangedObservation):
            if event.agent_state == AgentState.AWAITING_USER_INPUT:
                if exit_on_message:
                    message = '/exit'
                elif fake_user_response_fn is None:
                    message = read_input(config.cli_multiline_input)
                else:
                    message = fake_user_response_fn(controller.get_state())
                action = MessageAction(content=message)
                event_stream.add_event(action, EventSource.USER)

    event_stream.subscribe(EventStreamSubscriber.MAIN, on_event, sid)

    end_states = [
        AgentState.FINISHED,
        AgentState.REJECTED,
        AgentState.ERROR,
        AgentState.PAUSED,
        AgentState.STOPPED,
    ]

    try:
        # Create a task for the main agent loop
        agent_task = asyncio.create_task(
            run_agent_until_done(controller, runtime, memory, end_states)
        )

        # Wait for either the agent to complete or shutdown signal
        done, pending = await asyncio.wait(
            [agent_task, asyncio.create_task(shutdown_event.wait())],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel any pending tasks
        for task in pending:
            task.cancel()

        # Wait for all cancelled tasks to complete in parallel
        await asyncio.gather(*pending, return_exceptions=True)

        # Check if shutdown was requested
        if shutdown_event.is_set():
            logger.info('Graceful shutdown requested.')

            # Perform graceful cleanup sequence
            try:
                # 1. Stop the agent controller first to prevent new LLM calls
                logger.debug('Stopping agent controller...')
                await controller.close()

                # 2. Stop the EventStream to prevent new events from being processed
                logger.debug('Stopping EventStream...')
                event_stream.close()

                # 3. Give time for in-flight operations to complete before closing runtime
                logger.debug('Waiting for in-flight operations to complete...')
                await asyncio.sleep(0.3)

                # 4. Close the runtime to avoid bash session interruption errors
                logger.debug('Closing runtime...')
                runtime.close()

                # 5. Give a brief moment for final cleanup to complete
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.warning(f'Error during graceful cleanup: {e}')

    except Exception as e:
        logger.error(f'Exception in main loop: {e}')

    # save session when we're about to close
    if config.file_store is not None and config.file_store != 'memory':
        end_state = controller.get_state()
        # NOTE: the saved state does not include delegates events
        end_state.save_to_session(
            event_stream.sid, event_stream.file_store, event_stream.user_id
        )

    await controller.close(set_stop_state=False)

    state = controller.get_state()

    # save trajectories if applicable
    if config.save_trajectory_path is not None:
        # if save_trajectory_path is a folder, use session id as file name
        if os.path.isdir(config.save_trajectory_path):
            file_path = os.path.join(config.save_trajectory_path, sid + '.json')
        else:
            file_path = config.save_trajectory_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        histories = controller.get_trajectory(config.save_screenshots_in_trajectory)
        with open(file_path, 'w') as f:
            json.dump(histories, f, indent=4)

    return state


def auto_continue_response(
    state: State,
    encapsulate_solution: bool = False,
    try_parse: Callable[[Action | None], str] | None = None,
) -> str:
    """Default function to generate user responses.
    Tell the agent to proceed without asking for more input, or finish the interaction.
    """
    message = (
        'Please continue on whatever approach you think is suitable.\n'
        'If you think you have solved the task, please finish the interaction.\n'
        'IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN RESPONSE.\n'
    )
    return message


def load_replay_log(trajectory_path: str) -> tuple[list[Event] | None, Action]:
    """Load trajectory from given path, serialize it to a list of events, and return
    two things:
    1) A list of events except the first action
    2) First action (user message, a.k.a. initial task)
    """
    try:
        path = Path(trajectory_path).resolve()

        if not path.exists():
            raise ValueError(f'Trajectory file not found: {path}')

        if not path.is_file():
            raise ValueError(f'Trajectory path is a directory, not a file: {path}')

        with open(path, 'r', encoding='utf-8') as file:
            events = ReplayManager.get_replay_events(json.load(file))
            assert isinstance(events[0], MessageAction)
            return events[1:], events[0]
    except json.JSONDecodeError as e:
        raise ValueError(f'Invalid JSON format in {trajectory_path}: {e}')


if __name__ == '__main__':
    args = parse_arguments()

    config: OpenHandsConfig = setup_config_from_args(args)

    # Read task from file, CLI args, or stdin
    task_str = read_task(args, config.cli_multiline_input)

    initial_user_action: Action = NullAction()
    if config.replay_trajectory_path:
        if task_str:
            raise ValueError(
                'User-specified task is not supported under trajectory replay mode'
            )
    else:
        if not task_str:
            raise ValueError('No task provided. Please specify a task through -t, -f.')

        # Create actual initial user action
        initial_user_action = MessageAction(content=task_str)

    # Set session name
    session_name = args.name
    sid = generate_sid(config, session_name)

    asyncio.run(
        run_controller(
            config=config,
            initial_user_action=initial_user_action,
            sid=sid,
            fake_user_response_fn=None
            if args.no_auto_continue
            else auto_continue_response,
        )
    )
