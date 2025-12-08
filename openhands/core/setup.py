import hashlib
import os
import uuid
from typing import Callable

from pydantic import SecretStr

import openhands.agenthub  # noqa F401 (we import this to get the agents registered)
from openhands.controller import AgentController
from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import (
    OpenHandsConfig,
)
from openhands.core.config.config_utils import DEFAULT_WORKSPACE_MOUNT_PATH_IN_SANDBOX
from openhands.core.logger import openhands_logger as logger
from openhands.events import EventStream
from openhands.events.event import Event
from openhands.integrations.provider import (
    PROVIDER_TOKEN_TYPE,
    ProviderToken,
    ProviderType,
)
from openhands.llm.llm_registry import LLMRegistry
from openhands.memory.memory import Memory
from openhands.microagent.microagent import BaseMicroagent
from openhands.runtime import get_runtime_cls
from openhands.runtime.base import Runtime
from openhands.server.services.conversation_stats import ConversationStats
from openhands.storage import get_file_store
from openhands.storage.data_models.secrets import Secrets
from openhands.utils.async_utils import GENERAL_TIMEOUT, call_async_from_sync


def create_runtime(
    config: OpenHandsConfig,
    llm_registry: LLMRegistry | None = None,
    sid: str | None = None,
    headless_mode: bool = True,
    agent: Agent | None = None,
    git_provider_tokens: PROVIDER_TOKEN_TYPE | None = None,
) -> Runtime:
    """Create a runtime for the agent to run on.

    Args:
        config: The app config.
        sid: (optional) The session id. IMPORTANT: please don't set this unless you know what you're doing.
            Set it to incompatible value will cause unexpected behavior on RemoteRuntime.
        headless_mode: Whether the agent is run in headless mode. `create_runtime` is typically called within evaluation scripts,
            where we don't want to have the VSCode UI open, so it defaults to True.
        agent: (optional) The agent instance to use for configuring the runtime.

    Returns:
        The created Runtime instance (not yet connected or initialized).
    """
    # if sid is provided on the command line, use it as the name of the event stream
    # otherwise generate it on the basis of the configured jwt_secret
    # we can do this better, this is just so that the sid is retrieved when we want to restore the session
    session_id = sid or generate_sid(config)

    # set up the event stream
    file_store = get_file_store(config.file_store, config.file_store_path)
    event_stream = EventStream(session_id, file_store)

    # agent class
    if agent:
        agent_cls = type(agent)
    else:
        agent_cls = Agent.get_cls(config.default_agent)

    # runtime and tools
    runtime_cls = get_runtime_cls(config.runtime)
    logger.debug(f'Initializing runtime: {runtime_cls.__name__}')
    runtime: Runtime = runtime_cls(
        config=config,
        event_stream=event_stream,
        sid=session_id,
        plugins=agent_cls.sandbox_plugins,
        headless_mode=headless_mode,
        llm_registry=llm_registry or LLMRegistry(config),
        git_provider_tokens=git_provider_tokens,
    )

    # Log the plugins that have been registered with the runtime for debugging purposes
    logger.debug(
        f'Runtime created with plugins: {[plugin.name for plugin in runtime.plugins]}'
    )

    return runtime


def get_provider_tokens():
    """Retrieve provider tokens from environment variables and return them as a dictionary.

    Returns:
        A dictionary mapping ProviderType to ProviderToken if tokens are found, otherwise None.
    """
    # Collect provider tokens from environment variables if available
    provider_tokens = {}
    if 'GITHUB_TOKEN' in os.environ:
        github_token = SecretStr(os.environ['GITHUB_TOKEN'])
        provider_tokens[ProviderType.GITHUB] = ProviderToken(token=github_token)

    if 'GITLAB_TOKEN' in os.environ:
        gitlab_token = SecretStr(os.environ['GITLAB_TOKEN'])
        provider_tokens[ProviderType.GITLAB] = ProviderToken(token=gitlab_token)

    if 'BITBUCKET_TOKEN' in os.environ:
        bitbucket_token = SecretStr(os.environ['BITBUCKET_TOKEN'])
        provider_tokens[ProviderType.BITBUCKET] = ProviderToken(token=bitbucket_token)

    # Wrap provider tokens in Secrets if any tokens were found
    secret_store = (
        Secrets(provider_tokens=provider_tokens) if provider_tokens else None  # type: ignore[arg-type]
    )
    return secret_store.provider_tokens if secret_store else None


def initialize_repository_for_runtime(
    runtime: Runtime,
    immutable_provider_tokens: PROVIDER_TOKEN_TYPE | None = None,
    selected_repository: str | None = None,
) -> str | None:
    """Initialize the repository for the runtime by cloning or initializing it,
    running setup scripts, and setting up git hooks if present.

    Args:
        runtime: The runtime to initialize the repository for.
        immutable_provider_tokens: (optional) Provider tokens to use for authentication.
        selected_repository: (optional) The repository to use.

    Returns:
        The repository directory path if a repository was cloned, None otherwise.
    """
    # If provider tokens are not provided, attempt to retrieve them from the environment
    if not immutable_provider_tokens:
        immutable_provider_tokens = get_provider_tokens()

    logger.debug(f'Selected repository {selected_repository}.')

    # Clone or initialize the repository using the runtime
    repo_directory = call_async_from_sync(
        runtime.clone_or_init_repo,
        GENERAL_TIMEOUT,
        immutable_provider_tokens,
        selected_repository,
        None,
    )
    # Run setup script if it exists in the repository
    runtime.maybe_run_setup_script()
    # Set up git hooks if pre-commit.sh exists in the repository
    runtime.maybe_setup_git_hooks()

    return repo_directory


def create_memory(
    runtime: Runtime,
    event_stream: EventStream,
    sid: str,
    selected_repository: str | None = None,
    repo_directory: str | None = None,
    status_callback: Callable | None = None,
    conversation_instructions: str | None = None,
    working_dir: str = DEFAULT_WORKSPACE_MOUNT_PATH_IN_SANDBOX,
) -> Memory:
    """Create a memory for the agent to use.

    Args:
        runtime: The runtime to use.
        event_stream: The event stream it will subscribe to.
        sid: The session id.
        selected_repository: The repository to clone and start with, if any.
        repo_directory: The repository directory, if any.
        status_callback: Optional callback function to handle status updates.
        conversation_instructions: Optional instructions that are passed to the agent
    """
    memory = Memory(
        event_stream=event_stream,
        sid=sid,
        status_callback=status_callback,
    )

    memory.set_conversation_instructions(conversation_instructions)

    if runtime:
        # sets available hosts
        memory.set_runtime_info(runtime, {}, working_dir)

        # loads microagents from repo/.openhands/microagents
        microagents: list[BaseMicroagent] = runtime.get_microagents_from_selected_repo(
            selected_repository
        )
        memory.load_user_workspace_microagents(microagents)

        if selected_repository and repo_directory:
            memory.set_repository_info(selected_repository, repo_directory)

    return memory


def create_agent(config: OpenHandsConfig, llm_registry: LLMRegistry) -> Agent:
    agent_cls: type[Agent] = Agent.get_cls(config.default_agent)
    agent_config = config.get_agent_config(config.default_agent)
    # Pass the runtime information from the main config to the agent config
    agent_config.runtime = config.runtime
    config.get_llm_config_from_agent(config.default_agent)
    agent = agent_cls(config=agent_config, llm_registry=llm_registry)
    return agent


def create_controller(
    agent: Agent,
    runtime: Runtime,
    config: OpenHandsConfig,
    conversation_stats: ConversationStats,
    headless_mode: bool = True,
    replay_events: list[Event] | None = None,
) -> tuple[AgentController, State | None]:
    event_stream = runtime.event_stream
    initial_state = None
    try:
        logger.debug(
            f'Trying to restore agent state from session {event_stream.sid} if available'
        )
        initial_state = State.restore_from_session(
            event_stream.sid, event_stream.file_store
        )
    except Exception as e:
        logger.debug(f'Cannot restore agent state: {e}')

    controller = AgentController(
        agent=agent,
        conversation_stats=conversation_stats,
        iteration_delta=config.max_iterations,
        budget_per_task_delta=config.max_budget_per_task,
        agent_to_llm_config=config.get_agent_to_llm_config_map(),
        event_stream=event_stream,
        initial_state=initial_state,
        headless_mode=headless_mode,
        confirmation_mode=config.security.confirmation_mode,
        replay_events=replay_events,
        security_analyzer=runtime.security_analyzer,
    )
    return (controller, initial_state)


def generate_sid(config: OpenHandsConfig, session_name: str | None = None) -> str:
    """Generate a session id based on the session name and the jwt secret.

    The session ID is kept short to ensure Kubernetes resource names don't exceed
    the 63-character limit when prefixed with 'openhands-runtime-' (18 chars).
    Total length is limited to 32 characters to allow for suffixes like '-svc', '-pvc'.
    """
    session_name = session_name or str(uuid.uuid4())
    jwt_secret = config.jwt_secret

    hash_str = hashlib.sha256(f'{session_name}{jwt_secret}'.encode('utf-8')).hexdigest()

    # Limit total session ID length to 32 characters for Kubernetes compatibility:
    # - 'openhands-runtime-' (18 chars) + session_id (32 chars) = 50 chars
    # - Leaves 13 chars for suffixes like '-svc' (4), '-pvc' (4), '-ingress-code' (13)
    if len(session_name) > 16:
        # If session_name is too long, use first 16 chars + 15-char hash for better readability
        # e.g., "vscode-extension" -> "vscode-extensio-{15-char-hash}"
        session_id = f'{session_name[:16]}-{hash_str[:15]}'
    else:
        # If session_name is short enough, use it + remaining space for hash
        remaining_chars = 32 - len(session_name) - 1  # -1 for the dash
        session_id = f'{session_name}-{hash_str[:remaining_chars]}'

    return session_id[:32]  # Ensure we never exceed 32 characters
