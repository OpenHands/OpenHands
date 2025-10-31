import asyncio
import atexit
import copy
import json
import os
import random
import shlex
import shutil
import string
import tempfile
from abc import abstractmethod
from pathlib import Path
from types import MappingProxyType
from typing import Callable, cast
from zipfile import ZipFile

import httpx

from openhands.core.config import OpenHandsConfig, SandboxConfig
from openhands.core.config.mcp_config import MCPConfig, MCPStdioServerConfig
from openhands.core.exceptions import (
    AgentRuntimeDisconnectedError,
)
from openhands.core.logger import openhands_logger as logger
from openhands.events import EventSource, EventStream, EventStreamSubscriber
from openhands.events.action import (
    Action,
    ActionConfirmationStatus,
    AgentThinkAction,
    BrowseInteractiveAction,
    BrowseURLAction,
    CmdRunAction,
    FileEditAction,
    FileReadAction,
    FileWriteAction,
    IPythonRunCellAction,
    TaskTrackingAction,
)
from openhands.events.action.mcp import MCPAction
from openhands.events.event import Event
from openhands.events.observation import (
    AgentThinkObservation,
    CmdOutputObservation,
    ErrorObservation,
    FileReadObservation,
    NullObservation,
    Observation,
    TaskTrackingObservation,
    UserRejectObservation,
)
from openhands.events.serialization.action import ACTION_TYPE_TO_CLASS
from openhands.integrations.provider import (
    PROVIDER_TOKEN_TYPE,
    ProviderHandler,
    ProviderType,
)
from openhands.integrations.service_types import AuthenticationError
from openhands.llm.llm_registry import LLMRegistry
from openhands.microagent import (
    BaseMicroagent,
    load_microagents_from_dir,
)
from openhands.runtime.plugins import (
    JupyterRequirement,
    PluginRequirement,
    VSCodeRequirement,
)
from openhands.runtime.runtime_status import RuntimeStatus
from openhands.runtime.utils.edit import FileEditRuntimeMixin
from openhands.runtime.utils.git_handler import CommandResult, GitHandler
from openhands.security import SecurityAnalyzer, options
from openhands.storage.locations import get_conversation_dir
from openhands.utils.async_utils import (
    GENERAL_TIMEOUT,
    call_async_from_sync,
    call_sync_from_async,
)


def _default_env_vars(sandbox_config: SandboxConfig) -> dict[str, str]:
    ret = {}
    for key in os.environ:
        if key.startswith('SANDBOX_ENV_'):
            sandbox_key = key.removeprefix('SANDBOX_ENV_')
            ret[sandbox_key] = os.environ[key]
    if sandbox_config.enable_auto_lint:
        ret['ENABLE_AUTO_LINT'] = 'true'
    return ret


class Runtime(FileEditRuntimeMixin):
    """Abstract base class for agent runtime environments.

    This is an extension point in OpenHands that allows applications to customize how
    agents interact with the external environment. The runtime provides a sandbox with:
    - Bash shell access
    - Browser interaction
    - Filesystem operations
    - Git operations
    - Environment variable management

    Applications can substitute their own implementation by:
    1. Creating a class that inherits from Runtime
    2. Implementing all required methods
    3. Setting the runtime name in configuration or using get_runtime_cls()

    The class is instantiated via get_impl() in get_runtime_cls().

    Built-in implementations include:
    - DockerRuntime: Containerized environment using Docker
    - RemoteRuntime: Remote execution environment
    - LocalRuntime: Local execution for development
    - KubernetesRuntime: Kubernetes-based execution environment
    - CLIRuntime: Command-line interface runtime

    Args:
        sid: Session ID that uniquely identifies the current user session
    """

    sid: str
    config: OpenHandsConfig
    initial_env_vars: dict[str, str]
    attach_to_existing: bool
    status_callback: Callable[[str, RuntimeStatus, str], None] | None
    runtime_status: RuntimeStatus | None
    _runtime_initialized: bool = False
    security_analyzer: 'SecurityAnalyzer | None' = None

    def __init__(
        self,
        config: OpenHandsConfig,
        event_stream: EventStream,
        llm_registry: LLMRegistry,
        sid: str = 'default',
        plugins: list[PluginRequirement] | None = None,
        env_vars: dict[str, str] | None = None,
        status_callback: Callable[[str, RuntimeStatus, str], None] | None = None,
        attach_to_existing: bool = False,
        headless_mode: bool = False,
        user_id: str | None = None,
        git_provider_tokens: PROVIDER_TOKEN_TYPE | None = None,
    ):
        self.git_handler = GitHandler(
            execute_shell_fn=self._execute_shell_fn_git_handler,
            create_file_fn=self._create_file_fn_git_handler,
        )
        self.sid = sid
        self.event_stream = event_stream
        if event_stream:
            event_stream.subscribe(
                EventStreamSubscriber.RUNTIME, self.on_event, self.sid
            )
        self.plugins = (
            copy.deepcopy(plugins) if plugins is not None and len(plugins) > 0 else []
        )
        # add VSCode plugin if not in headless mode
        if not headless_mode:
            self.plugins.append(VSCodeRequirement())

        self.status_callback = status_callback
        self.attach_to_existing = attach_to_existing

        self.config = copy.deepcopy(config)
        atexit.register(self.close)

        self.initial_env_vars = _default_env_vars(config.sandbox)
        if env_vars is not None:
            self.initial_env_vars.update(env_vars)

        self.provider_handler = ProviderHandler(
            provider_tokens=git_provider_tokens
            or cast(PROVIDER_TOKEN_TYPE, MappingProxyType({})),
            external_auth_id=user_id,
            external_token_manager=True,
        )
        raw_env_vars: dict[str, str] = call_async_from_sync(
            self.provider_handler.get_env_vars, GENERAL_TIMEOUT, True, None, False
        )
        self.initial_env_vars.update(raw_env_vars)

        self._vscode_enabled = any(
            isinstance(plugin, VSCodeRequirement) for plugin in self.plugins
        )

        # Load mixins
        FileEditRuntimeMixin.__init__(
            self,
            enable_llm_editor=config.get_agent_config().enable_llm_editor,
            llm_registry=llm_registry,
        )

        self.user_id = user_id
        self.git_provider_tokens = git_provider_tokens
        self.runtime_status = None

        # Initialize security analyzer
        self.security_analyzer = None
        if self.config.security.security_analyzer:
            analyzer_cls = options.SecurityAnalyzers.get(
                self.config.security.security_analyzer, SecurityAnalyzer
            )
            self.security_analyzer = analyzer_cls()
            self.security_analyzer.set_event_stream(self.event_stream)
            logger.debug(
                f'Security analyzer {analyzer_cls.__name__} initialized for runtime {self.sid}'
            )

    @property
    def runtime_initialized(self) -> bool:
        return self._runtime_initialized

    def setup_initial_env(self) -> None:
        if self.attach_to_existing:
            return
        logger.debug(f'Adding env vars: {self.initial_env_vars.keys()}')
        self.add_env_vars(self.initial_env_vars)
        if self.config.sandbox.runtime_startup_env_vars:
            self.add_env_vars(self.config.sandbox.runtime_startup_env_vars)

        # Configure git settings
        self._setup_git_config()

    def close(self) -> None:
        """This should only be called by conversation manager or closing the session.
        If called for instance by error handling, it could prevent recovery.
        """
        pass

    @classmethod
    async def delete(cls, conversation_id: str) -> None:
        pass

    def log(self, level: str, message: str) -> None:
        message = f'[runtime {self.sid}] {message}'
        getattr(logger, level)(message, stacklevel=2)

    def set_runtime_status(
        self, runtime_status: RuntimeStatus, msg: str = '', level: str = 'info'
    ):
        """Sends a status message if the callback function was provided."""
        self.runtime_status = runtime_status
        if self.status_callback:
            self.status_callback(level, runtime_status, msg)

    # ====================================================================

    def add_env_vars(self, env_vars: dict[str, str]) -> None:
        env_vars = {key.upper(): value for key, value in env_vars.items()}

        # Add env vars to the IPython shell (if Jupyter is used)
        if any(isinstance(plugin, JupyterRequirement) for plugin in self.plugins):
            code = 'import os\n'
            for key, value in env_vars.items():
                # Note: json.dumps gives us nice escaping for free
                code += f'os.environ["{key}"] = {json.dumps(value)}\n'
            code += '\n'
            self.run_ipython(IPythonRunCellAction(code))
            # Note: we don't log the vars values, they're leaking info
            logger.debug('Added env vars to IPython')

        # Check if we're on Windows
        import os
        import sys

        is_windows = os.name == 'nt' or sys.platform == 'win32'

        if is_windows:
            # Add env vars using PowerShell commands for Windows
            cmd = ''
            for key, value in env_vars.items():
                # Use PowerShell's $env: syntax for environment variables
                # Note: json.dumps gives us nice escaping for free
                cmd += f'$env:{key} = {json.dumps(value)}; '

            if not cmd:
                return

            cmd = cmd.strip()
            logger.debug('Adding env vars to PowerShell')  # don't log the values

            obs = self.run(CmdRunAction(cmd))
            if not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0:
                raise RuntimeError(
                    f'Failed to add env vars [{env_vars.keys()}] to environment: {obs.content}'
                )

            # We don't add to profile persistence on Windows as it's more complex
            # and varies between PowerShell versions
            logger.debug(f'Added env vars to PowerShell session: {env_vars.keys()}')

        else:
            # Original bash implementation for Unix systems
            cmd = ''
            bashrc_cmd = ''
            for key, value in env_vars.items():
                # Note: json.dumps gives us nice escaping for free
                cmd += f'export {key}={json.dumps(value)}; '
                # Add to .bashrc if not already present
                bashrc_cmd += f'touch ~/.bashrc; grep -q "^export {key}=" ~/.bashrc || echo "export {key}={json.dumps(value)}" >> ~/.bashrc; '

            if not cmd:
                return

            cmd = cmd.strip()
            logger.debug('Adding env vars to bash')  # don't log the values

            obs = self.run(CmdRunAction(cmd))
            if not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0:
                raise RuntimeError(
                    f'Failed to add env vars [{env_vars.keys()}] to environment: {obs.content}'
                )

            # Add to .bashrc for persistence
            bashrc_cmd = bashrc_cmd.strip()
            logger.debug(f'Adding env var to .bashrc: {env_vars.keys()}')
            obs = self.run(CmdRunAction(bashrc_cmd))
            if not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0:
                raise RuntimeError(
                    f'Failed to add env vars [{env_vars.keys()}] to .bashrc: {obs.content}'
                )

    def on_event(self, event: Event) -> None:
        if isinstance(event, Action):
            asyncio.get_event_loop().run_until_complete(self._handle_action(event))

    async def _export_latest_git_provider_tokens(self, event: Action) -> None:
        """Refresh runtime provider tokens when agent attemps to run action with provider token"""
        providers_called = ProviderHandler.check_cmd_action_for_provider_token_ref(
            event
        )

        if not providers_called:
            return

        provider_handler = ProviderHandler(
            provider_tokens=self.git_provider_tokens
            or cast(PROVIDER_TOKEN_TYPE, MappingProxyType({})),
            external_auth_id=self.user_id,
            external_token_manager=True,
            session_api_key=self.session_api_key,
            sid=self.sid,
        )

        logger.info(
            f'Fetching latest provider tokens for runtime: {self.sid}, '
            f'providers: {providers_called}'
        )
        env_vars = await provider_handler.get_env_vars(
            providers=providers_called, expose_secrets=False, get_latest=True
        )
        logger.info(
            f'Successfully fetched {len(env_vars)} token(s) for runtime: {self.sid}'
        )

        if len(env_vars) == 0:
            return

        try:
            if self.event_stream:
                await provider_handler.set_event_stream_secrets(
                    self.event_stream, env_vars=env_vars
                )
            self.add_env_vars(provider_handler.expose_env_vars(env_vars))
        except Exception as e:
            logger.error(
                f'Failed to export latest github token to runtime: {self.sid}, {e}',
                exc_info=True,
            )

    async def _handle_action(self, event: Action) -> None:
        if event.timeout is None:
            # We don't block the command if this is a default timeout action
            event.set_hard_timeout(self.config.sandbox.timeout, blocking=False)
        assert event.timeout is not None
        try:
            await self._export_latest_git_provider_tokens(event)
            if isinstance(event, MCPAction):
                observation: Observation = await self.call_tool_mcp(event)
            else:
                observation = await call_sync_from_async(self.run_action, event)
        except PermissionError as e:
            # Handle PermissionError specially - convert to ErrorObservation
            # so the agent can receive feedback and continue execution
            observation = ErrorObservation(content=str(e))
        except (httpx.NetworkError, AgentRuntimeDisconnectedError) as e:
            runtime_status = RuntimeStatus.ERROR_RUNTIME_DISCONNECTED
            error_message = f'{type(e).__name__}: {str(e)}'
            self.log('error', f'Unexpected error while running action: {error_message}')
            self.log('error', f'Problematic action: {str(event)}')
            self.set_runtime_status(runtime_status, error_message, level='error')
            return
        except Exception as e:
            runtime_status = RuntimeStatus.ERROR
            error_message = f'{type(e).__name__}: {str(e)}'
            self.log('error', f'Unexpected error while running action: {error_message}')
            self.log('error', f'Problematic action: {str(event)}')
            self.set_runtime_status(runtime_status, error_message, level='error')
            return

        observation._cause = event.id  # type: ignore[attr-defined]
        observation.tool_call_metadata = event.tool_call_metadata

        # this might be unnecessary, since source should be set by the event stream when we're here
        source = event.source if event.source else EventSource.AGENT
        if isinstance(observation, NullObservation):
            # don't add null observations to the event stream
            return
        self.event_stream.add_event(observation, source)  # type: ignore[arg-type]

    async def clone_or_init_repo(
        self,
        git_provider_tokens: PROVIDER_TOKEN_TYPE | None,
        selected_repository: str | None,
        selected_branch: str | None,
    ) -> str:
        if not selected_repository:
            if self.config.init_git_in_empty_workspace:
                logger.debug(
                    'No repository selected. Initializing a new git repository in the workspace.'
                )
                action = CmdRunAction(
                    command=f'git init && git config --global --add safe.directory {self.workspace_root}'
                )
                await call_sync_from_async(self.run_action, action)
            else:
                logger.info(
                    'In workspace mount mode, not initializing a new git repository.'
                )
            return ''

        remote_repo_url = await self.provider_handler.get_authenticated_git_url(
            selected_repository
        )

        if not remote_repo_url:
            raise ValueError('Missing either Git token or valid repository')

        if self.status_callback:
            self.status_callback(
                'info', RuntimeStatus.SETTING_UP_WORKSPACE, 'Setting up workspace...'
            )

        dir_name = selected_repository.split('/')[-1]

        # Generate a random branch name to avoid conflicts
        random_str = ''.join(
            random.choices(string.ascii_lowercase + string.digits, k=8)
        )
        openhands_workspace_branch = f'openhands-workspace-{random_str}'

        repo_path = self.workspace_root / dir_name
        quoted_repo_path = shlex.quote(str(repo_path))
        quoted_remote_repo_url = shlex.quote(remote_repo_url)

        # Clone repository command
        clone_command = f'git clone {quoted_remote_repo_url} {quoted_repo_path}'

        # Checkout to appropriate branch
        checkout_command = (
            f'git checkout {selected_branch}'
            if selected_branch
            else f'git checkout -b {openhands_workspace_branch}'
        )

        clone_action = CmdRunAction(command=clone_command)
        await call_sync_from_async(self.run_action, clone_action)

        cd_checkout_action = CmdRunAction(
            command=f'cd {quoted_repo_path} && {checkout_command}'
        )
        action = cd_checkout_action
        self.log('info', f'Cloning repo: {selected_repository}')
        await call_sync_from_async(self.run_action, action)

        if remote_repo_url:
            set_remote_action = CmdRunAction(
                command=(
                    f'cd {quoted_repo_path} && '
                    f'git remote set-url origin {quoted_remote_repo_url}'
                )
            )
            obs = await call_sync_from_async(self.run_action, set_remote_action)
            if isinstance(obs, CmdOutputObservation) and obs.exit_code == 0:
                self.log(
                    'info',
                    f'Set git remote origin to authenticated URL for {selected_repository}',
                )
            else:
                self.log(
                    'warning',
                    (
                        'Failed to set git remote origin while ensuring fresh token '
                        f'for {selected_repository}: '
                        f'{obs.content if isinstance(obs, CmdOutputObservation) else "unknown error"}'
                    ),
                )

        return dir_name

    def maybe_run_setup_script(self):
        """Run .openhands/setup.sh if it exists in the workspace or repository."""
        setup_script = '.openhands/setup.sh'
        read_obs = self.read(FileReadAction(path=setup_script))
        if isinstance(read_obs, ErrorObservation):
            return

        if self.status_callback:
            self.status_callback(
                'info', RuntimeStatus.SETTING_UP_WORKSPACE, 'Setting up workspace...'
            )

        # setup scripts time out after 10 minutes
        action = CmdRunAction(
            f'chmod +x {setup_script} && source {setup_script}',
            blocking=True,
            hidden=True,
        )
        action.set_hard_timeout(600)

        # Add the action to the event stream as an ENVIRONMENT event
        source = EventSource.ENVIRONMENT
        self.event_stream.add_event(action, source)

        # Execute the action
        self.run_action(action)

    @property
    def workspace_root(self) -> Path:
        """Return the workspace root path."""
        return Path(self.config.workspace_mount_path_in_sandbox)

    def maybe_setup_git_hooks(self):
        """Set up git hooks if .openhands/pre-commit.sh exists in the workspace or repository."""
        pre_commit_script = '.openhands/pre-commit.sh'
        read_obs = self.read(FileReadAction(path=pre_commit_script))
        if isinstance(read_obs, ErrorObservation):
            return

        if self.status_callback:
            self.status_callback(
                'info', RuntimeStatus.SETTING_UP_GIT_HOOKS, 'Setting up git hooks...'
            )

        # Ensure the git hooks directory exists
        action = CmdRunAction('mkdir -p .git/hooks')
        obs = self.run_action(action)
        if isinstance(obs, CmdOutputObservation) and obs.exit_code != 0:
            self.log('error', f'Failed to create git hooks directory: {obs.content}')
            return

        # Make the pre-commit script executable
        action = CmdRunAction(f'chmod +x {pre_commit_script}')
        obs = self.run_action(action)
        if isinstance(obs, CmdOutputObservation) and obs.exit_code != 0:
            self.log(
                'error', f'Failed to make pre-commit script executable: {obs.content}'
            )
            return

        # Check if there's an existing pre-commit hook
        pre_commit_hook = '.git/hooks/pre-commit'
        pre_commit_local = '.git/hooks/pre-commit.local'

        # Read the existing pre-commit hook if it exists
        read_obs = self.read(FileReadAction(path=pre_commit_hook))
        if not isinstance(read_obs, ErrorObservation):
            # If the existing hook wasn't created by OpenHands, preserve it
            if 'This hook was installed by OpenHands' not in read_obs.content:
                self.log('info', 'Preserving existing pre-commit hook')
                # Move the existing hook to pre-commit.local
                action = CmdRunAction(f'mv {pre_commit_hook} {pre_commit_local}')
                obs = self.run_action(action)
                if isinstance(obs, CmdOutputObservation) and obs.exit_code != 0:
                    self.log(
                        'error',
                        f'Failed to preserve existing pre-commit hook: {obs.content}',
                    )
                    return

                # Make it executable
                action = CmdRunAction(f'chmod +x {pre_commit_local}')
                obs = self.run_action(action)
                if isinstance(obs, CmdOutputObservation) and obs.exit_code != 0:
                    self.log(
                        'error',
                        f'Failed to make preserved hook executable: {obs.content}',
                    )
                    return

        # Create the pre-commit hook that calls our script
        pre_commit_hook_content = f"""#!/bin/bash
# This hook was installed by OpenHands
# It calls the pre-commit script in the .openhands directory

if [ -x "{pre_commit_script}" ]; then
    source "{pre_commit_script}"
    exit $?
else
    echo "Warning: {pre_commit_script} not found or not executable"
    exit 0
fi
"""

        # Write the pre-commit hook
        write_obs = self.write(
            FileWriteAction(path=pre_commit_hook, content=pre_commit_hook_content)
        )
        if isinstance(write_obs, ErrorObservation):
            self.log('error', f'Failed to write pre-commit hook: {write_obs.content}')
            return

        # Make the pre-commit hook executable
        action = CmdRunAction(f'chmod +x {pre_commit_hook}')
        obs = self.run_action(action)
        if isinstance(obs, CmdOutputObservation) and obs.exit_code != 0:
            self.log(
                'error', f'Failed to make pre-commit hook executable: {obs.content}'
            )
            return

        self.log('info', 'Git pre-commit hook installed successfully')

    def _load_microagents_from_directory(
        self, microagents_dir: Path, source_description: str
    ) -> list[BaseMicroagent]:
        """Load microagents from a directory.

        Args:
            microagents_dir: Path to the directory containing microagents
            source_description: Description of the source for logging purposes

        Returns:
            A list of loaded microagents
        """
        loaded_microagents: list[BaseMicroagent] = []

        self.log(
            'info',
            f'Attempting to list files in {source_description} microagents directory: {microagents_dir}',
        )

        files = self.list_files(str(microagents_dir))

        if not files:
            self.log(
                'debug',
                f'No files found in {source_description} microagents directory: {microagents_dir}',
            )
            return loaded_microagents

        self.log(
            'info',
            f'Found {len(files)} files in {source_description} microagents directory',
        )
        zip_path = self.copy_from(str(microagents_dir))
        microagent_folder = tempfile.mkdtemp()

        try:
            with ZipFile(zip_path, 'r') as zip_file:
                zip_file.extractall(microagent_folder)

            zip_path.unlink()
            repo_agents, knowledge_agents = load_microagents_from_dir(microagent_folder)

            self.log(
                'info',
                f'Loaded {len(repo_agents)} repo agents and {len(knowledge_agents)} knowledge agents from {source_description}',
            )

            loaded_microagents.extend(repo_agents.values())
            loaded_microagents.extend(knowledge_agents.values())
        except Exception as e:
            self.log('error', f'Failed to load agents from {source_description}: {e}')
        finally:
            shutil.rmtree(microagent_folder)

        return loaded_microagents

    def _is_gitlab_repository(self, repo_name: str) -> bool:
        """Check if a repository is hosted on GitLab.

        Args:
            repo_name: Repository name (e.g., "gitlab.com/org/repo" or "org/repo")

        Returns:
            True if the repository is hosted on GitLab, False otherwise
        """
        try:
            provider_handler = ProviderHandler(
                self.git_provider_tokens or MappingProxyType({})
            )
            repository = call_async_from_sync(
                provider_handler.verify_repo_provider,
                GENERAL_TIMEOUT,
                repo_name,
            )
            return repository.git_provider == ProviderType.GITLAB
        except Exception:
            # If we can't determine the provider, assume it's not GitLab
            # This is a safe fallback since we'll just use the default .openhands
            return False

    def _is_azure_devops_repository(self, repo_name: str) -> bool:
        """Check if a repository is hosted on Azure DevOps.

        Args:
            repo_name: Repository name (e.g., "org/project/repo")

        Returns:
            True if the repository is hosted on Azure DevOps, False otherwise
        """
        try:
            provider_handler = ProviderHandler(
                self.git_provider_tokens or MappingProxyType({})
            )
            repository = call_async_from_sync(
                provider_handler.verify_repo_provider,
                GENERAL_TIMEOUT,
                repo_name,
            )
            return repository.git_provider == ProviderType.AZURE_DEVOPS
        except Exception:
            # If we can't determine the provider, assume it's not Azure DevOps
            return False

    def get_microagents_from_org_or_user(
        self, selected_repository: str
    ) -> list[BaseMicroagent]:
        """Load microagents from the organization or user level repository.

        For example, if the repository is github.com/acme-co/api, this will check if
        github.com/acme-co/.openhands exists. If it does, it will clone it and load
        the microagents from the ./microagents/ folder.

        For GitLab repositories, it will use openhands-config instead of .openhands
        since GitLab doesn't support repository names starting with non-alphanumeric
        characters.

        For Azure DevOps repositories, it will use org/openhands-config/openhands-config
        format to match Azure DevOps's three-part repository structure (org/project/repo).

        Args:
            selected_repository: The repository path (e.g., "github.com/acme-co/api")

        Returns:
            A list of loaded microagents from the org/user level repository
        """
        loaded_microagents: list[BaseMicroagent] = []

        self.log(
            'debug',
            f'Starting org-level microagent loading for repository: {selected_repository}',
        )

        repo_parts = selected_repository.split('/')

        if len(repo_parts) < 2:
            self.log(
                'warning',
                f'Repository path has insufficient parts ({len(repo_parts)} < 2), skipping org-level microagents',
            )
            return loaded_microagents

        # Determine repository type
        is_azure_devops = self._is_azure_devops_repository(selected_repository)
        is_gitlab = self._is_gitlab_repository(selected_repository)

        # Extract the org/user name
        # Azure DevOps format: org/project/repo (3 parts) - extract org (first part)
        # GitHub/GitLab/Bitbucket format: owner/repo (2 parts) - extract owner (first part)
        if is_azure_devops and len(repo_parts) >= 3:
            org_name = repo_parts[0]  # Get org from org/project/repo
        else:
            org_name = repo_parts[-2]  # Get owner from owner/repo

        self.log(
            'info',
            f'Extracted org/user name: {org_name}',
        )
        self.log(
            'debug',
            f'Repository type detection - is_gitlab: {is_gitlab}, is_azure_devops: {is_azure_devops}',
        )

        # For GitLab and Azure DevOps, use openhands-config (since .openhands is not a valid repo name)
        # For other providers, use .openhands
        if is_gitlab:
            org_openhands_repo = f'{org_name}/openhands-config'
        elif is_azure_devops:
            # Azure DevOps format: org/project/repo
            # For org-level config, use: org/openhands-config/openhands-config
            org_openhands_repo = f'{org_name}/openhands-config/openhands-config'
        else:
            org_openhands_repo = f'{org_name}/.openhands'

        self.log(
            'info',
            f'Checking for org-level microagents at {org_openhands_repo}',
        )

        # Try to clone the org-level repo
        try:
            # Create a temporary directory for the org-level repo
            org_repo_dir = self.workspace_root / f'org_openhands_{org_name}'
            self.log(
                'debug',
                f'Creating temporary directory for org repo: {org_repo_dir}',
            )

            # Get authenticated URL and do a shallow clone (--depth 1) for efficiency
            try:
                remote_url = call_async_from_sync(
                    self.provider_handler.get_authenticated_git_url,
                    GENERAL_TIMEOUT,
                    org_openhands_repo,
                    is_optional=True,
                )
            except AuthenticationError as e:
                self.log(
                    'debug',
                    f'org-level microagent directory {org_openhands_repo} not found: {str(e)}',
                )
                raise
            except Exception as e:
                self.log(
                    'debug',
                    f'Failed to get authenticated URL for {org_openhands_repo}: {str(e)}',
                )
                raise

            clone_cmd = (
                f'GIT_TERMINAL_PROMPT=0 git clone --depth 1 {remote_url} {org_repo_dir}'
            )
            self.log(
                'info',
                'Executing clone command for org-level repo',
            )

            action = CmdRunAction(command=clone_cmd)
            obs = self.run_action(action)

            if isinstance(obs, CmdOutputObservation) and obs.exit_code == 0:
                self.log(
                    'info',
                    f'Successfully cloned org-level microagents from {org_openhands_repo}',
                )

                # Load microagents from the org-level repo
                org_microagents_dir = org_repo_dir / 'microagents'
                self.log(
                    'info',
                    f'Looking for microagents in directory: {org_microagents_dir}',
                )

                loaded_microagents = self._load_microagents_from_directory(
                    org_microagents_dir, 'org-level'
                )

                self.log(
                    'info',
                    f'Loaded {len(loaded_microagents)} microagents from org-level repository {org_openhands_repo}',
                )

                # Clean up the org repo directory
                action = CmdRunAction(f'rm -rf {org_repo_dir}')
                self.run_action(action)
            else:
                clone_error_msg = (
                    obs.content
                    if isinstance(obs, CmdOutputObservation)
                    else 'Unknown error'
                )
                exit_code = (
                    obs.exit_code if isinstance(obs, CmdOutputObservation) else 'N/A'
                )
                self.log(
                    'info',
                    f'No org-level microagents found at {org_openhands_repo} (exit_code: {exit_code})',
                )
                self.log(
                    'debug',
                    f'Clone command output: {clone_error_msg}',
                )

        except AuthenticationError as e:
            self.log(
                'debug',
                f'org-level microagent directory {org_openhands_repo} not found: {str(e)}',
            )
        except Exception as e:
            self.log(
                'debug',
                f'Error loading org-level microagents from {org_openhands_repo}: {str(e)}',
            )

        return loaded_microagents

    def get_microagents_from_selected_repo(
        self, selected_repository: str | None
    ) -> list[BaseMicroagent]:
        """Load microagents from the selected repository.
        If selected_repository is None, load microagents from the current workspace.
        This is the main entry point for loading microagents.

        This method also checks for user/org level microagents stored in a repository.
        For example, if the repository is github.com/acme-co/api, it will also check for
        github.com/acme-co/.openhands and load microagents from there if it exists.

        For GitLab repositories, it will use openhands-config instead of .openhands
        since GitLab doesn't support repository names starting with non-alphanumeric
        characters.
        """
        loaded_microagents: list[BaseMicroagent] = []
        microagents_dir = self.workspace_root / '.openhands' / 'microagents'
        repo_root = None

        # Check for user/org level microagents if a repository is selected
        if selected_repository:
            # Load microagents from the org/user level repository
            org_microagents = self.get_microagents_from_org_or_user(selected_repository)
            loaded_microagents.extend(org_microagents)

            # Continue with repository-specific microagents
            repo_root = self.workspace_root / selected_repository.split('/')[-1]
            microagents_dir = repo_root / '.openhands' / 'microagents'

        self.log(
            'info',
            f'Selected repo: {selected_repository}, loading microagents from {microagents_dir} (inside runtime)',
        )

        # Legacy Repo Instructions
        # Check for legacy .openhands_instructions file
        obs = self.read(
            FileReadAction(path=str(self.workspace_root / '.openhands_instructions'))
        )
        if isinstance(obs, ErrorObservation) and repo_root is not None:
            # If the instructions file is not found in the workspace root, try to load it from the repo root
            self.log(
                'debug',
                f'.openhands_instructions not present, trying to load from repository microagents_dir={microagents_dir}',
            )
            obs = self.read(
                FileReadAction(path=str(repo_root / '.openhands_instructions'))
            )

        if isinstance(obs, FileReadObservation):
            self.log('info', 'openhands_instructions microagent loaded.')
            loaded_microagents.append(
                BaseMicroagent.load(
                    path='.openhands_instructions',
                    microagent_dir=None,
                    file_content=obs.content,
                )
            )

        # Load microagents from directory
        repo_microagents = self._load_microagents_from_directory(
            microagents_dir, 'repository'
        )
        loaded_microagents.extend(repo_microagents)

        return loaded_microagents

    def run_action(self, action: Action) -> Observation:
        """Run an action and return the resulting observation.
        If the action is not runnable in any runtime, a NullObservation is returned.
        If the action is not supported by the current runtime, an ErrorObservation is returned.
        """
        if not action.runnable:
            if isinstance(action, AgentThinkAction):
                return AgentThinkObservation('Your thought has been logged.')
            elif isinstance(action, TaskTrackingAction):
                # Get the session-specific task file path
                conversation_dir = get_conversation_dir(
                    self.sid, self.event_stream.user_id
                )
                task_file_path = f'{conversation_dir}TASKS.md'

                if action.command == 'plan':
                    # Write the serialized task list to the session directory
                    content = '# Task List\n\n'
                    for i, task in enumerate(action.task_list, 1):
                        status_icon = {
                            'todo': '⏳',
                            'in_progress': '🔄',
                            'done': '✅',
                        }.get(task.get('status', 'todo'), '⏳')
                        content += f'{i}. {status_icon} {task.get("title", "")}\n{task.get("notes", "")}\n'

                    try:
                        self.event_stream.file_store.write(task_file_path, content)
                        return TaskTrackingObservation(
                            content=f'Task list has been updated with {len(action.task_list)} items. Stored in session directory: {task_file_path}',
                            command=action.command,
                            task_list=action.task_list,
                        )
                    except Exception as e:
                        return ErrorObservation(
                            f'Failed to write task list to session directory {task_file_path}: {str(e)}'
                        )

                elif action.command == 'view':
                    # Read the TASKS.md file from the session directory
                    try:
                        content = self.event_stream.file_store.read(task_file_path)
                        return TaskTrackingObservation(
                            content=content,
                            command=action.command,
                            task_list=[],  # Empty for view command
                        )
                    except FileNotFoundError:
                        return TaskTrackingObservation(
                            command=action.command,
                            task_list=[],
                            content='No task list found. Use the "plan" command to create one.',
                        )
                    except Exception as e:
                        return TaskTrackingObservation(
                            command=action.command,
                            task_list=[],
                            content=f'Failed to read the task list from session directory {task_file_path}. Error: {str(e)}',
                        )
                else:
                    return TaskTrackingObservation(
                        command=action.command,
                        task_list=[],
                        content=f'Unknown command: {action.command}',
                    )
            return NullObservation('')
        if (
            hasattr(action, 'confirmation_state')
            and action.confirmation_state
            == ActionConfirmationStatus.AWAITING_CONFIRMATION
        ):
            return NullObservation('')
        action_type = action.action  # type: ignore[attr-defined]
        if action_type not in ACTION_TYPE_TO_CLASS:
            return ErrorObservation(f'Action {action_type} does not exist.')
        if not hasattr(self, action_type):
            return ErrorObservation(
                f'Action {action_type} is not supported in the current runtime.'
            )
        if (
            getattr(action, 'confirmation_state', None)
            == ActionConfirmationStatus.REJECTED
        ):
            return UserRejectObservation(
                'Action has been rejected by the user! Waiting for further user input.'
            )
        observation = getattr(self, action_type)(action)
        return observation

    # ====================================================================
    # Context manager
    # ====================================================================

    def __enter__(self) -> 'Runtime':
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @abstractmethod
    async def connect(self) -> None:
        pass

    def _setup_git_config(self) -> None:
        """Configure git user settings during initial environment setup.

        This method is called automatically during setup_initial_env() to ensure
        git configuration is applied to the runtime environment.
        """
        # Get git configuration from config
        git_user_name = self.config.git_user_name
        git_user_email = self.config.git_user_email

        # Skip git configuration for CLI runtime to preserve user's local git settings
        is_cli_runtime = self.config.runtime == 'cli'
        if is_cli_runtime:
            logger.debug(
                "Skipping git configuration for CLI runtime - using user's local git config"
            )
            return

        # All runtimes (except CLI) use global git config
        cmd = f'git config --global user.name "{git_user_name}" && git config --global user.email "{git_user_email}"'

        # Execute git configuration command
        try:
            action = CmdRunAction(command=cmd)
            obs = self.run(action)
            if isinstance(obs, CmdOutputObservation) and obs.exit_code != 0:
                logger.warning(
                    f'Git config command failed: {cmd}, error: {obs.content}'
                )
            else:
                logger.info(
                    f'Successfully configured git: name={git_user_name}, email={git_user_email}'
                )
        except Exception as e:
            logger.warning(f'Failed to execute git config command: {cmd}, error: {e}')

    @abstractmethod
    def get_mcp_config(
        self, extra_stdio_servers: list[MCPStdioServerConfig] | None = None
    ) -> MCPConfig:
        pass

    # ====================================================================
    # Action execution
    # ====================================================================

    @abstractmethod
    def run(self, action: CmdRunAction) -> Observation:
        pass

    @abstractmethod
    def run_ipython(self, action: IPythonRunCellAction) -> Observation:
        pass

    @abstractmethod
    def read(self, action: FileReadAction) -> Observation:
        pass

    @abstractmethod
    def write(self, action: FileWriteAction) -> Observation:
        pass

    @abstractmethod
    def edit(self, action: FileEditAction) -> Observation:
        pass

    @abstractmethod
    def browse(self, action: BrowseURLAction) -> Observation:
        pass

    @abstractmethod
    def browse_interactive(self, action: BrowseInteractiveAction) -> Observation:
        pass

    @abstractmethod
    async def call_tool_mcp(self, action: MCPAction) -> Observation:
        pass

    # ====================================================================
    # File operations
    # ====================================================================

    @abstractmethod
    def copy_to(self, host_src: str, sandbox_dest: str, recursive: bool = False):
        raise NotImplementedError('This method is not implemented in the base class.')

    @abstractmethod
    def list_files(self, path: str | None = None) -> list[str]:
        """List files in the sandbox.

        If path is None, list files in the sandbox's initial working directory (e.g., /workspace).
        """
        raise NotImplementedError('This method is not implemented in the base class.')

    @abstractmethod
    def copy_from(self, path: str) -> Path:
        """Zip all files in the sandbox and return a path in the local filesystem."""
        raise NotImplementedError('This method is not implemented in the base class.')

    # ====================================================================
    # Authentication
    # ====================================================================

    @property
    def session_api_key(self) -> str | None:
        return None

    # ====================================================================
    # VSCode
    # ====================================================================

    @property
    def vscode_enabled(self) -> bool:
        return self._vscode_enabled

    @property
    def vscode_url(self) -> str | None:
        raise NotImplementedError('This method is not implemented in the base class.')

    @property
    def web_hosts(self) -> dict[str, int]:
        return {}

    # ====================================================================
    # Git
    # ====================================================================

    def _execute_shell_fn_git_handler(
        self, command: str, cwd: str | None
    ) -> CommandResult:
        """This function is used by the GitHandler to execute shell commands."""
        obs = self.run(
            CmdRunAction(command=command, is_static=True, hidden=True, cwd=cwd)
        )
        exit_code = 0
        content = ''

        if isinstance(obs, ErrorObservation):
            exit_code = -1

        if hasattr(obs, 'exit_code'):
            exit_code = obs.exit_code
        if hasattr(obs, 'content'):
            content = obs.content

        return CommandResult(content=content, exit_code=exit_code)

    def _create_file_fn_git_handler(self, path: str, content: str) -> int:
        """This function is used by the GitHandler to execute shell commands."""
        obs = self.write(FileWriteAction(path=path, content=content))
        if isinstance(obs, ErrorObservation):
            return -1
        return 0

    def get_git_changes(self, cwd: str) -> list[dict[str, str]] | None:
        self.git_handler.set_cwd(cwd)
        changes = self.git_handler.get_git_changes()
        return changes

    def get_git_diff(self, file_path: str, cwd: str) -> dict[str, str]:
        self.git_handler.set_cwd(cwd)
        return self.git_handler.get_git_diff(file_path)

    def get_workspace_branch(self, primary_repo_path: str | None = None) -> str | None:
        """
        Get the current branch of the workspace.

        Args:
            primary_repo_path: Path to the primary repository within the workspace.
                              If None, uses the workspace root.

        Returns:
            str | None: The current branch name, or None if not a git repository or error occurs.
        """
        if primary_repo_path:
            # Use the primary repository path
            git_cwd = str(self.workspace_root / primary_repo_path)
        else:
            # Use the workspace root
            git_cwd = str(self.workspace_root)

        self.git_handler.set_cwd(git_cwd)
        return self.git_handler.get_current_branch()

    @property
    def additional_agent_instructions(self) -> str:
        return ''

    def subscribe_to_shell_stream(
        self, callback: Callable[[str], None] | None = None
    ) -> bool:
        """Subscribe to shell command output stream.
        This method is meant to be overridden by runtime implementations
        that want to stream shell command output to external consumers.

        Args:
            callback: A function that will be called with each line of output from shell commands.
                     If None, any existing subscription will be removed.

        Returns False by default.
        """
        return False

    @classmethod
    def setup(cls, config: OpenHandsConfig, headless_mode: bool = False):
        """Set up the environment for runtimes to be created."""

    @classmethod
    def teardown(cls, config: OpenHandsConfig):
        """Tear down the environment in which runtimes are created."""
