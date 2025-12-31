"""This runtime runs commands locally using subprocess and performs file operations using Python's standard library.
It does not implement browser functionality.
"""

import asyncio
import os
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from binaryornot.check import is_binary
from openhands_aci.editor.editor import OHEditor
from openhands_aci.editor.exceptions import ToolError
from openhands_aci.editor.results import ToolResult
from openhands_aci.utils.diff import get_diff
from pydantic import SecretStr

from openhands.core.config import OpenHandsConfig
from openhands.core.config.mcp_config import MCPConfig, MCPStdioServerConfig
from openhands.core.logger import openhands_logger as logger
from openhands.events import EventStream
from openhands.events.action import (
    BrowseInteractiveAction,
    BrowseURLAction,
    CmdRunAction,
    FileEditAction,
    FileReadAction,
    FileWriteAction,
    IPythonRunCellAction,
)
from openhands.events.action.mcp import MCPAction
from openhands.events.event import FileEditSource, FileReadSource
from openhands.events.observation import (
    CmdOutputObservation,
    ErrorObservation,
    FileEditObservation,
    FileReadObservation,
    FileWriteObservation,
    Observation,
)
from openhands.integrations.provider import PROVIDER_TOKEN_TYPE
from openhands.llm.llm_registry import LLMRegistry
from openhands.runtime.base import Runtime
from openhands.runtime.plugins import PluginRequirement
from openhands.runtime.runtime_status import RuntimeStatus

if TYPE_CHECKING:
    from openhands.runtime.utils.windows_bash import WindowsPowershellSession

# Import Windows PowerShell support if on Windows
if sys.platform == 'win32':
    try:
        from openhands.runtime.utils.windows_exceptions import DotNetMissingError
        from openhands.runtime.utils.windows_bash import WindowsPowershellSession  # isort: skip
    except (ImportError, DotNetMissingError) as err:
        # Print a user-friendly error message without stack trace
        friendly_message = """
ERROR: PowerShell and .NET SDK are required but not properly configured

The .NET SDK and PowerShell are required for OpenHands CLI on Windows.
PowerShell integration cannot function without .NET Core.

Please install the .NET SDK by following the instructions at:
https://docs.all-hands.dev/usage/windows-without-wsl

After installing .NET SDK, restart your terminal and try again.
"""
        print(friendly_message, file=sys.stderr)
        logger.error(
            f'Windows runtime initialization failed: {type(err).__name__}: {str(err)}'
        )
        if (
            isinstance(err, DotNetMissingError)
            and hasattr(err, 'details')
            and err.details
        ):
            logger.debug(f'Details: {err.details}')

        # Exit the program with an error code
        sys.exit(1)


class CLIRuntime(Runtime):
    """A runtime implementation that runs commands locally using subprocess and performs
    file operations using Python's standard library. It does not implement browser functionality.

    Args:
        config (OpenHandsConfig): The application configuration.
        event_stream (EventStream): The event stream to subscribe to.
        sid (str, optional): The session ID. Defaults to 'default'.
        plugins (list[PluginRequirement] | None, optional): List of plugin requirements. Defaults to None.
        env_vars (dict[str, str] | None, optional): Environment variables to set. Defaults to None.
        status_callback (Callable | None, optional): Callback for status updates. Defaults to None.
        attach_to_existing (bool, optional): Whether to attach to an existing session. Defaults to False.
        headless_mode (bool, optional): Whether to run in headless mode. Defaults to False.
        user_id (str | None, optional): User ID for authentication. Defaults to None.
        git_provider_tokens (PROVIDER_TOKEN_TYPE | None, optional): Git provider tokens. Defaults to None.
    """

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
        super().__init__(
            config,
            event_stream,
            llm_registry,
            sid,
            plugins,
            env_vars,
            status_callback,
            attach_to_existing,
            headless_mode,
            user_id,
            git_provider_tokens,
        )

        # Set up workspace
        if self.config.workspace_base is not None:
            logger.warning(
                f'Workspace base path is set to {self.config.workspace_base}. '
                'It will be used as the path for the agent to run in. '
                'Be careful, the agent can EDIT files in this directory!'
            )
            self._workspace_path = self.config.workspace_base
        else:
            # Create a temporary directory for the workspace
            self._workspace_path = tempfile.mkdtemp(
                prefix=f'openhands_workspace_{sid}_'
            )
            logger.info(f'Created temporary workspace at {self._workspace_path}')

        # Runtime tests rely on this being set correctly.
        self.config.workspace_mount_path_in_sandbox = self._workspace_path

        # Initialize runtime state
        self._runtime_initialized = False
        self.file_editor = OHEditor(workspace_root=self._workspace_path)
        self._shell_stream_callback: Callable[[str], None] | None = None

        # Initialize PowerShell session on Windows
        self._is_windows = sys.platform == 'win32'
        self._powershell_session: WindowsPowershellSession | None = None

        logger.warning(
            'Initializing CLIRuntime. WARNING: NO SANDBOX IS USED. '
            'This runtime executes commands directly on the local system. '
            'Use with caution in untrusted environments.'
        )

    async def connect(self) -> None:
        """Initialize the runtime connection."""
        self.set_runtime_status(RuntimeStatus.STARTING_RUNTIME)

        # Ensure workspace directory exists
        os.makedirs(self._workspace_path, exist_ok=True)

        # Change to the workspace directory
        os.chdir(self._workspace_path)

        # Initialize PowerShell session if on Windows
        if self._is_windows:
            self._powershell_session = WindowsPowershellSession(
                work_dir=self._workspace_path,
                username=None,  # Use current user
                no_change_timeout_seconds=30,
                max_memory_mb=None,
            )

        if not self.attach_to_existing:
            await asyncio.to_thread(self.setup_initial_env)

        self._runtime_initialized = True
        self.set_runtime_status(RuntimeStatus.RUNTIME_STARTED)
        logger.info(f'CLIRuntime initialized with workspace at {self._workspace_path}')

    def add_env_vars(self, env_vars: dict[str, Any]) -> None:
        """Adds environment variables to the current runtime environment.
        For CLIRuntime, this means updating os.environ for the current process,
        so that subsequent commands inherit these variables.
        This overrides the BaseRuntime behavior which tries to run shell commands
        before it's initialized and modify .bashrc, which is not ideal for local CLI.
        """
        if not env_vars:
            return

        # We log only keys to avoid leaking sensitive values like tokens into logs.
        logger.info(
            f'[CLIRuntime] Setting environment variables for this session: {list(env_vars.keys())}'
        )

        for key, value in env_vars.items():
            if isinstance(value, SecretStr):
                os.environ[key] = value.get_secret_value()
                logger.warning(f'[CLIRuntime] Set os.environ["{key}"] (from SecretStr)')
            else:
                os.environ[key] = value
                logger.debug(f'[CLIRuntime] Set os.environ["{key}"]')

        # We don't use self.run() here because this method is called
        # during initialization before self._runtime_initialized is True.

    def _safe_terminate_process(self, process_obj, signal_to_send=signal.SIGTERM):
        """Safely attempts to terminate/kill a process group or a single process.

        Args:
            process_obj: the subprocess.Popen object started with start_new_session=True
            signal_to_send: the signal to send to the process group or process.
        """
        pid = getattr(process_obj, 'pid', None)
        if pid is None:
            return

        group_desc = (
            'kill process group'
            if signal_to_send == signal.SIGKILL
            else 'terminate process group'
        )
        process_desc = (
            'kill process' if signal_to_send == signal.SIGKILL else 'terminate process'
        )

        try:
            # Try to terminate/kill the entire process group
            logger.debug(f'[_safe_terminate_process] Original PID to act on: {pid}')
            pgid_to_kill = os.getpgid(
                pid
            )  # This might raise ProcessLookupError if pid is already gone
            logger.debug(
                f'[_safe_terminate_process] Attempting to {group_desc} for PID {pid} (PGID: {pgid_to_kill}) with {signal_to_send}.'
            )
            os.killpg(pgid_to_kill, signal_to_send)
            logger.debug(
                f'[_safe_terminate_process] Successfully sent signal {signal_to_send} to PGID {pgid_to_kill} (original PID: {pid}).'
            )
        except ProcessLookupError as e_pgid:
            logger.warning(
                f'[_safe_terminate_process] ProcessLookupError getting PGID for PID {pid} (it might have already exited): {e_pgid}. Falling back to direct kill/terminate.'
            )
            try:
                if signal_to_send == signal.SIGKILL:
                    process_obj.kill()
                else:
                    process_obj.terminate()
                logger.debug(
                    f'[_safe_terminate_process] Fallback: Terminated {process_desc} (PID: {pid}).'
                )
            except Exception as e_fallback:
                logger.error(
                    f'[_safe_terminate_process] Fallback: Error during {process_desc} (PID: {pid}): {e_fallback}'
                )
        except (AttributeError, OSError) as e_os:
            logger.error(
                f'[_safe_terminate_process] OSError/AttributeError during {group_desc} for PID {pid}: {e_os}. Falling back.'
            )
            # Fallback: try to terminate/kill the main process directly.
            try:
                if signal_to_send == signal.SIGKILL:
                    process_obj.kill()
                else:
                    process_obj.terminate()
                logger.debug(
                    f'[_safe_terminate_process] Fallback: Terminated {process_desc} (PID: {pid}).'
                )
            except Exception as e_fallback:
                logger.error(
                    f'[_safe_terminate_process] Fallback: Error during {process_desc} (PID: {pid}): {e_fallback}'
                )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.error(f'Error: {e}')

    def _execute_powershell_command(
        self, command: str, timeout: float
    ) -> CmdOutputObservation | ErrorObservation:
        """Execute a command using PowerShell session on Windows.

        Args:
            command: The command to execute
            timeout: Timeout in seconds for the command
        Returns:
            CmdOutputObservation containing the complete output and exit code
        """
        if self._powershell_session is None:
            return ErrorObservation(
                content='PowerShell session is not available.',
                error_id='POWERSHELL_SESSION_ERROR',
            )

        try:
            # Create a CmdRunAction for the PowerShell session
            from openhands.events.action import CmdRunAction

            ps_action = CmdRunAction(command=command)
            ps_action.set_hard_timeout(timeout)

            # Execute the command using the PowerShell session
            return self._powershell_session.execute(ps_action)

        except Exception as e:
            logger.error(f'Error executing PowerShell command "{command}": {e}')
            return ErrorObservation(
                content=f'Error executing PowerShell command "{command}": {str(e)}',
                error_id='POWERSHELL_EXECUTION_ERROR',
            )

    def _execute_shell_command(
        self, command: str, timeout: float
    ) -> CmdOutputObservation:
        """Execute a shell command and stream its output to a callback function.

        Args:
            command: The shell command to execute
            timeout: Timeout in seconds for the command
        Returns:
            CmdOutputObservation containing the complete output and exit code
        """
        output_lines = []
        timed_out = False
        start_time = time.monotonic()

        # Use shell=True to run complex bash commands
        process = subprocess.Popen(
            ['bash', '-c', command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Explicitly line-buffered for text mode
            universal_newlines=True,
            start_new_session=True,
        )
        logger.debug(
            f'[_execute_shell_command] PID of bash -c: {process.pid} for command: "{command}"'
        )

        exit_code = None

        try:
            if process.stdout:
                while process.poll() is None:
                    if (
                        timeout is not None
                        and (time.monotonic() - start_time) > timeout
                    ):
                        logger.debug(
                            f'Command "{command}" timed out after {timeout:.1f} seconds. Terminating.'
                        )
                        # Attempt to terminate the process group (SIGTERM)
                        self._safe_terminate_process(
                            process, signal_to_send=signal.SIGTERM
                        )
                        timed_out = True
                        break

                    ready_to_read, _, _ = select.select([process.stdout], [], [], 0.1)

                    if ready_to_read:
                        line = process.stdout.readline()
                        if line:
                            output_lines.append(line)
                            if self._shell_stream_callback:
                                self._shell_stream_callback(line)

            # Attempt to read any remaining data from stdout
            if process.stdout and not process.stdout.closed:
                try:
                    while line:
                        line = process.stdout.readline()
                        if line:
                            output_lines.append(line)
                            if self._shell_stream_callback:
                                self._shell_stream_callback(line)
                except Exception as e:
                    logger.warning(
                        f'Error reading directly from stdout after loop for "{command}": {e}'
                    )

            exit_code = process.returncode

            # If timeout occurred, ensure exit_code reflects this for the observation.
            if timed_out:
                exit_code = -1

        except Exception as e:
            logger.error(
                f'Outer exception in _execute_shell_command for "{command}": {e}'
            )
            if process and process.poll() is None:
                self._safe_terminate_process(process, signal_to_send=signal.SIGKILL)
            return CmdOutputObservation(
                command=command,
                content=''.join(output_lines) + f'\nError during execution: {e}',
                exit_code=-1,
            )

        complete_output = ''.join(output_lines)
        logger.debug(
            f'[_execute_shell_command] Complete output for "{command}" (len: {len(complete_output)}): {complete_output!r}'
        )
        obs_metadata = {'working_dir': self._workspace_path}
        if timed_out:
            obs_metadata['suffix'] = (
                f'[The command timed out after {timeout:.1f} seconds.]'
            )
            # exit_code = -1 # This is already set if timed_out is True

        return CmdOutputObservation(
            command=command,
            content=complete_output,
            exit_code=exit_code,
            metadata=obs_metadata,
        )

    def run(self, action: CmdRunAction) -> Observation:
        """Run a command using subprocess."""
        if not self._runtime_initialized:
            return ErrorObservation(
                f'Runtime not initialized for command: {action.command}'
            )

        if action.is_input:
            logger.warning(
                f"CLIRuntime received an action with `is_input=True` (command: '{action.command}'). "
                'CLIRuntime currently does not support sending input or signals to active processes. '
                'This action will be ignored and an error observation will be returned.'
            )
            return ErrorObservation(
                content=f"CLIRuntime does not support interactive input from the agent (e.g., 'C-c'). The command '{action.command}' was not sent to any process.",
                error_id='AGENT_ERROR$BAD_ACTION',
            )

        try:
            effective_timeout = (
                action.timeout
                if action.timeout is not None
                else self.config.sandbox.timeout
            )

            logger.debug(
                f'Running command in CLIRuntime: "{action.command}" with effective timeout: {effective_timeout}s'
            )

            # Use PowerShell on Windows if available, otherwise use subprocess
            if self._is_windows and self._powershell_session is not None:
                return self._execute_powershell_command(
                    action.command, timeout=effective_timeout
                )
            else:
                return self._execute_shell_command(
                    action.command, timeout=effective_timeout
                )
        except Exception as e:
            logger.error(
                f'Error in CLIRuntime.run for command "{action.command}": {str(e)}'
            )
            return ErrorObservation(
                f'Error running command "{action.command}": {str(e)}'
            )

    def run_ipython(self, action: IPythonRunCellAction) -> Observation:
        """Run a Python code cell.
        This functionality is not implemented in CLIRuntime.
        Users should also disable the Jupyter plugin in AgentConfig.
        """
        # This functionality is not implemented in CLIRuntime.
        # If you need to run IPython/Jupyter cells, please consider using a different runtime
        # or ensure the Jupyter plugin is disabled in your AgentConfig to avoid
        # attempting to use this disabled feature.
        logger.warning(
            "run_ipython is called on CLIRuntime, but it's not implemented. "
            'Please disable the Jupyter plugin in AgentConfig.'
        )
        return ErrorObservation(
            'Executing IPython cells is not implemented in CLIRuntime. '
        )

    def _sanitize_filename(self, filename: str) -> str:
        # if path is absolute, ensure it starts with _workspace_path
        if filename == '/workspace':
            actual_filename = self._workspace_path
        elif filename.startswith('/workspace/'):
            # Map /workspace/ to the actual workspace path
            # Note: /workspace is widely used, so we map it to allow using it with CLIRuntime
            actual_filename = os.path.join(
                self._workspace_path, filename[len('/workspace/') :]
            )
        elif filename.startswith('/'):
            if not filename.startswith(self._workspace_path):
                raise PermissionError(
                    f'Invalid path: {filename}. You can only work with files in {self._workspace_path}.'
                )
            actual_filename = filename
        else:
            actual_filename = os.path.join(self._workspace_path, filename.lstrip('/'))

        # Resolve the path to handle any '..' or '.' components
        resolved_path = os.path.realpath(actual_filename)

        # Check if the resolved path is still within the workspace
        if not resolved_path.startswith(self._workspace_path):
            raise PermissionError(
                f'Invalid path traversal: {filename}. Path resolves outside the workspace. Resolved: {resolved_path}, Workspace: {self._workspace_path}'
            )

        return resolved_path

    def read(self, action: FileReadAction) -> Observation:
        """Read a file using Python's standard library or OHEditor."""
        if not self._runtime_initialized:
            return ErrorObservation('Runtime not initialized')

        file_path = self._sanitize_filename(action.path)

        # Use OHEditor for OH_ACI implementation source
        if action.impl_source == FileReadSource.OH_ACI:
            result_str, _ = self._execute_file_editor(
                command='view',
                path=file_path,
                view_range=action.view_range,
            )

            return FileReadObservation(
                content=result_str,
                path=action.path,
                impl_source=FileReadSource.OH_ACI,
            )

        try:
            # Check if the file exists
            if not os.path.exists(file_path):
                return ErrorObservation(f'File not found: {action.path}')

            # Check if it's a directory
            if os.path.isdir(file_path):
                return ErrorObservation(f'Cannot read directory: {action.path}')

            # Cannot read binary files
            if is_binary(file_path):
                return ErrorObservation('ERROR_BINARY_FILE')

            # Read the file
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

            return FileReadObservation(content=content, path=action.path)
        except Exception as e:
            logger.error(f'Error reading file: {str(e)}')
            return ErrorObservation(f'Error reading file {action.path}: {str(e)}')

    def write(self, action: FileWriteAction) -> Observation:
        """Write to a file using Python's standard library."""
        if not self._runtime_initialized:
            return ErrorObservation('Runtime not initialized')

        file_path = self._sanitize_filename(action.path)

        try:
            # Create parent directories if they don't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Write to the file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(action.content)

            return FileWriteObservation(content='', path=action.path)
        except Exception as e:
            logger.error(f'Error writing to file: {str(e)}')
            return ErrorObservation(f'Error writing to file {action.path}: {str(e)}')

    def browse(self, action: BrowseURLAction) -> Observation:
        """Not implemented for CLI runtime."""
        return ErrorObservation(
            'Browser functionality is not implemented in CLIRuntime'
        )

    def browse_interactive(self, action: BrowseInteractiveAction) -> Observation:
        """Not implemented for CLI runtime."""
        return ErrorObservation(
            'Browser functionality is not implemented in CLIRuntime'
        )

    def _execute_file_editor(
        self,
        command: str,
        path: str,
        file_text: str | None = None,
        view_range: list[int] | None = None,
        old_str: str | None = None,
        new_str: str | None = None,
        insert_line: int | None = None,
        enable_linting: bool = False,
    ) -> tuple[str, tuple[str | None, str | None]]:
        """Execute file editor command and handle exceptions.

        Args:
            command: Editor command to execute
            path: File path
            file_text: Optional file text content
            view_range: Optional view range tuple (start, end)
            old_str: Optional string to replace
            new_str: Optional replacement string
            insert_line: Optional line number for insertion
            enable_linting: Whether to enable linting

        Returns:
            tuple: A tuple containing the output string and a tuple of old and new file content
        """
        result: ToolResult | None = None
        try:
            result = self.file_editor(
                command=command,
                path=path,
                file_text=file_text,
                view_range=view_range,
                old_str=old_str,
                new_str=new_str,
                insert_line=insert_line,
                enable_linting=enable_linting,
            )
        except ToolError as e:
            result = ToolResult(error=e.message)

        if result.error:
            return f'ERROR:\n{result.error}', (None, None)

        if not result.output:
            logger.warning(f'No output from file_editor for {path}')
            return '', (None, None)

        return result.output, (result.old_content, result.new_content)

    def edit(self, action: FileEditAction) -> Observation:
        """Edit a file using the OHEditor."""
        if not self._runtime_initialized:
            return ErrorObservation('Runtime not initialized')

        # Ensure the path is within the workspace
        file_path = self._sanitize_filename(action.path)

        # Check if it's a binary file
        if os.path.exists(file_path) and is_binary(file_path):
            return ErrorObservation('ERROR_BINARY_FILE')

        assert action.impl_source == FileEditSource.OH_ACI

        result_str, (old_content, new_content) = self._execute_file_editor(
            command=action.command,
            path=file_path,
            file_text=action.file_text,
            old_str=action.old_str,
            new_str=action.new_str,
            insert_line=action.insert_line,
            enable_linting=False,
        )

        return FileEditObservation(
            content=result_str,
            path=action.path,
            old_content=action.old_str,
            new_content=action.new_str,
            impl_source=FileEditSource.OH_ACI,
            diff=get_diff(
                old_contents=old_content or '',
                new_contents=new_content or '',
                filepath=action.path,
            ),
        )

    async def call_tool_mcp(self, action: MCPAction) -> Observation:
        """Execute an MCP tool action in CLI runtime.

        Args:
            action: The MCP action to execute

        Returns:
            Observation: The result of the MCP tool execution
        """
        # Check if we're on Windows - MCP is disabled on Windows
        if sys.platform == 'win32':
            self.log('info', 'MCP functionality is disabled on Windows')
            return ErrorObservation('MCP functionality is not available on Windows')

        # Import here to avoid circular imports
        from openhands.mcp.utils import call_tool_mcp as call_tool_mcp_handler
        from openhands.mcp.utils import create_mcp_clients

        try:
            # Get the MCP config for this runtime
            mcp_config = self.get_mcp_config()

            if (
                not mcp_config.sse_servers
                and not mcp_config.shttp_servers
                and not mcp_config.stdio_servers
            ):
                self.log('warning', 'No MCP servers configured')
                return ErrorObservation('No MCP servers configured')

            self.log(
                'debug',
                f'Creating MCP clients for action {action.name} with servers: '
                f'SSE={len(mcp_config.sse_servers)}, SHTTP={len(mcp_config.shttp_servers)}, '
                f'stdio={len(mcp_config.stdio_servers)}',
            )

            # Create clients for this specific operation
            mcp_clients = await create_mcp_clients(
                mcp_config.sse_servers,
                mcp_config.shttp_servers,
                self.sid,
                mcp_config.stdio_servers,
            )

            if not mcp_clients:
                self.log('warning', 'No MCP clients could be created')
                return ErrorObservation(
                    'No MCP clients could be created - check server configurations'
                )

            # Call the tool and return the result
            self.log(
                'debug',
                f'Executing MCP tool: {action.name} with arguments: {action.arguments}',
            )
            result = await call_tool_mcp_handler(mcp_clients, action)
            self.log('debug', f'MCP tool {action.name} executed successfully')
            return result

        except Exception as e:
            error_msg = f'Error executing MCP tool {action.name}: {str(e)}'
            self.log('error', error_msg)
            return ErrorObservation(error_msg)

    @property
    def workspace_root(self) -> Path:
        """Return the workspace root path."""
        return Path(os.path.abspath(self._workspace_path))

    def copy_to(self, host_src: str, sandbox_dest: str, recursive: bool = False):
        """Copy a file or directory from the host to the sandbox."""
        if not self._runtime_initialized:
            raise RuntimeError('Runtime not initialized')
        if not os.path.exists(host_src):  # Source must exist on host
            raise FileNotFoundError(f"Source path '{host_src}' does not exist.")

        dest = self._sanitize_filename(sandbox_dest)

        try:
            # Case 1: Source is a directory and recursive copy.
            if os.path.isdir(host_src) and recursive:
                # Target is dest / basename(host_src)
                final_target_dir = os.path.join(dest, os.path.basename(host_src))

                # If source and final target are the same, skip.
                if os.path.realpath(host_src) == os.path.realpath(final_target_dir):
                    logger.debug(
                        'Skipping recursive copy: source and target are identical.'
                    )
                    pass
                else:
                    # Ensure parent of final_target_dir exists.
                    os.makedirs(dest, exist_ok=True)
                    shutil.copytree(host_src, final_target_dir, dirs_exist_ok=True)
                    # Why: Copies dir host_src into dest. Merges if target exists.

            # Case 2: Source is a file.
            elif os.path.isfile(host_src):
                final_target_file_path: str
                # Scenario A: sandbox_dest is clearly a directory.
                if os.path.isdir(dest) or (sandbox_dest.endswith(('/', os.sep))):
                    target_dir = dest
                    os.makedirs(target_dir, exist_ok=True)
                    final_target_file_path = os.path.join(
                        target_dir, os.path.basename(host_src)
                    )
                    # Why: Copies file into specified directory.

                # Scenario B: sandbox_dest is likely a new directory (e.g., 'new_dir').
                elif not os.path.exists(dest) and '.' not in os.path.basename(dest):
                    target_dir = dest
                    os.makedirs(target_dir, exist_ok=True)
                    final_target_file_path = os.path.join(
                        target_dir, os.path.basename(host_src)
                    )
                    # Why: Creates 'new_dir' and copies file into it.

                # Scenario C: sandbox_dest is a full file path.
                else:
                    final_target_file_path = dest
                    os.makedirs(os.path.dirname(final_target_file_path), exist_ok=True)
                    # Why: Copies file to a specific path, possibly renaming.

                shutil.copy2(host_src, final_target_file_path)

            else:  # Source is not a valid file or directory.
                raise FileNotFoundError(
                    f"Source path '{host_src}' is not a valid file or directory."
                )

        except FileNotFoundError as e:
            logger.error(f'File not found during copy: {str(e)}')
            raise
        except shutil.SameFileError as e:
            # We can be lenient here, just ignore this error.
            logger.debug(
                f'Skipping copy as source and destination are the same: {str(e)}'
            )
            pass
        except Exception as e:
            logger.error(f'Unexpected error copying file: {str(e)}')
            raise RuntimeError(f'Unexpected error copying file: {str(e)}')

    def list_files(self, path: str | None = None) -> list[str]:
        """List files in the sandbox."""
        if not self._runtime_initialized:
            raise RuntimeError('Runtime not initialized')

        if path is None:
            dir_path = self._workspace_path
        else:
            dir_path = self._sanitize_filename(path)

        try:
            if not os.path.exists(dir_path):
                return []

            if not os.path.isdir(dir_path):
                return [dir_path]

            # List files in the directory
            return [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
        except Exception as e:
            logger.error(f'Error listing files: {str(e)}')
            return []

    def copy_from(self, path: str) -> Path:
        """Zip all files in the sandbox and return a path in the local filesystem."""
        if not self._runtime_initialized:
            raise RuntimeError('Runtime not initialized')

        source_path = self._sanitize_filename(path)

        if not os.path.exists(source_path):
            raise FileNotFoundError(f'Path not found: {path}')

        # Create a temporary zip file
        temp_zip = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
        temp_zip.close()

        try:
            with zipfile.ZipFile(temp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                if os.path.isdir(source_path):
                    # Add all files in the directory
                    for root, _, files in os.walk(source_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, source_path)
                            zipf.write(file_path, arcname)
                else:
                    # Add a single file
                    zipf.write(source_path, os.path.basename(source_path))

            return Path(temp_zip.name)
        except Exception as e:
            logger.error(f'Error creating zip file: {str(e)}')
            raise RuntimeError(f'Error creating zip file: {str(e)}')

    def close(self) -> None:
        # Clean up PowerShell session if it exists
        if self._powershell_session is not None:
            try:
                self._powershell_session.close()
                logger.debug('PowerShell session closed successfully.')
            except Exception as e:
                logger.warning(f'Error closing PowerShell session: {e}')
            finally:
                self._powershell_session = None

        self._runtime_initialized = False
        super().close()

    @classmethod
    async def delete(cls, conversation_id: str) -> None:
        """Delete any resources associated with a conversation."""
        # Look for temporary directories that might be associated with this conversation
        temp_dir = tempfile.gettempdir()
        prefix = f'openhands_workspace_{conversation_id}_'

        for item in os.listdir(temp_dir):
            if item.startswith(prefix):
                try:
                    path = os.path.join(temp_dir, item)
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                        logger.info(f'Deleted workspace directory: {path}')
                except Exception as e:
                    logger.error(f'Error deleting workspace directory: {str(e)}')

    @property
    def additional_agent_instructions(self) -> str:
        return '\n\n'.join(
            [
                f'Your working directory is {self._workspace_path}. You can only read and write files in this directory.',
                "You are working directly on the user's machine. In most cases, the working environment is already set up.",
            ]
        )

    def get_mcp_config(
        self, extra_stdio_servers: list[MCPStdioServerConfig] | None = None
    ) -> MCPConfig:
        """Get MCP configuration for CLI runtime.

        Args:
            extra_stdio_servers: Additional stdio servers to include in the config

        Returns:
            MCPConfig: The MCP configuration with stdio servers and any configured SSE/SHTTP servers
        """
        # Check if we're on Windows - MCP is disabled on Windows
        if sys.platform == 'win32':
            self.log('debug', 'MCP is disabled on Windows, returning empty config')
            return MCPConfig(sse_servers=[], stdio_servers=[], shttp_servers=[])

        # Note: we update the self.config.mcp directly for CLI runtime, which is different from other runtimes.
        mcp_config = self.config.mcp

        # Add any extra stdio servers
        if extra_stdio_servers:
            current_stdio_servers = list(mcp_config.stdio_servers)
            for extra_server in extra_stdio_servers:
                # Check if this stdio server is already in the config
                if extra_server not in current_stdio_servers:
                    current_stdio_servers.append(extra_server)
                    self.log('info', f'Added extra stdio server: {extra_server.name}')
            mcp_config.stdio_servers = current_stdio_servers

        self.log(
            'debug',
            f'CLI MCP config: {len(mcp_config.sse_servers)} SSE servers, '
            f'{len(mcp_config.stdio_servers)} stdio servers, '
            f'{len(mcp_config.shttp_servers)} SHTTP servers',
        )

        return mcp_config

    def subscribe_to_shell_stream(
        self, callback: Callable[[str], None] | None = None
    ) -> bool:
        """Subscribe to shell command output stream.

        Args:
            callback: A function that will be called with each line of output from shell commands.
                     If None, any existing subscription will be removed.
        """
        self._shell_stream_callback = callback
        return True
