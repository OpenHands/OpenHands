"""This module provides a Windows-specific implementation for running commands
in a PowerShell session using the pythonnet library to interact with the .NET
PowerShell SDK directly. This aims to provide a more robust and integrated
way to manage PowerShell processes compared to using temporary script files.
"""

import os
import re
import subprocess
import time
from pathlib import Path
from threading import RLock

import pythonnet

from openhands.core.logger import openhands_logger as logger
from openhands.events.action import CmdRunAction
from openhands.events.observation import ErrorObservation
from openhands.events.observation.commands import (
    CmdOutputMetadata,
    CmdOutputObservation,
)
from openhands.runtime.utils.bash_constants import TIMEOUT_MESSAGE_TEMPLATE
from openhands.runtime.utils.windows_exceptions import DotNetMissingError
from openhands.utils.shutdown_listener import should_continue

try:
    pythonnet.load('coreclr')
    logger.info("Successfully called pythonnet.load('coreclr')")

    # Now that pythonnet is initialized, import clr and System
    try:
        import clr

        logger.debug(f'Imported clr module from: {clr.__file__}')
        # Load System assembly *after* pythonnet is initialized
        clr.AddReference('System')
        import System
    except Exception as clr_sys_ex:
        error_msg = 'Failed to import .NET components.'
        details = str(clr_sys_ex)
        logger.error(f'{error_msg} Details: {details}')
        raise DotNetMissingError(error_msg, details)
except Exception as coreclr_ex:
    error_msg = 'Failed to load CoreCLR.'
    details = str(coreclr_ex)
    logger.error(f'{error_msg} Details: {details}')
    raise DotNetMissingError(error_msg, details)


def find_latest_pwsh_sdk_path(
    executable_name='pwsh.exe',
    dll_name='System.Management.Automation.dll',
    min_version=(7, 0, 0),
    env_var='PWSH_DIR',
):
    """
    Checks PWSH_DIR environment variable first to find pwsh and DLL.
    If not found or not suitable, scans all pwsh executables in PATH, runs --version to find latest >= min_version.
    Returns full DLL path if found, else None.
    """

    def parse_version(output):
        # Extract semantic version from pwsh --version output
        match = re.search(r'(\d+)\.(\d+)\.(\d+)', output)
        if match:
            return tuple(map(int, match.groups()))
        return None

    # Try environment variable override first
    pwsh_dir = os.environ.get(env_var)
    if pwsh_dir:
        pwsh_path = Path(pwsh_dir) / executable_name
        dll_path = Path(pwsh_dir) / dll_name
        if pwsh_path.is_file() and dll_path.is_file():
            try:
                completed = subprocess.run(
                    [str(pwsh_path), '--version'],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if completed.returncode == 0:
                    ver = parse_version(completed.stdout)
                    if ver and ver >= min_version:
                        logger.info(f'Found pwsh from env variable "{env_var}"')
                        return str(dll_path)
            except Exception:
                pass

    # Adjust executable_name for Windows if needed
    if os.name == 'nt' and not executable_name.lower().endswith('.exe'):
        executable_name += '.exe'

    # Search PATH for all pwsh executables
    paths = os.environ.get('PATH', '').split(os.pathsep)
    candidates = []
    for p in paths:
        exe_path = Path(p) / executable_name
        if exe_path.is_file() and os.access(str(exe_path), os.X_OK):
            try:
                completed = subprocess.run(
                    [str(exe_path), '--version'],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if completed.returncode == 0:
                    ver = parse_version(completed.stdout)
                    if ver:
                        candidates.append((ver, exe_path.resolve()))
            except Exception:
                pass

    # Sort candidates by version descending
    candidates.sort(key=lambda x: x[0], reverse=True)

    for ver, exe_path in candidates:
        if ver >= min_version:
            dll_path = exe_path.parent / dll_name
            if dll_path.is_file():
                return str(dll_path)

    return None


# Attempt to load the PowerShell SDK assembly only if clr and System loaded
ps_sdk_path = None
try:
    # Attempt primary detection via helper function
    ps_sdk_path = find_latest_pwsh_sdk_path()
    if ps_sdk_path:
        clr.AddReference(ps_sdk_path)
        logger.info(f'Loaded PowerShell SDK dynamically detected: {ps_sdk_path}')
    else:
        pwsh7_path = (
            Path(os.environ.get('ProgramFiles', 'C:\\Program Files'))
            / 'PowerShell'
            / '7'
            / 'System.Management.Automation.dll'
        )
        if pwsh7_path.exists():
            ps_sdk_path = str(pwsh7_path)
            clr.AddReference(ps_sdk_path)
            logger.info(f'Loaded PowerShell SDK (Core): {ps_sdk_path}')
        else:
            # Fallback to Windows PowerShell 5.1 bundled with Windows
            winps_path = (
                Path(os.environ.get('SystemRoot', 'C:\\Windows'))
                / 'System32'
                / 'WindowsPowerShell'
                / 'v1.0'
                / 'System.Management.Automation.dll'
            )
            if winps_path.exists():
                ps_sdk_path = str(winps_path)
                clr.AddReference(ps_sdk_path)
                logger.debug(f'Loaded PowerShell SDK (Desktop): {ps_sdk_path}')
            else:
                # Last resort: try loading by assembly name (might work if in GAC or path)
                clr.AddReference('System.Management.Automation')
                logger.info(
                    'Attempted to load PowerShell SDK by name (System.Management.Automation)'
                )

    from System.Management.Automation import JobState, PowerShell
    from System.Management.Automation.Language import Parser
    from System.Management.Automation.Runspaces import (
        RunspaceFactory,
        RunspaceState,
    )
except Exception as e:
    error_msg = 'Failed to load PowerShell SDK components.'
    details = f'{str(e)} (Path searched: {ps_sdk_path})'
    logger.error(f'{error_msg} Details: {details}')
    raise DotNetMissingError(error_msg, details)


class WindowsPowershellSession:
    """Manages a persistent PowerShell session using the .NET SDK via pythonnet.

    Allows executing commands within a single runspace, preserving state
    (variables, current directory) between calls.
    Handles basic timeout and captures output/error streams.
    """

    def __init__(
        self,
        work_dir: str,
        username: str | None = None,
        no_change_timeout_seconds: int = 30,
        max_memory_mb: int | None = None,
    ):
        """Initializes the PowerShell session.

        Args:
            work_dir: The starting working directory for the session.
            username: (Currently ignored) Username for execution. PowerShell SDK typically runs as the current user.
            no_change_timeout_seconds: Timeout in seconds if no output change is detected (currently NOT fully implemented).
            max_memory_mb: (Currently ignored) Maximum memory limit for the process.
        """
        # Initialize state flags early to prevent AttributeError in __del__ if init fails
        self._closed = False
        self._initialized = False
        self.runspace = None  # Initialize runspace to None

        if PowerShell is None:  # Check if SDK loading failed during module import
            # Logged critical error during import, just raise here to prevent instantiation
            error_msg = (
                'PowerShell SDK (System.Management.Automation.dll) could not be loaded.'
            )
            logger.error(error_msg)
            raise DotNetMissingError(error_msg)

        self.work_dir = os.path.abspath(work_dir)
        self.username = username
        self._cwd = self.work_dir
        self.NO_CHANGE_TIMEOUT_SECONDS = no_change_timeout_seconds
        self.max_memory_mb = max_memory_mb  # Stored, but not used yet.

        self.active_job = None
        self._job_lock = RLock()
        self._last_job_output = ''  # Stores cumulative output returned in the last observation for the active job
        self._last_job_error: list[
            str
        ] = []  # Stores cumulative errors returned in the last observation for the active job

        # Create and open the persistent runspace
        try:
            # Consider InitialSessionState for more control (e.g., execution policy)
            # iss = InitialSessionState.CreateDefault()
            # iss.ExecutionPolicy = Microsoft.PowerShell.ExecutionPolicy.Unrestricted # Requires importing Microsoft.PowerShell namespace
            # self.runspace = RunspaceFactory.CreateRunspace(iss)
            self.runspace = RunspaceFactory.CreateRunspace()
            self.runspace.Open()
            # Set initial working directory within the runspace
            self._set_initial_cwd()
            self._initialized = True  # Set to True only on successful initialization
            logger.info(f'PowerShell runspace created. Initial CWD set to: {self._cwd}')
        except Exception as e:
            logger.exception(f'Failed to create or open PowerShell runspace: {e}')
            self.close()  # Ensure cleanup if init fails partially
            raise RuntimeError(f'Failed to initialize PowerShell runspace: {e}')

    def _set_initial_cwd(self) -> None:
        """Sets the initial working directory in the runspace."""
        ps = None
        try:
            ps = PowerShell.Create()
            ps.Runspace = self.runspace
            ps.AddScript(f'Set-Location -Path "{self._cwd}"').Invoke()
            if ps.Streams.Error:
                errors = '\n'.join([str(err) for err in ps.Streams.Error])
                logger.warning(f"Error setting initial CWD to '{self._cwd}': {errors}")
                # Confirm actual CWD if setting failed
                self._confirm_cwd()
            else:
                logger.debug(f'Successfully set initial runspace CWD to {self._cwd}')
                # Optional: Confirm CWD even on success for robustness
                # self._confirm_cwd()
        except Exception as e:
            logger.exception(f'Exception setting initial CWD: {e}')
            # Attempt to confirm CWD even if setting threw an exception
            self._confirm_cwd()
        finally:
            if ps:
                ps.Dispose()

    def _confirm_cwd(self) -> None:
        """Confirms the actual CWD in the runspace and updates self._cwd."""
        ps_confirm = None
        try:
            ps_confirm = PowerShell.Create()
            ps_confirm.Runspace = self.runspace
            ps_confirm.AddScript('Get-Location')
            results = ps_confirm.Invoke()
            if results and results.Count > 0 and hasattr(results[0], 'Path'):
                actual_cwd = str(results[0].Path)
                if os.path.isdir(actual_cwd):
                    if actual_cwd != self._cwd:
                        logger.warning(
                            f'Runspace CWD ({actual_cwd}) differs from expected ({self._cwd}). Updating session CWD.'
                        )
                        self._cwd = actual_cwd
                    else:
                        logger.debug(f'Confirmed runspace CWD is {self._cwd}')
                else:
                    logger.error(
                        f'Get-Location returned an invalid path: {actual_cwd}. Session CWD may be inaccurate.'
                    )
            elif ps_confirm.Streams.Error:
                errors = '\n'.join([str(err) for err in ps_confirm.Streams.Error])
                logger.error(f'Error confirming runspace CWD: {errors}')
            else:
                logger.error('Could not confirm runspace CWD (No result or error).')
        except Exception as e:
            logger.error(f'Exception confirming CWD: {e}')
        finally:
            if ps_confirm:
                ps_confirm.Dispose()

    @property
    def cwd(self) -> str:
        """Gets the last known working directory of the session."""
        return self._cwd

    def _run_ps_command(
        self, script: str, log_output: bool = True
    ) -> list[System.Management.Automation.PSObject]:
        """Helper to run a simple synchronous command in the runspace."""
        if log_output:
            logger.debug(f"Running PS command: '{script}'")
        ps = None
        results = []
        try:
            ps = PowerShell.Create()
            ps.Runspace = self.runspace
            ps.AddScript(script)
            results = ps.Invoke()
        except Exception as e:
            logger.error(f'Exception running script: {script}\n{e}')
        finally:
            if ps:
                ps.Dispose()
        return results if results else []

    def _get_job_object(
        self, job_id: int | None
    ) -> System.Management.Automation.Job | None:
        """Retrieves a job object by its ID."""
        script = f'Get-Job -Id {job_id}'
        results = self._run_ps_command(script, log_output=False)
        if results and len(results) > 0:
            potential_job_wrapper = results[0]
            try:
                underlying_job = potential_job_wrapper.BaseObject
                # Basic check for job-like properties before returning
                _ = underlying_job.Id
                _ = underlying_job.JobStateInfo.State
                return underlying_job
            except AttributeError:
                logger.warning(f'Retrieved object is not a valid job. ID: {job_id}')
                return None
        return None

    def _receive_job_output(
        self, job: System.Management.Automation.Job, keep: bool = False
    ) -> tuple[str, list[str]]:
        """Receives output and errors from a job."""
        if not job:
            return '', []

        output_parts = []
        error_parts = []

        # Get error stream directly from job object if available
        try:
            current_job_obj = self._get_job_object(job.Id)
            if current_job_obj and current_job_obj.Error:
                error_records = current_job_obj.Error.ReadAll()
                if error_records:
                    error_parts.extend([str(e) for e in error_records])
        except Exception as read_err:
            logger.error(
                f'Failed to read job error stream directly for Job {job.Id}: {read_err}'
            )
            error_parts.append(f'[Direct Error Stream Read Exception: {read_err}]')

        # Run Receive-Job for the output stream
        keep_switch = '-Keep' if keep else ''
        script = f'Receive-Job -Job (Get-Job -Id {job.Id}) {keep_switch}'

        ps_receive = None
        try:
            ps_receive = PowerShell.Create()
            ps_receive.Runspace = self.runspace
            ps_receive.AddScript(script)

            # Collect output
            results = ps_receive.Invoke()
            if results:
                output_parts = [str(r) for r in results]

            # Collect errors from the Receive-Job command
            if ps_receive.Streams.Error:
                receive_job_errors = [str(e) for e in ps_receive.Streams.Error]
                logger.warning(
                    f'Errors during Receive-Job for Job ID {job.Id}: {receive_job_errors}'
                )
                error_parts.extend(receive_job_errors)

        except Exception as e:
            logger.error(f'Exception during Receive-Job for Job ID {job.Id}: {e}')
            error_parts.append(f'[Receive-Job Exception: {e}]')
        finally:
            if ps_receive:
                ps_receive.Dispose()

        final_combined_output = '\n'.join(output_parts)
        return final_combined_output, error_parts

    def _stop_active_job(self) -> CmdOutputObservation | ErrorObservation:
        """Stops the active job, collects final output, and cleans up."""
        with self._job_lock:
            job = self.active_job
            if not job:
                return ErrorObservation(
                    content='ERROR: No previous running command to interact with.'
                )

            job_id = job.Id  # type: ignore[unreachable]
            logger.info(f'Attempting to stop job ID: {job_id} via C-c.')

            # Attempt graceful stop
            stop_script = f'Stop-Job -Job (Get-Job -Id {job_id})'
            self._run_ps_command(stop_script)

            # Allow process time to potentially print shutdown messages
            time.sleep(0.5)

            # Get final output and errors
            final_output, final_errors = self._receive_job_output(job, keep=False)

            combined_output = final_output
            combined_errors = final_errors

            # Check job state after stopping
            final_job = self._get_job_object(job_id)
            final_state = final_job.JobStateInfo.State if final_job else JobState.Failed

            logger.info(f'Job {job_id} final state after stop attempt: {final_state}')

            # Clean up the job
            remove_script = f'Remove-Job -Job (Get-Job -Id {job_id})'
            self._run_ps_command(remove_script)

            # Clear the active job reference
            self.active_job = None

            # Construct result
            output_builder = [combined_output] if combined_output else []
            if combined_errors:
                output_builder.append('\n[ERROR STREAM]')
                output_builder.extend(combined_errors)

            # Determine exit code - 0 if Stopped/Completed, 1 otherwise
            exit_code = (
                0 if final_state in [JobState.Stopped, JobState.Completed] else 1
            )

            final_content = '\n'.join(output_builder).strip()

            current_cwd = self._cwd
            python_safe_cwd = current_cwd.replace('\\\\', '\\\\\\\\')
            metadata = CmdOutputMetadata(
                exit_code=exit_code, working_dir=python_safe_cwd
            )
            metadata.suffix = f'\n[The command completed with exit code {exit_code}. CTRL+C was sent.]'

            return CmdOutputObservation(
                content=final_content,
                command='C-c',
                metadata=metadata,
            )

    def _check_active_job(
        self, timeout_seconds: int
    ) -> CmdOutputObservation | ErrorObservation:
        """Checks the active job for new output and status, waiting up to timeout_seconds."""
        with self._job_lock:
            if not self.active_job:
                return ErrorObservation(
                    content='ERROR: No previous running command to retrieve logs from.'
                )

            job_id = self.active_job.Id  # type: ignore[unreachable]
            logger.info(
                f'Checking active job ID: {job_id} for new output (timeout={timeout_seconds}s).'
            )

            start_time = time.monotonic()
            monitoring_loop_finished = False
            accumulated_new_output_builder = []
            accumulated_new_errors = []
            exit_code = -1  # Assume running
            final_state = JobState.Running
            latest_cumulative_output = self._last_job_output
            latest_cumulative_errors = list(self._last_job_error)

            while not monitoring_loop_finished:
                if not should_continue():
                    logger.warning('Shutdown signal received during job check.')
                    monitoring_loop_finished = True
                    continue

                elapsed_seconds = time.monotonic() - start_time
                if elapsed_seconds > timeout_seconds:
                    logger.warning(f'Job check timed out after {timeout_seconds}s.')
                    monitoring_loop_finished = True
                    continue

                current_job_obj = self._get_job_object(job_id)
                if not current_job_obj:
                    logger.error(f'Job {job_id} object disappeared during check.')
                    accumulated_new_errors.append('[Job object lost during check]')
                    monitoring_loop_finished = True
                    exit_code = 1
                    final_state = JobState.Failed
                    if self.active_job and self.active_job.Id == job_id:
                        self.active_job = None
                    continue

                # Poll output with keep=True (returns cumulative output/errors)
                polled_cumulative_output, polled_cumulative_errors = (
                    self._receive_job_output(current_job_obj, keep=True)
                )

                # Detect new output since last poll
                new_output_detected = ''
                if polled_cumulative_output != latest_cumulative_output:
                    if polled_cumulative_output.startswith(latest_cumulative_output):
                        new_output_detected = polled_cumulative_output[
                            len(latest_cumulative_output) :
                        ]
                    else:
                        logger.warning(
                            f'Job {job_id} check: Cumulative output changed unexpectedly'
                        )
                        new_output_detected = polled_cumulative_output.removeprefix(
                            self._last_job_output
                        )

                    if new_output_detected.strip():
                        accumulated_new_output_builder.append(
                            new_output_detected.strip()
                        )

                # Detect new errors
                latest_cumulative_errors_set = set(latest_cumulative_errors)
                new_errors_detected = [
                    e
                    for e in polled_cumulative_errors
                    if e not in latest_cumulative_errors_set
                ]
                if new_errors_detected:
                    accumulated_new_errors.extend(new_errors_detected)

                latest_cumulative_output = polled_cumulative_output
                latest_cumulative_errors = polled_cumulative_errors

                # Check job state
                current_state = current_job_obj.JobStateInfo.State
                if current_state not in [JobState.Running, JobState.NotStarted]:
                    logger.info(
                        f'Job {job_id} finished check loop with state: {current_state}'
                    )
                    monitoring_loop_finished = True
                    final_state = current_state
                    continue

                time.sleep(0.1)  # Prevent busy-waiting

            # Process results after loop finished
            is_finished = final_state not in [JobState.Running, JobState.NotStarted]
            final_content = '\n'.join(accumulated_new_output_builder).strip()
            final_errors = list(accumulated_new_errors)

            if is_finished:
                logger.info(f'Job {job_id} has finished. Collecting final output.')
                final_job_obj = self._get_job_object(job_id)
                if final_job_obj:
                    # Final receive with keep=False to consume remaining output
                    final_cumulative_output, final_cumulative_errors = (
                        self._receive_job_output(final_job_obj, keep=False)
                    )

                    # Check for new output in final chunk
                    final_new_output_chunk = ''
                    if final_cumulative_output.startswith(latest_cumulative_output):
                        final_new_output_chunk = final_cumulative_output[
                            len(latest_cumulative_output) :
                        ]
                    elif final_cumulative_output:
                        final_new_output_chunk = final_cumulative_output.removeprefix(
                            self._last_job_output
                        )

                    if final_new_output_chunk.strip():
                        final_content = '\n'.join(
                            filter(
                                None, [final_content, final_new_output_chunk.strip()]
                            )
                        )

                    # Check for new errors in final chunk
                    latest_cumulative_errors_set = set(latest_cumulative_errors)
                    new_final_errors = [
                        e
                        for e in final_cumulative_errors
                        if e not in latest_cumulative_errors_set
                    ]
                    if new_final_errors:
                        final_errors.extend(new_final_errors)

                    # Determine exit code based on state
                    exit_code = 0 if final_state == JobState.Completed else 1

                    # Clean up job
                    remove_script = f'Remove-Job -Job (Get-Job -Id {job_id})'
                    self._run_ps_command(remove_script)
                    if self.active_job and self.active_job.Id == job_id:
                        self.active_job = None
                    self._last_job_output = ''
                    self._last_job_error = []
                else:
                    logger.warning(f'Could not get final job object {job_id}')
                    exit_code = 1
                    if self.active_job and self.active_job.Id == job_id:
                        self.active_job = None
                    self._last_job_output = ''
                    self._last_job_error = []
            else:
                # Update persistent state with latest cumulative values
                self._last_job_output = latest_cumulative_output
                self._last_job_error = list(set(latest_cumulative_errors))

            # Append errors to final content
            if final_errors:
                error_stream_text = '\n'.join(final_errors)
                if final_content:
                    final_content += f'\n[ERROR STREAM]\n{error_stream_text}'
                else:
                    final_content = f'[ERROR STREAM]\n{error_stream_text}'
                # Ensure exit code is non-zero if errors occurred
                if exit_code == 0 and final_state != JobState.Completed:
                    exit_code = 1

            current_cwd = self._cwd
            python_safe_cwd = current_cwd.replace('\\\\', '\\\\\\\\')
            metadata = CmdOutputMetadata(
                exit_code=exit_code, working_dir=python_safe_cwd
            )
            metadata.prefix = '[Below is the output of the previous command.]\n'

            if is_finished:
                metadata.suffix = (
                    f'\n[The command completed with exit code {exit_code}.]'
                )
            else:
                metadata.suffix = (
                    f'\n[The command timed out after {timeout_seconds} seconds. '
                    f'{TIMEOUT_MESSAGE_TEMPLATE}]'
                )

            return CmdOutputObservation(
                content=final_content,
                command='',
                metadata=metadata,
            )

    def _get_current_cwd(self) -> str:
        """Gets the current working directory from the runspace."""
        # Use helper to run Get-Location
        results = self._run_ps_command('Get-Location')

        # --- Add more detailed check logging ---
        if results and results.Count > 0:  # type: ignore[attr-defined]
            first_result = results[0]
            has_path_attr = hasattr(first_result, 'Path')

            if has_path_attr:
                # Original logic resumes here if hasattr is True
                fetched_cwd = str(first_result.Path)
                if os.path.isdir(fetched_cwd):
                    if fetched_cwd != self._cwd:
                        logger.info(
                            f"_get_current_cwd: Fetched CWD '{fetched_cwd}' differs from cached '{self._cwd}'. Updating cache."
                        )
                        self._cwd = fetched_cwd
                    return self._cwd
                else:
                    logger.warning(
                        f"_get_current_cwd: Path '{fetched_cwd}' is not a valid directory. Returning cached CWD: {self._cwd}"
                    )
                    return self._cwd
            else:
                # Handle cases where Path attribute is missing (e.g., unexpected object type)
                # Maybe the path is in BaseObject?
                try:
                    base_object = first_result.BaseObject
                    if hasattr(base_object, 'Path'):
                        fetched_cwd = str(base_object.Path)
                        if os.path.isdir(fetched_cwd):
                            if fetched_cwd != self._cwd:
                                logger.info(
                                    f"_get_current_cwd: Fetched CWD '{fetched_cwd}' (from BaseObject) differs from cached '{self._cwd}'. Updating cache."
                                )
                                self._cwd = fetched_cwd
                            return self._cwd
                        else:
                            logger.warning(
                                f"_get_current_cwd: Path '{fetched_cwd}' (from BaseObject) is not a valid directory. Returning cached CWD: {self._cwd}"
                            )
                            return self._cwd
                    else:
                        logger.error(
                            f'_get_current_cwd: BaseObject also lacks Path attribute. Cannot determine CWD from result: {first_result}'
                        )
                        return self._cwd  # Return cached
                except AttributeError as ae:
                    logger.error(
                        f'_get_current_cwd: Error accessing BaseObject or its Path: {ae}. Result: {first_result}'
                    )
                    return self._cwd  # Return cached
                except Exception as ex:
                    logger.error(
                        f'_get_current_cwd: Unexpected error checking BaseObject: {ex}. Result: {first_result}'
                    )
                    return self._cwd  # Return cached

        # This path is taken if _run_ps_command returned [] or results.Count was 0
        logger.error(
            f'_get_current_cwd: No valid results received from Get-Location call. Returning cached CWD: {self._cwd}'
        )
        return self._cwd

    def execute(self, action: CmdRunAction) -> CmdOutputObservation | ErrorObservation:
        """Executes a command, potentially as a PowerShell background job for long-running tasks.
        Aligned with bash.py behavior regarding command execution and messages.

        Args:
            action: The command execution action.

        Returns:
            CmdOutputObservation or ErrorObservation.
        """
        if not self._initialized or self._closed:
            return ErrorObservation(
                content='PowerShell session is not initialized or has been closed.'
            )

        command = action.command.strip()
        timeout_seconds = action.timeout or 60  # Default to 60 seconds hard timeout
        is_input = action.is_input  # Check if it's intended as input

        # Detect if this is a background command (ending with &)
        run_in_background = False
        if command.endswith('&'):
            run_in_background = True
            command = command[:-1].strip()  # Remove the & and extra spaces
            logger.info(f"Detected background command: '{command}'")

        logger.info(
            f"Received command: '{command}', Timeout: {timeout_seconds}s, is_input: {is_input}, background: {run_in_background}"
        )

        # --- Simplified Active Job Handling (aligned with bash.py) ---
        with self._job_lock:
            if self.active_job:
                active_job_obj = self._get_job_object(self.active_job.Id)  # type: ignore[unreachable]
                job_is_finished = False
                final_output = ''  # Initialize before conditional assignment
                final_errors = []  # Initialize before conditional assignment
                current_job_state = None  # Initialize
                finished_job_id = (
                    self.active_job.Id
                )  # Store ID before potentially clearing self.active_job

                if active_job_obj:
                    current_job_state = active_job_obj.JobStateInfo.State
                    if current_job_state not in [JobState.Running, JobState.NotStarted]:
                        job_is_finished = True
                        logger.info(
                            f'Active job {finished_job_id} was finished ({current_job_state}) before receiving new command. Cleaning up.'
                        )
                        # Assign final output/errors here
                        final_output, final_errors = self._receive_job_output(
                            active_job_obj, keep=False
                        )  # Consume final output
                        remove_script = (
                            f'Remove-Job -Job (Get-Job -Id {finished_job_id})'
                        )
                        self._run_ps_command(remove_script)
                        # --- Reset persistent state ---
                        self._last_job_output = ''
                        self._last_job_error = []
                        self.active_job = None
                    # else: job still running, job_is_finished remains False
                else:
                    # Job object disappeared, consider it finished/gone
                    logger.warning(
                        f'Could not retrieve active job object {finished_job_id}. Assuming finished and clearing.'
                    )
                    job_is_finished = True
                    current_job_state = (
                        JobState.Failed
                    )  # Assume failed if object is gone
                    # Assign final output/errors here
                    final_output = ''  # No output retrievable
                    final_errors = ['[ERROR: Job object disappeared during check]']
                    # --- Reset persistent state ---
                    self._last_job_output = ''
                    self._last_job_error = []
                    self.active_job = None

                # If the job was found to be finished *during this check*, return its final state now.
                if job_is_finished:
                    # --- Calculate final new output/errors ---
                    new_output = final_output.removeprefix(
                        self._last_job_output
                    )  # final_output was from keep=False
                    last_error_set = set(
                        self._last_job_error
                    )  # Use the state *before* reset
                    new_errors = [e for e in final_errors if e not in last_error_set]

                    # Construct and return the observation for the completed job using the state captured during cleanup
                    exit_code = 0 if current_job_state == JobState.Completed else 1
                    output_builder = [new_output] if new_output else []
                    if new_errors:
                        output_builder.append('\\n[ERROR STREAM]')
                        output_builder.extend(new_errors)
                    content_for_return = '\\n'.join(output_builder).strip()

                    current_cwd = self._cwd  # Use cached CWD as job is gone
                    python_safe_cwd = current_cwd.replace('\\\\', '\\\\\\\\')
                    metadata = CmdOutputMetadata(
                        exit_code=exit_code, working_dir=python_safe_cwd
                    )
                    # Indicate this output is from the *previous* command that just finished.
                    metadata.prefix = (
                        '[Below is the output of the previous command.]\\n'
                    )
                    metadata.suffix = (
                        f'\\n[The command completed with exit code {exit_code}.]'
                    )
                    logger.info(
                        f"Returning final output for job {finished_job_id} which finished before command '{command}' was processed."
                    )  # Use finished_job_id
                    return CmdOutputObservation(
                        content=content_for_return,
                        command=action.command,  # The command that triggered this check (e.g., '')
                        metadata=metadata,
                    )

                # If job was NOT finished, check incoming command
                # This block only runs if the job is still active (job_is_finished is False)
                if not job_is_finished:
                    if command == '':
                        logger.info(
                            'Received empty command while job running. Checking job status.'
                        )
                        # Pass the timeout from the empty command action to _check_active_job
                        return self._check_active_job(timeout_seconds)
                    elif command == 'C-c':
                        logger.info('Received C-c while job running. Stopping job.')
                        return self._stop_active_job()
                    elif is_input:
                        # PowerShell session doesn't directly support stdin injection like bash.py/tmux
                        # This requires a different approach (e.g., named pipes, or specific cmdlets).
                        # For now, return an error indicating this limitation.
                        logger.warning(
                            f"Received input command '{command}' while job active, but direct input injection is not supported in this implementation."
                        )
                        # Get *new* output since last observation to provide context
                        cumulative_output, cumulative_errors = self._receive_job_output(
                            self.active_job, keep=True
                        )
                        new_output = cumulative_output.removeprefix(
                            self._last_job_output
                        )
                        last_error_set = set(self._last_job_error)
                        new_errors = [
                            e for e in cumulative_errors if e not in last_error_set
                        ]
                        output_builder = [new_output] if new_output else []
                        if new_errors:
                            output_builder.append('\\n[ERROR STREAM]')
                            output_builder.extend(new_errors)
                        # --- UPDATE persistent state ---
                        # Even though input fails, the user saw this output now
                        self._last_job_output = cumulative_output
                        self._last_job_error = list(set(cumulative_errors))
                        current_cwd = self._cwd
                        python_safe_cwd = current_cwd.replace('\\\\', '\\\\\\\\')
                        metadata = CmdOutputMetadata(
                            exit_code=-1, working_dir=python_safe_cwd
                        )  # Still running
                        metadata.prefix = (
                            '[Below is the output of the previous command.]\\n'
                        )
                        metadata.suffix = (
                            f"\\n[Your input command '{command}' was NOT processed. Direct input to running processes (is_input=True) "
                            'is not supported by this PowerShell session implementation. You can use C-c to stop the process.]'
                        )
                        return CmdOutputObservation(
                            content='\\n'.join(output_builder).strip(),
                            command=action.command,
                            metadata=metadata,
                        )

                    else:
                        # Any other command arrives while a job is running -> Reject it (bash.py behavior)
                        logger.warning(
                            f"Received new command '{command}' while job {self.active_job.Id} is active. New command NOT executed."
                        )
                        # Get *new* output since last observation to provide context
                        cumulative_output, cumulative_errors = self._receive_job_output(
                            self.active_job, keep=True
                        )
                        new_output = cumulative_output.removeprefix(
                            self._last_job_output
                        )
                        last_error_set = set(self._last_job_error)
                        new_errors = [
                            e for e in cumulative_errors if e not in last_error_set
                        ]
                        output_builder = [new_output] if new_output else []
                        if new_errors:
                            output_builder.append('\\n[ERROR STREAM]')
                            output_builder.extend(new_errors)
                        # --- UPDATE persistent state ---
                        # Even though command fails, the user saw this output now
                        self._last_job_output = cumulative_output
                        self._last_job_error = list(set(cumulative_errors))

                        current_cwd = self._cwd  # Use cached CWD
                        python_safe_cwd = current_cwd.replace('\\\\', '\\\\\\\\')
                        metadata = CmdOutputMetadata(
                            exit_code=-1, working_dir=python_safe_cwd
                        )  # Exit code -1 indicates still running
                        metadata.prefix = (
                            '[Below is the output of the previous command.]\n'
                        )
                        metadata.suffix = (
                            f'\n[Your command "{command}" is NOT executed. '
                            f'The previous command is still running - You CANNOT send new commands until the previous command is completed. '
                            'By setting `is_input` to `true`, you can interact with the current process: '
                            f'{TIMEOUT_MESSAGE_TEMPLATE}]'
                        )

                        return CmdOutputObservation(
                            content='\\n'.join(output_builder).strip(),
                            command=action.command,  # Return the command that was attempted
                            metadata=metadata,
                        )
            # --- End Active Job Handling ---

        # --- If we reach here, there is no active job ---

        # Handle empty command when NO job is active
        if command == '':
            logger.warning('Received empty command string (no active job).')
            current_cwd = self._get_current_cwd()  # Update CWD just in case
            python_safe_cwd = current_cwd.replace('\\\\', '\\\\\\\\')
            metadata = CmdOutputMetadata(exit_code=0, working_dir=python_safe_cwd)
            # Align error message with bash.py
            error_content = 'ERROR: No previous running command to retrieve logs from.'
            logger.warning(
                f'Returning specific error message for empty command: {error_content}'
            )
            # No extra suffix needed
            # metadata.suffix = f"\n[Empty command received (no active job). CWD: {metadata.working_dir}]"
            return CmdOutputObservation(
                content=error_content, command='', metadata=metadata
            )

        # Handle C-* when NO job is active/relevant
        if command.startswith('C-') and len(command) == 3:
            logger.warning(
                f'Received control character command: {command}. Not supported when no job active.'
            )
            current_cwd = self._cwd  # Use cached CWD
            python_safe_cwd = current_cwd.replace('\\\\', '\\\\\\\\')
            # Align error message with bash.py (no running command to interact with)
            return ErrorObservation(
                content='ERROR: No previous running command to interact with.'
            )

        # --- Validate command structure using PowerShell Parser ---
        # (Keep existing validation logic as it's PowerShell specific and useful)
        parse_errors = None
        statements = None
        try:
            # Parse the input command string
            ast, _, parse_errors = Parser.ParseInput(command, None)
            if parse_errors and parse_errors.Length > 0:
                error_messages = '\n'.join(
                    [
                        f'  - {err.Message} at Line {err.Extent.StartLineNumber}, Column {err.Extent.StartColumnNumber}'
                        for err in parse_errors
                    ]
                )
                logger.error(f'Command failed PowerShell parsing:\n{error_messages}')
                return ErrorObservation(
                    content=(
                        f'ERROR: Command could not be parsed by PowerShell.\n'
                        f'Syntax errors detected:\n{error_messages}'
                    )
                )
            statements = ast.EndBlock.Statements
            if statements.Count > 1:
                logger.error(
                    f'Detected {statements.Count} statements in the command. Only one is allowed.'
                )
                # Align error message with bash.py
                splited_cmds = [
                    str(s.Extent.Text) for s in statements
                ]  # Try to get text
                return ErrorObservation(
                    content=(
                        f'ERROR: Cannot execute multiple commands at once.\n'
                        f'Please run each command separately OR chain them into a single command via PowerShell operators (e.g., ; or |).\n'
                        f'Detected commands:\n{"\n".join(f"({i + 1}) {cmd}" for i, cmd in enumerate(splited_cmds))}'
                    )
                )
            elif statements.Count == 0 and not command.strip().startswith('#'):
                logger.warning(
                    'Received command that resulted in zero executable statements (likely whitespace or comment).'
                )
                # Treat as empty command if it parses to nothing
                return CmdOutputObservation(
                    content='',
                    command=command,
                    metadata=CmdOutputMetadata(exit_code=0, working_dir=self._cwd),
                )

        except Exception as parse_ex:
            logger.exception(f'Exception during PowerShell command parsing: {parse_ex}')
            return ErrorObservation(
                content=f'ERROR: An exception occurred while parsing the command: {parse_ex}'
            )
        # --- End validation ---

        # === Synchronous Execution Path (for CWD commands) ===
        if statements and statements.Count == 1:
            statement = statements[0]
            try:
                from System.Management.Automation.Language import (
                    CommandAst,
                    PipelineAst,
                )

                # Check PipelineAst
                if isinstance(statement, PipelineAst):
                    pipeline_elements = statement.PipelineElements
                    if (
                        pipeline_elements
                        and pipeline_elements.Count == 1
                        and isinstance(pipeline_elements[0], CommandAst)
                    ):
                        command_ast = pipeline_elements[0]
                        command_name = command_ast.GetCommandName()
                        if command_name and command_name.lower() in [
                            'set-location',
                            'cd',
                            'push-location',
                            'pop-location',
                        ]:
                            logger.info(
                                f'execute: Identified CWD command via PipelineAst: {command_name}'
                            )
                            # Run command and prepare proper CmdOutputObservation
                            ps_results = self._run_ps_command(command)
                            # Get current working directory after CWD command
                            current_cwd = self._get_current_cwd()
                            python_safe_cwd = current_cwd.replace('\\\\', '\\\\\\\\')

                            # Convert results to string output if any
                            output = (
                                '\n'.join([str(r) for r in ps_results])
                                if ps_results
                                else ''
                            )

                            return CmdOutputObservation(
                                content=output,
                                command=command,
                                metadata=CmdOutputMetadata(
                                    exit_code=0, working_dir=python_safe_cwd
                                ),
                            )
                # Check direct CommandAst
                elif isinstance(statement, CommandAst):
                    command_name = statement.GetCommandName()
                    if command_name and command_name.lower() in [
                        'set-location',
                        'cd',
                        'push-location',
                        'pop-location',
                    ]:
                        logger.info(
                            f'execute: Identified CWD command via direct CommandAst: {command_name}'
                        )
                        # Run command and prepare proper CmdOutputObservation
                        ps_results = self._run_ps_command(command)
                        # Get current working directory after CWD command
                        current_cwd = self._get_current_cwd()
                        python_safe_cwd = current_cwd.replace('\\\\', '\\\\\\\\')

                        # Convert results to string output if any
                        output = (
                            '\n'.join([str(r) for r in ps_results])
                            if ps_results
                            else ''
                        )

                        return CmdOutputObservation(
                            content=output,
                            command=command,
                            metadata=CmdOutputMetadata(
                                exit_code=0, working_dir=python_safe_cwd
                            ),
                        )
            except ImportError as imp_err:
                logger.error(
                    f'execute: Failed to import CommandAst: {imp_err}. Cannot check for CWD commands.'
                )
            except Exception as ast_err:
                logger.error(f'execute: Error checking command AST: {ast_err}')

        # === Asynchronous Execution Path (for non-CWD commands) ===
        logger.info(
            f"execute: Entering asynchronous execution path for command: '{command}'"
        )

        # --- Start the command as a new asynchronous job ---
        # Reset state for the new job
        self._last_job_output = ''
        self._last_job_error = []

        ps_start = None
        job = None
        output_builder = []
        all_errors = []
        exit_code = 1
        timed_out = False
        job_start_failed = False
        job_id = None

        try:
            ps_start = PowerShell.Create()
            ps_start.Runspace = self.runspace
            escaped_cwd = self._cwd.replace("'", "''")
            # Check $? after the command. If it's false, exit 1.
            start_job_script = f"Start-Job -ScriptBlock {{ Set-Location '{escaped_cwd}'; {command}; if (-not $?) {{ exit 1 }} }}"

            logger.info(f'Starting command as PowerShell job: {command}')
            ps_start.AddScript(start_job_script)
            start_results = ps_start.Invoke()

            if ps_start.Streams.Error:
                errors = [str(e) for e in ps_start.Streams.Error]
                logger.error(f'Errors during Start-Job execution: {errors}')
                all_errors.extend(errors)

            ps_get = PowerShell.Create()
            ps_get.Runspace = self.runspace
            get_job_script = 'Get-Job | Sort-Object -Property Id -Descending | Select-Object -First 1'
            ps_get.AddScript(get_job_script)
            get_results = ps_get.Invoke()

            if ps_get.Streams.Error:
                errors = [str(e) for e in ps_get.Streams.Error]
                logger.error(f'Errors getting latest job: {errors}')
                all_errors.extend(errors)
                job_start_failed = True

            if not job_start_failed and get_results and len(get_results) > 0:
                potential_job = get_results[0]
                try:
                    underlying_job = potential_job.BaseObject
                    job_state_test = underlying_job.JobStateInfo.State
                    job = underlying_job
                    job_id = job.Id

                    # For background commands, don't track the job in the session
                    if not run_in_background:
                        with self._job_lock:
                            self.active_job = job

                    logger.info(
                        f'Job retrieved successfully. Job ID: {job.Id}, State: {job_state_test}, Background: {run_in_background}'
                    )

                    if job_state_test == JobState.Failed:
                        logger.error(f'Job {job.Id} failed immediately after starting.')
                        output_chunk, error_chunk = self._receive_job_output(
                            job, keep=False
                        )
                        if output_chunk:
                            output_builder.append(output_chunk)
                        if error_chunk:
                            all_errors.extend(error_chunk)
                        job_start_failed = True
                        remove_script = f'Remove-Job -Job (Get-Job -Id {job.Id})'
                        self._run_ps_command(remove_script)
                        with self._job_lock:
                            self.active_job = None
                except AttributeError as e:
                    logger.exception(
                        f'Get-Job returned an object without expected properties on BaseObject: {e}'
                    )
                    all_errors.append('Get-Job did not return a valid Job object.')
                    job_start_failed = True

            elif not job_start_failed:
                logger.error('Get-Job did not return any results.')
                all_errors.append('Get-Job did not return any results.')
                job_start_failed = True

        except Exception as start_ex:
            logger.exception(f'Exception during job start/retrieval: {start_ex}')
            all_errors.append(f'[Job Start/Get Exception: {start_ex}]')
            job_start_failed = True
        finally:
            if ps_start:
                ps_start.Dispose()
            if 'ps_get' in locals() and ps_get:
                ps_get.Dispose()

        if job_start_failed:
            current_cwd = self._get_current_cwd()
            python_safe_cwd = current_cwd.replace('\\\\', '\\\\\\\\')
            metadata = CmdOutputMetadata(exit_code=1, working_dir=python_safe_cwd)
            # Use ErrorObservation for critical failures like job start
            return ErrorObservation(
                content='Failed to start PowerShell job.\n[ERRORS]\n'
                + '\n'.join(all_errors)
            )

        # For background commands, return immediately with success
        if run_in_background:
            current_cwd = self._get_current_cwd()
            python_safe_cwd = current_cwd.replace('\\\\', '\\\\\\\\')
            metadata = CmdOutputMetadata(exit_code=0, working_dir=python_safe_cwd)
            metadata.suffix = f'\n[Command started as background job {job_id}.]'
            return CmdOutputObservation(
                content=f'[Started background job {job_id}]',
                command=f'{command} &',
                metadata=metadata,
            )

        # --- Monitor the Job ---
        start_time = time.monotonic()
        monitoring_loop_finished = False
        shutdown_requested = False
        final_state = JobState.Failed

        latest_cumulative_output = (
            ''  # Tracks the absolute latest cumulative output seen in this loop
        )
        latest_cumulative_errors = []  # Tracks the absolute latest cumulative errors seen in this loop

        while not monitoring_loop_finished:
            if not should_continue():
                logger.warning('Shutdown signal received during job monitoring.')
                shutdown_requested = True
                monitoring_loop_finished = True
                exit_code = -1
                continue

            elapsed_seconds = time.monotonic() - start_time
            if elapsed_seconds > timeout_seconds:
                logger.warning(
                    f'Command job monitoring exceeded timeout ({timeout_seconds}s). Leaving job running.'
                )
                timed_out = True
                monitoring_loop_finished = True
                exit_code = -1
                continue

            current_job_obj = self._get_job_object(job_id)
            if not current_job_obj:
                logger.error(f'Job {job_id} object disappeared during monitoring.')
                all_errors.append('[Job object lost during monitoring]')
                monitoring_loop_finished = True
                exit_code = 1
                final_state = JobState.Failed
                # Reset state as job is gone
                self._last_job_output = ''
                self._last_job_error = []
                continue

            # Poll output (keep=True) -> Returns CUMULATIVE output/errors
            polled_cumulative_output, polled_cumulative_errors = (
                self._receive_job_output(current_job_obj, keep=True)
            )

            # Update the latest cumulative state seen in this loop
            latest_cumulative_output = polled_cumulative_output
            latest_cumulative_errors = polled_cumulative_errors

            # Check job state
            current_state = current_job_obj.JobStateInfo.State
            if current_state not in [JobState.Running, JobState.NotStarted]:
                logger.info(
                    f'Job {job_id} finished monitoring loop with state: {current_state}'
                )
                monitoring_loop_finished = True
                final_state = current_state
                continue

            time.sleep(0.1)

        # --- Monitoring loop finished ---

        job_finished_naturally = (
            not timed_out
            and not shutdown_requested
            and final_state in [JobState.Completed, JobState.Stopped, JobState.Failed]
        )

        determined_cwd = self._cwd
        final_output_content = ''
        final_error_content = []

        if job_finished_naturally:
            logger.info(
                f'Job {job_id} finished naturally with state: {final_state}. Clearing final output buffer.'
            )
            final_cumulative_output = ''
            final_cumulative_errors: list[str] = []
            final_job_obj = self._get_job_object(job_id)
            if final_job_obj:
                # Get final output/errors with keep=False
                final_cumulative_output, final_cumulative_errors = (
                    self._receive_job_output(final_job_obj, keep=False)
                )
                # Always calculate the output relative to the last observation returned
                final_output_content = final_cumulative_output.removeprefix(
                    self._last_job_output
                )
                # Also calculate final errors relative to last observation returned
                last_error_set = set(self._last_job_error)
                final_error_content = [
                    e for e in final_cumulative_errors if e not in last_error_set
                ]
            else:
                logger.warning(
                    f'Could not get final job object {job_id} to clear output buffer.'
                )
                # If object is gone, output is what was last seen relative to last observation
                final_output_content = latest_cumulative_output.removeprefix(
                    self._last_job_output
                )
                last_error_set = set(self._last_job_error)
                final_error_content = [
                    e for e in latest_cumulative_errors if e not in last_error_set
                ]

            exit_code = 0 if final_state == JobState.Completed else 1

            if final_state == JobState.Completed:
                logger.info(f'Job {job_id} completed successfully. Querying final CWD.')
                determined_cwd = self._get_current_cwd()
            else:
                logger.info(
                    f'Job {job_id} finished but did not complete successfully ({final_state}). Using cached CWD: {self._cwd}'
                )
                determined_cwd = self._cwd

            with self._job_lock:  # Lock to clear active_job
                remove_script = f'Remove-Job -Job (Get-Job -Id {job_id})'
                self._run_ps_command(remove_script)
                self.active_job = None
                logger.info(f'Cleaned up finished job {job_id}')

        else:
            logger.info(
                f'Job {job_id} did not finish naturally (timeout={timed_out}, shutdown={shutdown_requested}). Using cached CWD: {self._cwd}'
            )
            determined_cwd = self._cwd
            # Exit code is already -1 from loop exit reason

            # --- Calculate new output/errors relative to last observation (using latest from loop) ---
            final_output_content = latest_cumulative_output.removeprefix(
                self._last_job_output
            )
            final_error_content = [
                e for e in latest_cumulative_errors if e not in self._last_job_error
            ]

            # --- Update persistent state ---
            self._last_job_output = latest_cumulative_output
            self._last_job_error = list(
                set(latest_cumulative_errors)
            )  # Store unique errors

        python_safe_cwd = determined_cwd.replace('\\\\', '\\\\\\\\')

        # Combine unique output chunks for final observation
        # Using a set ensures uniqueness if chunks were identical across polls
        # Join accumulated output_builder parts
        final_output = final_output_content
        if final_error_content:  # Use the calculated final *new* errors
            error_stream_text = '\n'.join(final_error_content)
            if final_output:
                final_output += f'\n[ERROR STREAM]\n{error_stream_text}'
            else:
                final_output = f'[ERROR STREAM]\n{error_stream_text}'
            if exit_code == 0:  # Only check exit code if job finished naturally
                logger.info(
                    f'Detected errors in stream ({len(final_error_content)} records) but job state was Completed. Forcing exit_code to 1.'
                )
                exit_code = 1

        # Create metadata
        metadata = CmdOutputMetadata(exit_code=exit_code, working_dir=python_safe_cwd)

        # Determine Suffix
        if timed_out:
            # Align suffix with bash.py timeout message
            suffix = (
                f'\n[The command timed out after {timeout_seconds} seconds. '
                f'{TIMEOUT_MESSAGE_TEMPLATE}]'
            )
        elif shutdown_requested:
            # Align suffix with bash.py equivalent (though bash.py might not have specific shutdown message)
            suffix = f'\n[Command execution cancelled due to shutdown signal. Exit Code: {exit_code}]'
        elif job_finished_naturally:
            # Align suffix with bash.py completed message
            suffix = f'\n[The command completed with exit code {exit_code}.]'
        else:  # Should not happen, but defensive fallback
            suffix = f'\n[Command execution finished. State: {final_state}, Exit Code: {exit_code}]'

        metadata.suffix = suffix

        return CmdOutputObservation(
            content=final_output, command=command, metadata=metadata
        )

    def close(self) -> None:
        """Closes the PowerShell runspace and releases resources, stopping any active job."""
        if self._closed:
            return

        logger.info('Closing PowerShell session runspace.')

        # Stop and remove any active job before closing runspace
        with self._job_lock:
            if self.active_job:
                logger.warning(  # type: ignore[unreachable]
                    f'Session closing with active job {self.active_job.Id}. Attempting to stop and remove.'
                )
                job_id = self.active_job.Id
                try:
                    # Ensure job object exists before trying to stop/remove
                    active_job_obj = self._get_job_object(job_id)
                    if active_job_obj:
                        stop_script = f'Stop-Job -Job (Get-Job -Id {job_id})'
                        self._run_ps_command(
                            stop_script
                        )  # Use helper before runspace closes
                        time.sleep(0.1)
                        remove_script = f'Remove-Job -Job (Get-Job -Id {job_id})'
                        self._run_ps_command(remove_script)
                        logger.info(
                            f'Stopped and removed active job {job_id} during close.'
                        )
                    else:
                        logger.warning(
                            f'Could not find job object {job_id} to stop/remove during close.'
                        )
                except Exception as e:
                    logger.error(
                        f'Error stopping/removing job {job_id} during close: {e}'
                    )
                # --- Reset state even if stop/remove failed ---
                self._last_job_output = ''
                self._last_job_error = []
                self.active_job = None

        if hasattr(self, 'runspace') and self.runspace:
            try:
                # Check state using System.Management.Automation.Runspaces namespace
                # Get the state info object first to avoid potential pythonnet issues with nested access
                runspace_state_info = self.runspace.RunspaceStateInfo
                if runspace_state_info.State == RunspaceState.Opened:
                    self.runspace.Close()
                self.runspace.Dispose()
                logger.info('PowerShell runspace closed and disposed.')
            except Exception as e:
                logger.exception(f'Error closing/disposing PowerShell runspace: {e}')

        self.runspace = None
        self._initialized = False
        self._closed = True

    def __del__(self) -> None:
        """Destructor ensures the runspace is closed."""
        self.close()
