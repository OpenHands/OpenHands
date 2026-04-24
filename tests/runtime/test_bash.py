"""Bash-related tests for the DockerRuntime, which connects to the ActionExecutor running in the sandbox."""

import os
import sys
import time
from pathlib import Path

import pytest
from conftest import (
    _close_test_runtime,
    _load_runtime,
)

from openhands.core.logger import openhands_logger as logger
from openhands.events.action import CmdRunAction
from openhands.events.observation import CmdOutputObservation, ErrorObservation
from openhands.runtime.impl.cli.cli_runtime import CLIRuntime
from openhands.runtime.impl.local.local_runtime import LocalRuntime
from openhands.runtime.utils.bash_constants import TIMEOUT_MESSAGE_TEMPLATE


def get_timeout_suffix(timeout_seconds):
    """Helper function to generate the expected timeout suffix."""
    return (
        f'[The command timed out after {timeout_seconds} seconds. '
        f'{TIMEOUT_MESSAGE_TEMPLATE}]'
    )


# ============================================================================================================================
# Bash-specific tests
# ============================================================================================================================


# Helper function to determine if running on Windows
def is_windows():
    return sys.platform == 'win32'


def _run_cmd_action(runtime, custom_command: str):
    action = CmdRunAction(command=custom_command)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    assert isinstance(obs, (CmdOutputObservation, ErrorObservation))
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    return obs


# Get platform-appropriate command
def get_platform_command(linux_cmd, windows_cmd):
    return windows_cmd if is_windows() else linux_cmd


def test_bash_server(temp_dir, runtime_cls, run_as_openhands, dynamic_port):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        # Use python -u for unbuffered output, potentially helping capture initial output on Windows
        action = CmdRunAction(command=f'python -u -m http.server {dynamic_port}')
        action.set_hard_timeout(1)
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert isinstance(obs, CmdOutputObservation)
        assert obs.exit_code == -1
        assert 'Serving HTTP on' in obs.content

        if runtime_cls == CLIRuntime:
            assert '[The command timed out after 1.0 seconds.]' in obs.metadata.suffix
        else:
            assert get_timeout_suffix(1.0) in obs.metadata.suffix

        action = CmdRunAction(command='C-c', is_input=True)
        action.set_hard_timeout(30)
        obs_interrupt = runtime.run_action(action)
        logger.info(obs_interrupt, extra={'msg_type': 'OBSERVATION'})

        if runtime_cls == CLIRuntime:
            assert isinstance(obs_interrupt, ErrorObservation)
            assert (
                "CLIRuntime does not support interactive input from the agent (e.g., 'C-c'). The command 'C-c' was not sent to any process."
                in obs_interrupt.content
            )
            assert obs_interrupt.error_id == 'AGENT_ERROR$BAD_ACTION'
        else:
            assert isinstance(obs_interrupt, CmdOutputObservation)
            assert obs_interrupt.exit_code == 0
            if not is_windows():
                # Linux/macOS behavior
                assert 'Keyboard interrupt received, exiting.' in obs_interrupt.content
                assert (
                    config.workspace_mount_path_in_sandbox
                    in obs_interrupt.metadata.working_dir
                )

        # Verify the server is actually stopped by trying to start another one
        # on the same port (regardless of OS)
        action = CmdRunAction(command='ls')
        action.set_hard_timeout(3)
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert isinstance(obs, CmdOutputObservation)
        assert obs.exit_code == 0
        # Check that the interrupt message is NOT present in subsequent output
        assert 'Keyboard interrupt received, exiting.' not in obs.content
        # Check working directory remains correct after interrupt handling
        if runtime_cls == CLIRuntime:
            # For CLIRuntime, working_dir is the absolute host path
            assert obs.metadata.working_dir == config.workspace_base
        else:
            # For other runtimes (e.g., Docker), it's relative to or contains the sandbox path
            assert config.workspace_mount_path_in_sandbox in obs.metadata.working_dir

        # run it again!
        action = CmdRunAction(command=f'python -u -m http.server {dynamic_port}')
        action.set_hard_timeout(1)
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert isinstance(obs, CmdOutputObservation)
        assert obs.exit_code == -1
        assert 'Serving HTTP on' in obs.content

    finally:
        _close_test_runtime(runtime)


def test_bash_background_server(temp_dir, runtime_cls, run_as_openhands, dynamic_port):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    server_port = dynamic_port
    try:
        # Start the server, expect it to timeout (run in background manner)
        action = CmdRunAction(f'python3 -m http.server {server_port} &')
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert isinstance(obs, CmdOutputObservation)

        if runtime_cls == CLIRuntime:
            # The '&' does not detach cleanly; the PTY session remains active.
            # the main cmd ends, then the server may receive SIGHUP.
            assert obs.exit_code == 0

            # Give the server a moment to be ready
            time.sleep(1)

            # `curl --fail` exits non-zero if connection fails or server returns an error.
            # Use a short connect timeout as the server is expected to be down.
            curl_action = CmdRunAction(
                f'curl --fail --connect-timeout 1 http://localhost:{server_port}'
            )
            curl_obs = runtime.run_action(curl_action)
            logger.info(curl_obs, extra={'msg_type': 'OBSERVATION'})
            assert isinstance(curl_obs, CmdOutputObservation)
            assert curl_obs.exit_code != 0

            # Confirm with pkill (CLIRuntime is assumed non-Windows here).
            # pkill returns 1 if no processes were matched.
            kill_action = CmdRunAction('pkill -f "http.server"')
            kill_obs = runtime.run_action(kill_action)
            logger.info(kill_obs, extra={'msg_type': 'OBSERVATION'})
            assert isinstance(kill_obs, CmdOutputObservation)
            # For CLIRuntime, bash -c "cmd &" exits quickly, orphaning "cmd".
            # CLIRuntime's timeout tries to kill the already-exited bash -c.
            # The orphaned http.server continues running.
            # So, pkill should find and kill the server.
            assert kill_obs.exit_code == 0
        else:
            assert obs.exit_code == 0

            # Give the server a moment to be ready
            time.sleep(1)

            # Verify the server is running by curling it
            if is_windows():
                curl_action = CmdRunAction(
                    f'Invoke-WebRequest -Uri http://localhost:{server_port} -UseBasicParsing | Select-Object -ExpandProperty Content'
                )
            else:
                curl_action = CmdRunAction(f'curl http://localhost:{server_port}')
            curl_obs = runtime.run_action(curl_action)
            logger.info(curl_obs, extra={'msg_type': 'OBSERVATION'})
            assert isinstance(curl_obs, CmdOutputObservation)
            assert curl_obs.exit_code == 0
            # Check for content typical of python http.server directory listing
            assert 'Directory listing for' in curl_obs.content

            # Kill the server
            if is_windows():
                # This assumes PowerShell context if LocalRuntime is used on Windows.
                kill_action = CmdRunAction('Get-Job | Stop-Job')
            else:
                kill_action = CmdRunAction('pkill -f "http.server"')
            kill_obs = runtime.run_action(kill_action)
            logger.info(kill_obs, extra={'msg_type': 'OBSERVATION'})
            assert isinstance(kill_obs, CmdOutputObservation)
            assert kill_obs.exit_code == 0

    finally:
        _close_test_runtime(runtime)


def test_multiline_commands(temp_dir, runtime_cls):
    runtime, config = _load_runtime(temp_dir, runtime_cls)
    try:
        if is_windows():
            # Windows PowerShell version using backticks for line continuation
            obs = _run_cmd_action(runtime, 'Write-Output `\n "foo"')
            assert obs.exit_code == 0, 'The exit code should be 0.'
            assert 'foo' in obs.content

            # test multiline output
            obs = _run_cmd_action(runtime, 'Write-Output "hello`nworld"')
            assert obs.exit_code == 0, 'The exit code should be 0.'
            assert 'hello\nworld' in obs.content

            # test whitespace
            obs = _run_cmd_action(runtime, 'Write-Output "a`n`n`nz"')
            assert obs.exit_code == 0, 'The exit code should be 0.'
            assert '\n\n\n' in obs.content
        else:
            # Original Linux bash version
            # single multiline command
            obs = _run_cmd_action(runtime, 'echo \\\n -e "foo"')
            assert obs.exit_code == 0, 'The exit code should be 0.'
            assert 'foo' in obs.content

            # test multiline echo
            obs = _run_cmd_action(runtime, 'echo -e "hello\nworld"')
            assert obs.exit_code == 0, 'The exit code should be 0.'
            assert 'hello\nworld' in obs.content

            # test whitespace
            obs = _run_cmd_action(runtime, 'echo -e "a\\n\\n\\nz"')
            assert obs.exit_code == 0, 'The exit code should be 0.'
            assert '\n\n\n' in obs.content
    finally:
        _close_test_runtime(runtime)


@pytest.mark.skipif(
    is_windows(), reason='Test relies on Linux bash-specific complex commands'
)
def test_complex_commands(temp_dir, runtime_cls, run_as_openhands):
    cmd = """count=0; tries=0; while [ $count -lt 3 ]; do result=$(echo "Heads"); tries=$((tries+1)); echo "Flip $tries: $result"; if [ "$result" = "Heads" ]; then count=$((count+1)); else count=0; fi; done; echo "Got 3 heads in a row after $tries flips!";"""

    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        obs = _run_cmd_action(runtime, cmd)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert obs.exit_code == 0, 'The exit code should be 0.'
        assert 'Got 3 heads in a row after 3 flips!' in obs.content

    finally:
        _close_test_runtime(runtime)


def test_no_ps2_in_output(temp_dir, runtime_cls, run_as_openhands):
    """Test that the PS2 sign is not added to the output of a multiline command."""
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        if is_windows():
            obs = _run_cmd_action(runtime, 'Write-Output "hello`nworld"')
        else:
            obs = _run_cmd_action(runtime, 'echo -e "hello\nworld"')
        assert obs.exit_code == 0, 'The exit code should be 0.'

        assert 'hello\nworld' in obs.content
        assert '>' not in obs.content
    finally:
        _close_test_runtime(runtime)


@pytest.mark.skipif(
    is_windows(), reason='Test uses Linux-specific bash loops and sed commands'
)
def test_multiline_command_loop(temp_dir, runtime_cls):
    # https://github.com/OpenHands/OpenHands/issues/3143
    init_cmd = """mkdir -p _modules && \
for month in {01..04}; do
    for day in {01..05}; do
        touch "_modules/2024-${month}-${day}-sample.md"
    done
done && echo "created files"
"""
    follow_up_cmd = """for file in _modules/*.md; do
    new_date=$(echo $file | sed -E 's/2024-(01|02|03|04)-/2024-/;s/2024-01/2024-08/;s/2024-02/2024-09/;s/2024-03/2024-10/;s/2024-04/2024-11/')
    mv "$file" "$new_date"
done && echo "success"
"""
    runtime, config = _load_runtime(temp_dir, runtime_cls)
    try:
        obs = _run_cmd_action(runtime, init_cmd)
        assert obs.exit_code == 0, 'The exit code should be 0.'
        assert 'created files' in obs.content

        obs = _run_cmd_action(runtime, follow_up_cmd)
        assert obs.exit_code == 0, 'The exit code should be 0.'
        assert 'success' in obs.content
    finally:
        _close_test_runtime(runtime)


@pytest.mark.skipif(
    os.getenv('TEST_RUNTIME') == 'cli',
    reason='CLIRuntime uses bash -c which handles newline-separated commands. This test expects rejection. See test_cliruntime_multiple_newline_commands.',
)
def test_multiple_multiline_commands(temp_dir, runtime_cls, run_as_openhands):
    if is_windows():
        cmds = [
            'Get-ChildItem',
            'Write-Output "hello`nworld"',
            """Write-Output "hello it's me\"""",
            """Write-Output `
    ('hello ' + `
    'world')""",
            """Write-Output 'hello\nworld\nare\nyou\nthere?'""",
            """Write-Output 'hello\nworld\nare\nyou\n\nthere?'""",
            """Write-Output 'hello\nworld "'""",  # Escape the trailing double quote
        ]
    else:
        cmds = [
            'ls -l',
            'echo -e "hello\nworld"',
            """echo -e "hello it's me\"""",
            """echo \\
    -e 'hello' \\
    world""",
            """echo -e 'hello\\nworld\\nare\\nyou\\nthere?'""",
            """echo -e 'hello\nworld\nare\nyou\n\nthere?'""",
            """echo -e 'hello\nworld "'""",
        ]
    joined_cmds = '\n'.join(cmds)

    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        # First test that running multiple commands at once fails
        obs = _run_cmd_action(runtime, joined_cmds)
        assert isinstance(obs, ErrorObservation)
        assert 'Cannot execute multiple commands at once' in obs.content

        # Now run each command individually and verify they work
        results = []
        for cmd in cmds:
            obs = _run_cmd_action(runtime, cmd)
            assert isinstance(obs, CmdOutputObservation)
            assert obs.exit_code == 0
            results.append(obs.content)

        # Verify all expected outputs are present
        if is_windows():
            # Get-ChildItem should execute successfully (no specific content check needed)
            pass  # results[0] contains directory listing output
        else:
            assert 'total 0' in results[0]  # ls -l
        assert 'hello\nworld' in results[1]  # echo -e "hello\nworld"
        assert "hello it's me" in results[2]  # echo -e "hello it\'s me"
        assert 'hello world' in results[3]  # echo -e 'hello' world
        assert (
            'hello\nworld\nare\nyou\nthere?' in results[4]
        )  # echo -e 'hello\nworld\nare\nyou\nthere?'
        assert (
            'hello\nworld\nare\nyou\n\nthere?' in results[5]
        )  # echo -e with literal newlines
        assert 'hello\nworld "' in results[6]  # echo -e with quote
    finally:
        _close_test_runtime(runtime)


def test_cliruntime_multiple_newline_commands(temp_dir, run_as_openhands):
    # This test is specific to CLIRuntime
    runtime_cls = CLIRuntime
    if is_windows():
        # Minimal check for Windows if CLIRuntime were to support it robustly with PowerShell for this.
        # For now, this test primarily targets the bash -c behavior on non-Windows.
        pytest.skip(
            'CLIRuntime newline command test primarily for non-Windows bash behavior'
        )
        # cmds = [
        #     'Get-ChildItem -Name .git_config', # Simpler command
        #     'Write-Output "hello`nworld"'
        # ]
        # expected_outputs = ['.git_config', 'hello\nworld']
    else:
        cmds = [
            'echo "hello"',  # A command that will always work
            'echo -e "hello\nworld"',
            """echo -e "hello it's me\"""",
        ]
        expected_outputs = [
            'hello',  # Simple string output
            'hello\nworld',
            "hello it's me",
        ]  # Simplified expectations
    joined_cmds = '\n'.join(cmds)

    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        obs = _run_cmd_action(runtime, joined_cmds)
        assert isinstance(obs, CmdOutputObservation)
        assert obs.exit_code == 0
        # Check that parts of each command's expected output are present
        for expected_part in expected_outputs:
            assert expected_part in obs.content
    finally:
        _close_test_runtime(runtime)


def test_cmd_run(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        if is_windows():
            # Windows PowerShell version
            obs = _run_cmd_action(
                runtime, f'Get-ChildItem -Path {config.workspace_mount_path_in_sandbox}'
            )
            assert obs.exit_code == 0

            obs = _run_cmd_action(runtime, 'Get-ChildItem')
            assert obs.exit_code == 0

            obs = _run_cmd_action(runtime, 'New-Item -ItemType Directory -Path test')
            assert obs.exit_code == 0

            obs = _run_cmd_action(runtime, 'Get-ChildItem')
            assert obs.exit_code == 0
            assert 'test' in obs.content

            obs = _run_cmd_action(runtime, 'New-Item -ItemType File -Path test/foo.txt')
            assert obs.exit_code == 0

            obs = _run_cmd_action(runtime, 'Get-ChildItem test')
            assert obs.exit_code == 0
            assert 'foo.txt' in obs.content

            # clean up
            _run_cmd_action(runtime, 'Remove-Item -Recurse -Force test')
            assert obs.exit_code == 0
        else:
            # Unix version
            obs = _run_cmd_action(
                runtime, f'ls -l {config.workspace_mount_path_in_sandbox}'
            )
            assert obs.exit_code == 0

            obs = _run_cmd_action(runtime, 'ls -l')
            assert obs.exit_code == 0
            assert 'total 0' in obs.content

            obs = _run_cmd_action(runtime, 'mkdir test')
            assert obs.exit_code == 0

            obs = _run_cmd_action(runtime, 'ls -l')
            assert obs.exit_code == 0
            if (
                run_as_openhands
                and runtime_cls != CLIRuntime
                and runtime_cls != LocalRuntime
            ):
                assert 'openhands' in obs.content
            elif runtime_cls == LocalRuntime or runtime_cls == CLIRuntime:
                # For CLI and Local runtimes, the user depends on the actual environment
                # In CI it might be a non-root user, in cloud environments it might be root
                # We just check that the command succeeded and the directory was created
                pass  # Skip user-specific assertions for environment independence
            else:
                assert 'root' in obs.content
            assert 'test' in obs.content

            obs = _run_cmd_action(runtime, 'touch test/foo.txt')
            assert obs.exit_code == 0

            obs = _run_cmd_action(runtime, 'ls -l test')
            assert obs.exit_code == 0
            assert 'foo.txt' in obs.content

            # clean up: this is needed, since CI will not be
            # run as root, and this test may leave a file
            # owned by root
            _run_cmd_action(runtime, 'rm -rf test')
            assert obs.exit_code == 0
    finally:
        _close_test_runtime(runtime)


@pytest.mark.skipif(
    sys.platform != 'win32' and os.getenv('TEST_RUNTIME') == 'cli',
    reason='CLIRuntime runs as the host user, so ~ is the host home. This test assumes a sandboxed user.',
)
def test_run_as_user_correct_home_dir(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        if is_windows():
            # Windows PowerShell version
            obs = _run_cmd_action(runtime, 'cd $HOME && Get-Location')
            assert obs.exit_code == 0
            # Check for Windows-style home paths
            if runtime_cls == LocalRuntime:
                assert (
                    os.getenv('USERPROFILE') in obs.content
                    or os.getenv('HOME') in obs.content
                )
            # For non-local runtime, we are less concerned with precise paths
        else:
            # Original Linux version
            obs = _run_cmd_action(runtime, 'cd ~ && pwd')
            assert obs.exit_code == 0
            if runtime_cls == LocalRuntime:
                assert os.getenv('HOME') in obs.content
            elif run_as_openhands:
                assert '/home/openhands' in obs.content
            else:
                assert '/root' in obs.content
    finally:
        _close_test_runtime(runtime)


def test_multi_cmd_run_in_single_line(temp_dir, runtime_cls):
    runtime, config = _load_runtime(temp_dir, runtime_cls)
    try:
        if is_windows():
            # Windows PowerShell version using semicolon
            obs = _run_cmd_action(runtime, 'Get-Location && Get-ChildItem')
            assert obs.exit_code == 0
            assert config.workspace_mount_path_in_sandbox in obs.content
            # Git config is now handled by runtime base class, not as a file
        else:
            # Original Linux version using &&
            obs = _run_cmd_action(runtime, 'pwd && ls -l')
            assert obs.exit_code == 0
            assert config.workspace_mount_path_in_sandbox in obs.content
            assert 'total 0' in obs.content
    finally:
        _close_test_runtime(runtime)


def test_stateful_cmd(temp_dir, runtime_cls):
    runtime, config = _load_runtime(temp_dir, runtime_cls)
    try:
        if is_windows():
            # Windows PowerShell version
            obs = _run_cmd_action(
                runtime, 'New-Item -ItemType Directory -Path test -Force'
            )
            assert obs.exit_code == 0, 'The exit code should be 0.'

            obs = _run_cmd_action(runtime, 'Set-Location test')
            assert obs.exit_code == 0, 'The exit code should be 0.'

            obs = _run_cmd_action(runtime, 'Get-Location')
            assert obs.exit_code == 0, 'The exit code should be 0.'
            # Account for both forward and backward slashes in path
            norm_path = config.workspace_mount_path_in_sandbox.replace(
                '\\', '/'
            ).replace('//', '/')
            test_path = f'{norm_path}/test'.replace('//', '/')
            assert test_path in obs.content.replace('\\', '/')
        else:
            # Original Linux version
            obs = _run_cmd_action(runtime, 'mkdir -p test')
            assert obs.exit_code == 0, 'The exit code should be 0.'

            if runtime_cls == CLIRuntime:
                # For CLIRuntime, test CWD change and command execution within a single action
                # as CWD is enforced in the workspace.
                obs = _run_cmd_action(runtime, 'cd test && pwd')
            else:
                # For other runtimes, test stateful CWD change across actions
                obs = _run_cmd_action(runtime, 'cd test')
                assert obs.exit_code == 0, 'The exit code should be 0 for cd test.'
                obs = _run_cmd_action(runtime, 'pwd')

            assert obs.exit_code == 0, (
                'The exit code for the pwd command (or combined command) should be 0.'
            )
            assert (
                f'{config.workspace_mount_path_in_sandbox}/test' in obs.content.strip()
            )
    finally:
        _close_test_runtime(runtime)


def test_failed_cmd(temp_dir, runtime_cls):
    runtime, config = _load_runtime(temp_dir, runtime_cls)
    try:
        obs = _run_cmd_action(runtime, 'non_existing_command')
        assert obs.exit_code != 0, 'The exit code should not be 0 for a failed command.'
    finally:
        _close_test_runtime(runtime)


def _create_test_file(host_temp_dir):
    # Single file
    with open(os.path.join(host_temp_dir, 'test_file.txt'), 'w') as f:
        f.write('Hello, World!')


def test_copy_single_file(temp_dir, runtime_cls):
    runtime, config = _load_runtime(temp_dir, runtime_cls)
    try:
        sandbox_dir = config.workspace_mount_path_in_sandbox
        sandbox_file = os.path.join(sandbox_dir, 'test_file.txt')
        _create_test_file(temp_dir)
        runtime.copy_to(os.path.join(temp_dir, 'test_file.txt'), sandbox_dir)

        if is_windows():
            obs = _run_cmd_action(runtime, f'Get-ChildItem -Path {sandbox_dir}')
            assert obs.exit_code == 0
            assert 'test_file.txt' in obs.content

            obs = _run_cmd_action(runtime, f'Get-Content {sandbox_file}')
            assert obs.exit_code == 0
            assert 'Hello, World!' in obs.content
        else:
            obs = _run_cmd_action(runtime, f'ls -alh {sandbox_dir}')
            assert obs.exit_code == 0
            assert 'test_file.txt' in obs.content

            obs = _run_cmd_action(runtime, f'cat {sandbox_file}')
            assert obs.exit_code == 0
            assert 'Hello, World!' in obs.content
    finally:
        _close_test_runtime(runtime)


def _create_host_test_dir_with_files(test_dir):
    logger.debug(f'creating `{test_dir}`')
    if not os.path.isdir(test_dir):
        os.makedirs(test_dir, exist_ok=True)
    logger.debug('creating test files in `test_dir`')
    with open(os.path.join(test_dir, 'file1.txt'), 'w') as f:
        f.write('File 1 content')
    with open(os.path.join(test_dir, 'file2.txt'), 'w') as f:
        f.write('File 2 content')


def test_copy_directory_recursively(temp_dir, runtime_cls):
    runtime, config = _load_runtime(temp_dir, runtime_cls)

    sandbox_dir = config.workspace_mount_path_in_sandbox
    try:
        temp_dir_copy = os.path.join(temp_dir, 'test_dir')
        # We need a separate directory, since temp_dir is mounted to /workspace
        _create_host_test_dir_with_files(temp_dir_copy)

        runtime.copy_to(temp_dir_copy, sandbox_dir, recursive=True)

        if is_windows():
            obs = _run_cmd_action(runtime, f'Get-ChildItem -Path {sandbox_dir}')
            assert obs.exit_code == 0
            assert 'test_dir' in obs.content
            assert 'file1.txt' not in obs.content
            assert 'file2.txt' not in obs.content

            obs = _run_cmd_action(
                runtime, f'Get-ChildItem -Path {sandbox_dir}/test_dir'
            )
            assert obs.exit_code == 0
            assert 'file1.txt' in obs.content
            assert 'file2.txt' in obs.content

            obs = _run_cmd_action(
                runtime, f'Get-Content {sandbox_dir}/test_dir/file1.txt'
            )
            assert obs.exit_code == 0
            assert 'File 1 content' in obs.content
        else:
            obs = _run_cmd_action(runtime, f'ls -alh {sandbox_dir}')
            assert obs.exit_code == 0
            assert 'test_dir' in obs.content
            assert 'file1.txt' not in obs.content
            assert 'file2.txt' not in obs.content

            obs = _run_cmd_action(runtime, f'ls -alh {sandbox_dir}/test_dir')
            assert obs.exit_code == 0
            assert 'file1.txt' in obs.content
            assert 'file2.txt' in obs.content

            obs = _run_cmd_action(runtime, f'cat {sandbox_dir}/test_dir/file1.txt')
            assert obs.exit_code == 0
            assert 'File 1 content' in obs.content
    finally:
        _close_test_runtime(runtime)


def test_copy_to_non_existent_directory(temp_dir, runtime_cls):
    runtime, config = _load_runtime(temp_dir, runtime_cls)
    try:
        sandbox_dir = config.workspace_mount_path_in_sandbox
        _create_test_file(temp_dir)
        runtime.copy_to(
            os.path.join(temp_dir, 'test_file.txt'), f'{sandbox_dir}/new_dir'
        )

        obs = _run_cmd_action(runtime, f'cat {sandbox_dir}/new_dir/test_file.txt')
        assert obs.exit_code == 0
        assert 'Hello, World!' in obs.content
    finally:
        _close_test_runtime(runtime)


def test_overwrite_existing_file(tmp_path_factory, runtime_cls):
    temp_dir = tmp_path_factory.mktemp('mount')
    host_temp_dir = tmp_path_factory.mktemp('host')

    runtime, config = _load_runtime(temp_dir, runtime_cls)
    try:
        sandbox_dir = config.workspace_mount_path_in_sandbox
        sandbox_file = os.path.join(sandbox_dir, 'test_file.txt')

        if is_windows():
            # Check initial state
            obs = _run_cmd_action(runtime, f'Get-ChildItem -Path {sandbox_dir}')
            assert obs.exit_code == 0
            assert 'test_file.txt' not in obs.content

            # Create an empty file
            obs = _run_cmd_action(
                runtime, f'New-Item -ItemType File -Path {sandbox_file} -Force'
            )
            assert obs.exit_code == 0

            # Verify file exists and is empty
            obs = _run_cmd_action(runtime, f'Get-ChildItem -Path {sandbox_dir}')
            assert obs.exit_code == 0
            assert 'test_file.txt' in obs.content

            obs = _run_cmd_action(runtime, f'Get-Content {sandbox_file}')
            assert obs.exit_code == 0
            assert obs.content.strip() == ''  # Empty file
            assert 'Hello, World!' not in obs.content

            # Create host file and copy to overwrite
            _create_test_file(str(host_temp_dir))
            runtime.copy_to(str(host_temp_dir / 'test_file.txt'), sandbox_dir)

            # Verify file content is overwritten
            obs = _run_cmd_action(runtime, f'Get-Content {sandbox_file}')
            assert obs.exit_code == 0
            assert 'Hello, World!' in obs.content
        else:
            # Original Linux version
            obs = _run_cmd_action(runtime, f'ls -alh {sandbox_dir}')
            assert obs.exit_code == 0
            assert 'test_file.txt' not in obs.content  # Check initial state

            obs = _run_cmd_action(runtime, f'touch {sandbox_file}')
            assert obs.exit_code == 0

            obs = _run_cmd_action(runtime, f'ls -alh {sandbox_dir}')
            assert obs.exit_code == 0
            assert 'test_file.txt' in obs.content

            obs = _run_cmd_action(runtime, f'cat {sandbox_file}')
            assert obs.exit_code == 0
            assert obs.content.strip() == ''  # Empty file
            assert 'Hello, World!' not in obs.content

            _create_test_file(str(host_temp_dir))
            runtime.copy_to(str(host_temp_dir / 'test_file.txt'), sandbox_dir)

            obs = _run_cmd_action(runtime, f'cat {sandbox_file}')
            assert obs.exit_code == 0
            assert 'Hello, World!' in obs.content
    finally:
        _close_test_runtime(runtime)


def test_copy_non_existent_file(temp_dir, runtime_cls):
    runtime, config = _load_runtime(temp_dir, runtime_cls)
    try:
        sandbox_dir = config.workspace_mount_path_in_sandbox
        with pytest.raises(FileNotFoundError):
            runtime.copy_to(
                os.path.join(sandbox_dir, 'non_existent_file.txt'),
                f'{sandbox_dir}/should_not_exist.txt',
            )

        obs = _run_cmd_action(runtime, f'ls {sandbox_dir}/should_not_exist.txt')
        assert obs.exit_code != 0  # File should not exist
    finally:
        _close_test_runtime(runtime)


def test_copy_from_directory(temp_dir, runtime_cls):
    runtime, config = _load_runtime(temp_dir, runtime_cls)
    sandbox_dir = config.workspace_mount_path_in_sandbox
    try:
        temp_dir_copy = os.path.join(temp_dir, 'test_dir')
        # We need a separate directory, since temp_dir is mounted to /workspace
        _create_host_test_dir_with_files(temp_dir_copy)

        # Initial state
        runtime.copy_to(temp_dir_copy, sandbox_dir, recursive=True)

        path_to_copy_from = f'{sandbox_dir}/test_dir'
        result = runtime.copy_from(path=path_to_copy_from)

        # Result is returned as a path
        assert isinstance(result, Path)

        if result.exists() and not is_windows():
            result.unlink()
    finally:
        _close_test_runtime(runtime)


@pytest.mark.skipif(
    is_windows(), reason='Test uses Linux-specific file permissions and sudo commands'
)
def test_git_operation(temp_dir, runtime_cls):
    # do not mount workspace, since workspace mount by tests will be owned by root
    # while the user_id we get via os.getuid() is different from root
    # which causes permission issues
    runtime, config = _load_runtime(
        temp_dir=temp_dir,
        use_workspace=False,
        runtime_cls=runtime_cls,
        # Need to use non-root user to expose issues
        run_as_openhands=True,
    )
    # this will happen if permission of runtime is not properly configured
    # fatal: detected dubious ownership in repository at config.workspace_mount_path_in_sandbox
    try:
        if runtime_cls != LocalRuntime and runtime_cls != CLIRuntime:
            # on local machine, permissionless sudo will probably not be available
            obs = _run_cmd_action(runtime, 'sudo chown -R openhands:root .')
            assert obs.exit_code == 0

        # check the ownership of the current directory
        obs = _run_cmd_action(runtime, 'ls -alh .')
        assert obs.exit_code == 0
        # drwx--S--- 2 openhands root   64 Aug  7 23:32 .
        # drwxr-xr-x 1 root      root 4.0K Aug  7 23:33 ..
        for line in obs.content.split('\n'):
            if runtime_cls == LocalRuntime or runtime_cls == CLIRuntime:
                continue  # skip these checks

            if ' ..' in line:
                # parent directory should be owned by root
                assert 'root' in line
                assert 'openhands' not in line
            elif ' .' in line:
                # current directory should be owned by openhands
                # and its group should be root
                assert 'openhands' in line
                assert 'root' in line

        # make sure all git operations are allowed
        obs = _run_cmd_action(runtime, 'git init')
        assert obs.exit_code == 0

        # create a file
        obs = _run_cmd_action(runtime, 'echo "hello" > test_file.txt')
        assert obs.exit_code == 0

        if runtime_cls == LocalRuntime or runtime_cls == CLIRuntime:
            # set git config author in CI only, not on local machine
            logger.info('Setting git config author')
            obs = _run_cmd_action(
                runtime,
                'git config user.name "openhands" && git config user.email "openhands@all-hands.dev"',
            )
            assert obs.exit_code == 0

            # Set up git config - list current settings (should be empty or just what was set)
            obs = _run_cmd_action(runtime, 'git config --list')
            assert obs.exit_code == 0

        # git add
        obs = _run_cmd_action(runtime, 'git add test_file.txt')
        assert obs.exit_code == 0

        # git diff
        obs = _run_cmd_action(runtime, 'git diff --no-color --cached')
        assert obs.exit_code == 0
        assert 'b/test_file.txt' in obs.content
        assert '+hello' in obs.content

        # git commit
        obs = _run_cmd_action(runtime, 'git commit -m "test commit"')
        assert obs.exit_code == 0
    finally:
        _close_test_runtime(runtime)


def test_python_version(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        obs = runtime.run_action(CmdRunAction(command='python --version'))

        assert isinstance(obs, CmdOutputObservation), (
            'The observation should be a CmdOutputObservation.'
        )
        assert obs.exit_code == 0, 'The exit code should be 0.'
        assert 'Python 3' in obs.content, 'The output should contain "Python 3".'
    finally:
        _close_test_runtime(runtime)


def test_pwd_property(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        # Create a subdirectory and verify pwd updates
        obs = _run_cmd_action(runtime, 'mkdir -p random_dir')
        assert obs.exit_code == 0

        obs = _run_cmd_action(runtime, 'cd random_dir && pwd')
        assert obs.exit_code == 0
        assert 'random_dir' in obs.content
    finally:
        _close_test_runtime(runtime)


def test_basic_command(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        if is_windows():
            # Test simple command
            obs = _run_cmd_action(runtime, "Write-Output 'hello world'")
            assert 'hello world' in obs.content
            assert obs.exit_code == 0

            # Test command with error
            obs = _run_cmd_action(runtime, 'nonexistent_command')
            assert obs.exit_code != 0
            assert 'not recognized' in obs.content or 'command not found' in obs.content

            # Test command with special characters
            obs = _run_cmd_action(
                runtime, 'Write-Output "hello   world    with`nspecial  chars"'
            )
            assert 'hello   world    with\nspecial  chars' in obs.content
            assert obs.exit_code == 0

            # Test multiple commands in sequence
            obs = _run_cmd_action(
                runtime,
                'Write-Output "first" && Write-Output "second" && Write-Output "third"',
            )
            assert 'first' in obs.content
            assert 'second' in obs.content
            assert 'third' in obs.content
            assert obs.exit_code == 0
        else:
            # Original Linux version
            # Test simple command
            obs = _run_cmd_action(runtime, "echo 'hello world'")
            assert 'hello world' in obs.content
            assert obs.exit_code == 0

            # Test command with error
            obs = _run_cmd_action(runtime, 'nonexistent_command')
            assert obs.exit_code == 127
            assert 'nonexistent_command: command not found' in obs.content

            # Test command with special characters
            obs = _run_cmd_action(
                runtime, "echo 'hello   world    with\nspecial  chars'"
            )
            assert 'hello   world    with\nspecial  chars' in obs.content
            assert obs.exit_code == 0

            # Test multiple commands in sequence
            obs = _run_cmd_action(
                runtime, 'echo "first" && echo "second" && echo "third"'
            )
            assert 'first' in obs.content
            assert 'second' in obs.content
            assert 'third' in obs.content
            assert obs.exit_code == 0
    finally:
        _close_test_runtime(runtime)


@pytest.mark.skipif(
    is_windows(), reason='Powershell does not support interactive commands'
)
@pytest.mark.skipif(
    os.getenv('TEST_RUNTIME') == 'cli',
    reason='CLIRuntime does not support interactive commands from the agent.',
)
def test_interactive_command(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(
        temp_dir,
        runtime_cls,
        run_as_openhands,
        runtime_startup_env_vars={'NO_CHANGE_TIMEOUT_SECONDS': '1'},
    )
    try:
        # Test interactive command
        action = CmdRunAction('read -p "Enter name: " name && echo "Hello $name"')
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        # This should trigger SOFT timeout, so no need to set hard timeout
        assert 'Enter name:' in obs.content
        assert '[The command has no new output after 1 seconds.' in obs.metadata.suffix

        action = CmdRunAction('John', is_input=True)
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert 'Hello John' in obs.content
        assert '[The command completed with exit code 0.]' in obs.metadata.suffix

        # Test multiline command input with here document
        action = CmdRunAction("""cat << EOF
line 1
line 2
EOF""")
        obs = runtime.run_action(action)
        assert 'line 1\nline 2' in obs.content
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert '[The command completed with exit code 0.]' in obs.metadata.suffix
        assert obs.exit_code == 0
    finally:
        _close_test_runtime(runtime)


@pytest.mark.skipif(
    is_windows(),
    reason='Test relies on Linux-specific commands like seq and bash for loops',
)
def test_long_output(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        # Generate a long output
        action = CmdRunAction('for i in $(seq 1 5000); do echo "Line $i"; done')
        action.set_hard_timeout(10)
        obs = runtime.run_action(action)
        assert obs.exit_code == 0
        assert 'Line 1' in obs.content
        assert 'Line 5000' in obs.content
    finally:
        _close_test_runtime(runtime)


@pytest.mark.skipif(
    is_windows(),
    reason='Test relies on Linux-specific commands like seq and bash for loops',
)
@pytest.mark.skipif(
    os.getenv('TEST_RUNTIME') == 'cli',
    reason='CLIRuntime does not truncate command output.',
)
def test_long_output_exceed_history_limit(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        # Generate a long output
        action = CmdRunAction('for i in $(seq 1 50000); do echo "Line $i"; done')
        action.set_hard_timeout(30)
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert obs.exit_code == 0
        assert 'Previous command outputs are truncated' in obs.metadata.prefix
        assert 'Line 40000' in obs.content
        assert 'Line 50000' in obs.content
    finally:
        _close_test_runtime(runtime)


@pytest.mark.skipif(
    is_windows(), reason='Test uses Linux-specific temp directory and bash for loops'
)
def test_long_output_from_nested_directories(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        # Create nested directories with many files
        setup_cmd = 'mkdir -p /tmp/test_dir && cd /tmp/test_dir && for i in $(seq 1 100); do mkdir -p "folder_$i"; for j in $(seq 1 100); do touch "folder_$i/file_$j.txt"; done; done'
        setup_action = CmdRunAction(setup_cmd.strip())
        setup_action.set_hard_timeout(60)
        obs = runtime.run_action(setup_action)
        assert obs.exit_code == 0

        # List the directory structure recursively
        action = CmdRunAction('ls -R /tmp/test_dir')
        action.set_hard_timeout(60)
        obs = runtime.run_action(action)
        assert obs.exit_code == 0

        # Verify output contains expected files
        assert 'folder_1' in obs.content
        assert 'file_1.txt' in obs.content
        assert 'folder_100' in obs.content
        assert 'file_100.txt' in obs.content
    finally:
        _close_test_runtime(runtime)


@pytest.mark.skipif(
    is_windows(),
    reason='Test uses Linux-specific commands like find and grep with complex syntax',
)
def test_command_backslash(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        # Create a file with the content "implemented_function"
        action = CmdRunAction(
            'mkdir -p /tmp/test_dir && echo "implemented_function" > /tmp/test_dir/file_1.txt'
        )
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert obs.exit_code == 0

        # Reproduce an issue we ran into during evaluation
        # find /workspace/sympy__sympy__1.0 -type f -exec grep -l "implemented_function" {} \;
        # find: missing argument to `-exec'
        # --> This is unexpected output due to incorrect escaping of \;
        # This tests for correct escaping of \;
        action = CmdRunAction(
            'find /tmp/test_dir -type f -exec grep -l "implemented_function" {} \\;'
        )
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert obs.exit_code == 0
        assert '/tmp/test_dir/file_1.txt' in obs.content
    finally:
        _close_test_runtime(runtime)


@pytest.mark.skipif(
    is_windows(), reason='Test uses Linux-specific ps aux, awk, and grep commands'
)
@pytest.mark.skipif(
    os.getenv('TEST_RUNTIME') == 'cli',
    reason='CLIRuntime does not support interactive commands from the agent.',
)
def test_stress_long_output_with_soft_and_hard_timeout(
    temp_dir, runtime_cls, run_as_openhands
):
    runtime, config = _load_runtime(
        temp_dir,
        runtime_cls,
        run_as_openhands,
        runtime_startup_env_vars={'NO_CHANGE_TIMEOUT_SECONDS': '1'},
        docker_runtime_kwargs={
            'cpu_period': 100000,  # 100ms
            'cpu_quota': 100000,  # Can use 100ms out of each 100ms period (1 CPU)
            'mem_limit': '4G',  # 4 GB of memory
        },
    )
    try:
        # Run a command that generates long output multiple times
        for i in range(10):
            start_time = time.time()

            # Check tmux memory usage (in KB)
            mem_action = CmdRunAction(
                'ps aux | awk \'{printf "%8.1f KB  %s\\n", $6, $0}\' | sort -nr | grep "/usr/bin/tmux" | grep -v grep | awk \'{print $1}\''
            )
            mem_obs = runtime.run_action(mem_action)
            assert mem_obs.exit_code == 0
            logger.info(
                f'Tmux memory usage (iteration {i}): {mem_obs.content.strip()} KB'
            )

            # Check action_execution_server mem
            mem_action = CmdRunAction(
                'ps aux | awk \'{printf "%8.1f KB  %s\\n", $6, $0}\' | sort -nr | grep "action_execution_server" | grep "/openhands/poetry" | grep -v grep | awk \'{print $1}\''
            )
            mem_obs = runtime.run_action(mem_action)
            assert mem_obs.exit_code == 0
            logger.info(
                f'Action execution server memory usage (iteration {i}): {mem_obs.content.strip()} KB'
            )

            # Test soft timeout
            action = CmdRunAction(
                'read -p "Do you want to continue? [Y/n] " answer; if [[ $answer == "Y" ]]; then echo "Proceeding with operation..."; echo "Operation completed successfully!"; else echo "Operation cancelled."; exit 1; fi'
            )
            obs = runtime.run_action(action)
            assert 'Do you want to continue?' in obs.content
            assert obs.exit_code == -1  # Command is still running, waiting for input

            # Send the confirmation
            action = CmdRunAction('Y', is_input=True)
            obs = runtime.run_action(action)
            assert 'Proceeding with operation...' in obs.content
            assert 'Operation completed successfully!' in obs.content
            assert obs.exit_code == 0
            assert '[The command completed with exit code 0.]' in obs.metadata.suffix

            # Test hard timeout w/ long output
            # Generate long output with 1000 asterisks per line
            action = CmdRunAction(
                f'export i={i}; for j in $(seq 1 100); do echo "Line $j - Iteration $i - $(printf \'%1000s\' | tr " " "*")"; sleep 1; done'
            )
            action.set_hard_timeout(2)
            obs = runtime.run_action(action)

            # Verify the output
            assert obs.exit_code == -1
            assert f'Line 1 - Iteration {i}' in obs.content
            # assert f'Line 1000 - Iteration {i}' in obs.content
            # assert '[The command completed with exit code 0.]' in obs.metadata.suffix

            # Because hard-timeout is triggered, the terminal will in a weird state
            # where it will not accept any new commands.
            obs = runtime.run_action(CmdRunAction('ls'))
            assert obs.exit_code == -1
            assert 'The previous command is still running' in obs.metadata.suffix

            # We need to send a Ctrl+C to reset the terminal.
            obs = runtime.run_action(CmdRunAction('C-c', is_input=True))
            assert obs.exit_code == 130

            # Now make sure the terminal is in a good state
            obs = runtime.run_action(CmdRunAction('ls'))
            assert obs.exit_code == 0

            duration = time.time() - start_time
            logger.info(f'Completed iteration {i} in {duration:.2f} seconds')

    finally:
        _close_test_runtime(runtime)


@pytest.mark.skipif(
    os.getenv('TEST_RUNTIME') == 'cli',
    reason='FIXME: CLIRuntime does not watch previously timed-out commands except for getting full output a short time after timeout.',
)
def test_command_output_continuation(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        if is_windows():
            # Windows PowerShell version
            action = CmdRunAction(
                '1..5 | ForEach-Object { Write-Output $_; Start-Sleep 3 }'
            )
            action.set_hard_timeout(2.5)
            obs = runtime.run_action(action)
            assert obs.content.strip() == '1'
            assert obs.metadata.prefix == ''
            assert '[The command timed out after 2.5 seconds.' in obs.metadata.suffix

            # Continue watching output
            action = CmdRunAction('')
            action.set_hard_timeout(2.5)
            obs = runtime.run_action(action)
            assert (
                '[Below is the output of the previous command.]' in obs.metadata.prefix
            )
            assert obs.content.strip() == '2'
            assert '[The command timed out after 2.5 seconds.' in obs.metadata.suffix

            # Continue until completion
            for expected in ['3', '4', '5']:
                action = CmdRunAction('')
                action.set_hard_timeout(2.5)
                obs = runtime.run_action(action)
                assert (
                    '[Below is the output of the previous command.]'
                    in obs.metadata.prefix
                )
                assert obs.content.strip() == expected
                assert (
                    '[The command timed out after 2.5 seconds.' in obs.metadata.suffix
                )

            # Final empty command to complete
            action = CmdRunAction('')
            obs = runtime.run_action(action)
            assert '[The command completed with exit code 0.]' in obs.metadata.suffix
        else:
            # Original Linux version
            # Start a command that produces output slowly
            action = CmdRunAction('for i in {1..5}; do echo $i; sleep 3; done')
            action.set_hard_timeout(2.5)
            obs = runtime.run_action(action)
            assert obs.content.strip() == '1'
            assert obs.metadata.prefix == ''
            assert '[The command timed out after 2.5 seconds.' in obs.metadata.suffix

            # Continue watching output
            action = CmdRunAction('')
            action.set_hard_timeout(2.5)
            obs = runtime.run_action(action)
            assert (
                '[Below is the output of the previous command.]' in obs.metadata.prefix
            )
            assert obs.content.strip() == '2'
            assert '[The command timed out after 2.5 seconds.' in obs.metadata.suffix

            # Continue until completion
            for expected in ['3', '4', '5']:
                action = CmdRunAction('')
                action.set_hard_timeout(2.5)
                obs = runtime.run_action(action)
                assert (
                    '[Below is the output of the previous command.]'
                    in obs.metadata.prefix
                )
                assert obs.content.strip() == expected
                assert (
                    '[The command timed out after 2.5 seconds.' in obs.metadata.suffix
                )

            # Final empty command to complete
            action = CmdRunAction('')
            obs = runtime.run_action(action)
            assert '[The command completed with exit code 0.]' in obs.metadata.suffix
    finally:
        _close_test_runtime(runtime)


@pytest.mark.skipif(
    os.getenv('TEST_RUNTIME') == 'cli',
    reason='FIXME: CLIRuntime does not implement empty command behavior.',
)
def test_long_running_command_follow_by_execute(
    temp_dir, runtime_cls, run_as_openhands
):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        if is_windows():
            action = CmdRunAction('1..3 | ForEach-Object { Write-Output $_; sleep 3 }')
        else:
            # Test command that produces output slowly
            action = CmdRunAction('for i in {1..3}; do echo $i; sleep 3; done')

        action.set_hard_timeout(2.5)
        obs = runtime.run_action(action)
        assert '1' in obs.content  # First number should appear before timeout
        assert obs.metadata.exit_code == -1  # -1 indicates command is still running
        assert '[The command timed out after 2.5 seconds.' in obs.metadata.suffix
        assert obs.metadata.prefix == ''

        # Continue watching output
        action = CmdRunAction('')
        action.set_hard_timeout(2.5)
        obs = runtime.run_action(action)
        assert '2' in obs.content
        assert obs.metadata.prefix == '[Below is the output of the previous command.]\n'
        assert '[The command timed out after 2.5 seconds.' in obs.metadata.suffix
        assert obs.metadata.exit_code == -1  # -1 indicates command is still running

        # Test command that produces no output
        action = CmdRunAction('sleep 15')
        action.set_hard_timeout(2.5)
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert '3' not in obs.content
        assert obs.metadata.prefix == '[Below is the output of the previous command.]\n'
        assert 'The previous command is still running' in obs.metadata.suffix
        assert obs.metadata.exit_code == -1  # -1 indicates command is still running

        # Finally continue again
        action = CmdRunAction('')
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert '3' in obs.content
        assert '[The command completed with exit code 0.]' in obs.metadata.suffix
    finally:
        _close_test_runtime(runtime)


@pytest.mark.skipif(
    os.getenv('TEST_RUNTIME') == 'cli',
    reason='FIXME: CLIRuntime does not implement empty command behavior.',
)
def test_empty_command_errors(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        # Test empty command without previous command - behavior should be the same on all platforms
        obs = runtime.run_action(CmdRunAction(''))
        assert isinstance(obs, CmdOutputObservation)
        assert (
            'ERROR: No previous running command to retrieve logs from.' in obs.content
        )
    finally:
        _close_test_runtime(runtime)


@pytest.mark.skipif(
    is_windows(), reason='Powershell does not support interactive commands'
)
@pytest.mark.skipif(
    os.getenv('TEST_RUNTIME') == 'cli',
    reason='CLIRuntime does not support interactive commands from the agent.',
)
def test_python_interactive_input(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        # Test Python program that asks for input - same for both platforms
        python_script = """name = input('Enter your name: '); age = input('Enter your age: '); print(f'Hello {name}, you are {age} years old')"""

        # Start Python with the interactive script
        # For both platforms we can use the same command
        obs = runtime.run_action(CmdRunAction(f'python -c "{python_script}"'))
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert 'Enter your name:' in obs.content
        assert obs.metadata.exit_code == -1  # -1 indicates command is still running

        # Send first input (name)
        obs = runtime.run_action(CmdRunAction('Alice', is_input=True))
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert 'Enter your age:' in obs.content
        assert obs.metadata.exit_code == -1

        # Send second input (age)
        obs = runtime.run_action(CmdRunAction('25', is_input=True))
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert 'Hello Alice, you are 25 years old' in obs.content
        assert obs.metadata.exit_code == 0
        assert '[The command completed with exit code 0.]' in obs.metadata.suffix
    finally:
        _close_test_runtime(runtime)


@pytest.mark.skipif(
    is_windows(), reason='Powershell does not support interactive commands'
)
@pytest.mark.skipif(
    os.getenv('TEST_RUNTIME') == 'cli',
    reason='CLIRuntime does not support interactive commands from the agent.',
)
def test_python_interactive_input_without_set_input(
    temp_dir, runtime_cls, run_as_openhands
):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        # Test Python program that asks for input
        python_script = """name = input('Enter your name: '); age = input('Enter your age: '); print(f'Hello {name}, you are {age} years old')"""

        # Start Python with the interactive script
        obs = runtime.run_action(CmdRunAction(f'python -c "{python_script}"'))
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert 'Enter your name:' in obs.content
        assert obs.metadata.exit_code == -1  # -1 indicates command is still running

        # Send first input (name)
        obs = runtime.run_action(CmdRunAction('Alice', is_input=False))
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert 'Enter your age:' not in obs.content
        assert (
            'Your command "Alice" is NOT executed. The previous command is still running'
            in obs.metadata.suffix
        )
        assert obs.metadata.exit_code == -1

        # Try again now with input
        obs = runtime.run_action(CmdRunAction('Alice', is_input=True))
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert 'Enter your age:' in obs.content
        assert obs.metadata.exit_code == -1

        obs = runtime.run_action(CmdRunAction('25', is_input=True))
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert 'Hello Alice, you are 25 years old' in obs.content
        assert obs.metadata.exit_code == 0
        assert '[The command completed with exit code 0.]' in obs.metadata.suffix
    finally:
        _close_test_runtime(runtime)


def test_bash_remove_prefix(temp_dir, runtime_cls, run_as_openhands):
    runtime, config = _load_runtime(temp_dir, runtime_cls, run_as_openhands)
    try:
        # create a git repo - same for both platforms
        action = CmdRunAction(
            'git init && git remote add origin https://github.com/OpenHands/OpenHands'
        )
        obs = runtime.run_action(action)
        # logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert obs.metadata.exit_code == 0

        # Check git remote - same for both platforms
        obs = runtime.run_action(CmdRunAction('git remote -v'))
        # logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert obs.metadata.exit_code == 0
        assert 'https://github.com/OpenHands/OpenHands' in obs.content
        assert 'git remote -v' not in obs.content
    finally:
        _close_test_runtime(runtime)
