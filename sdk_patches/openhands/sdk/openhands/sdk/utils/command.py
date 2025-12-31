import shlex
import subprocess
import sys
import threading

from openhands.sdk.logger import get_logger


logger = get_logger(__name__)


def execute_command(
    cmd: list[str] | str,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    timeout: float | None = None,
    print_output: bool = True,
) -> subprocess.CompletedProcess:
    # For string commands, use shell=True to handle shell operators properly
    if isinstance(cmd, str):
        cmd_to_run = cmd
        use_shell = True
        logger.info("$ %s", cmd)
    else:
        cmd_to_run = cmd
        use_shell = False
        logger.info("$ %s", " ".join(shlex.quote(c) for c in cmd))

    proc = subprocess.Popen(
        cmd_to_run,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        shell=use_shell,
    )
    if proc is None:
        raise RuntimeError("Failed to start process")

    # Read line by line, echo to parent stdout/stderr
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    if proc.stdout is None or proc.stderr is None:
        raise RuntimeError("Failed to capture stdout/stderr")

    def read_stream(stream, lines, output_stream):
        try:
            for line in stream:
                if print_output:
                    output_stream.write(line)
                    output_stream.flush()
                lines.append(line)
        except Exception as e:
            logger.error(f"Failed to read stream: {e}")

    # Read stdout and stderr concurrently to avoid deadlock
    stdout_thread = threading.Thread(
        target=read_stream, args=(proc.stdout, stdout_lines, sys.stdout)
    )
    stderr_thread = threading.Thread(
        target=read_stream, args=(proc.stderr, stderr_lines, sys.stderr)
    )

    stdout_thread.start()
    stderr_thread.start()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout_thread.join()
        stderr_thread.join()
        return subprocess.CompletedProcess(
            cmd_to_run,
            -1,  # Indicate timeout with -1 exit code
            "".join(stdout_lines),
            "".join(stderr_lines),
        )

    stdout_thread.join(timeout=timeout)
    stderr_thread.join(timeout=timeout)

    return subprocess.CompletedProcess(
        cmd_to_run,
        proc.returncode,
        "".join(stdout_lines),
        "".join(stderr_lines),
    )
