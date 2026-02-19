# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
import threading
from typing import Callable

import docker


class LogStreamer:
    """Streams Docker container logs to stdout.

    This class provides a way to stream logs from a Docker container directly to stdout
    through the provided logging function.
    """

    def __init__(
        self,
        container: docker.models.containers.Container,
        logFn: Callable[[str, str], None],
    ):
        self.log = logFn
        # Initialize all attributes before starting the thread on this instance
        self.stdout_thread = None
        self.log_generator = None
        self._stop_event = threading.Event()

        try:
            self.log_generator = container.logs(stream=True, follow=True)
            # Start the stdout streaming thread
            self.stdout_thread = threading.Thread(target=self._stream_logs)
            self.stdout_thread.daemon = True
            self.stdout_thread.start()
        except Exception as e:
            self.log('error', f'Failed to initialize log streaming: {e}')

    def _stream_logs(self) -> None:
        """Stream logs from the Docker container to stdout."""
        if not self.log_generator:
            self.log('error', 'Log generator not initialized')
            return

        try:
            for log_line in self.log_generator:
                if self._stop_event.is_set():
                    break
                if log_line:
                    decoded_line = log_line.decode('utf-8').rstrip()
                    self.log('debug', f'[inside container] {decoded_line}')
        except Exception as e:
            self.log('error', f'Error streaming docker logs to stdout: {e}')

    def __del__(self) -> None:
        if (
            hasattr(self, 'stdout_thread')
            and self.stdout_thread
            and self.stdout_thread.is_alive()
        ):
            self.close(timeout=5)

    def close(self, timeout: float = 5.0) -> None:
        """Clean shutdown of the log streaming."""
        self._stop_event.set()
        if self.stdout_thread and self.stdout_thread.is_alive():
            self.stdout_thread.join(timeout)
        # Close the log generator to release the file descriptor
        if self.log_generator is not None:
            self.log_generator.close()
