"""Bash-related tests for the DockerRuntime, which connects to the ActionExecutor running in the sandbox."""

import os
import socket
import time

import docker
import pytest

from openhands.core.logger import openhands_logger as logger

# ============================================================================================================================
# Bash-specific tests
# ============================================================================================================================

pytestmark = pytest.mark.skipif(
    os.environ.get('TEST_RUNTIME') == 'cli',
    reason='CLIRuntime does not support MCP actions',
)


@pytest.fixture
def sse_mcp_docker_server():
    """Manages the lifecycle of the SSE MCP Docker container for tests, using a random available port."""
    image_name = 'supercorp/supergateway'

    # Find a free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        host_port = s.getsockname()[1]

    container_internal_port = (
        8080  # The port the MCP server listens on *inside* the container
    )

    container_command_args = [
        '--stdio',
        'npx -y @modelcontextprotocol/server-filesystem@2025.8.18 /',
        '--port',
        str(container_internal_port),  # MCP server inside container listens on this
        '--baseUrl',
        f'http://localhost:{host_port}',  # The URL used to access the server from the host
    ]
    client = docker.from_env()
    container = None
    log_streamer = None

    # Import LogStreamer here as it's specific to this fixture's needs
    from openhands.runtime.utils.log_streamer import LogStreamer

    try:
        logger.info(
            f'Starting Docker container {image_name} with command: {" ".join(container_command_args)} '
            f'and mapping internal port {container_internal_port} to host port {host_port}',
            extra={'msg_type': 'ACTION'},
        )
        container = client.containers.run(
            image_name,
            command=container_command_args,
            ports={
                f'{container_internal_port}/tcp': host_port
            },  # Map container's internal port to the random host port
            detach=True,
            auto_remove=True,
            stdin_open=True,
        )
        logger.info(
            f'Container {container.short_id} started, listening on host port {host_port}.'
        )

        log_streamer = LogStreamer(
            container,
            lambda level, msg: getattr(logger, level.lower())(
                f'[MCP server {container.short_id}] {msg}'
            ),
        )
        # Wait for the server to initialize, as in the original tests
        time.sleep(10)

        yield {'url': f'http://localhost:{host_port}/sse'}

    finally:
        if container:
            logger.info(f'Stopping container {container.short_id}...')
            try:
                container.stop(timeout=5)
                logger.info(
                    f'Container {container.short_id} stopped (and should be auto-removed).'
                )
            except docker.errors.NotFound:
                logger.info(
                    f'Container {container.short_id} not found, likely already stopped and removed.'
                )
            except Exception as e:
                logger.error(f'Error stopping container {container.short_id}: {e}')
        if log_streamer:
            log_streamer.close()
