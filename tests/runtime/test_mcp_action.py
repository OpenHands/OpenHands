"""Bash-related tests for the DockerRuntime, which connects to the ActionExecutor running in the sandbox."""

import json
import os
import socket
import time

import docker
import pytest
from conftest import (
    _load_runtime,
)

import openhands
from openhands.core.config import MCPConfig
from openhands.core.config.mcp_config import MCPSSEServerConfig, MCPStdioServerConfig
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import CmdRunAction, MCPAction
from openhands.events.observation import CmdOutputObservation, MCPObservation

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


@pytest.mark.skip('This test is flaky')
def test_default_activated_tools():
    import importlib.resources

    # Use importlib.resources to access the config file properly
    # This works both when running from source and from installed package
    try:
        with importlib.resources.as_file(
            importlib.resources.files('openhands').joinpath(
                'runtime', 'mcp', 'config.json'
            )
        ) as config_path:
            assert config_path.exists(), f'MCP config file not found at {config_path}'
            with open(config_path, 'r') as f:
                mcp_config = json.load(f)
    except (FileNotFoundError, ImportError):
        # Fallback to the old method for development environments
        project_root = os.path.dirname(openhands.__file__)
        mcp_config_path = os.path.join(project_root, 'runtime', 'mcp', 'config.json')
        assert os.path.exists(mcp_config_path), (
            f'MCP config file not found at {mcp_config_path}'
        )
        with open(mcp_config_path, 'r') as f:
            mcp_config = json.load(f)

    assert 'mcpServers' in mcp_config
    assert 'default' in mcp_config['mcpServers']
    assert 'tools' in mcp_config
    # no tools are always activated yet
    assert len(mcp_config['tools']) == 0


@pytest.mark.skip('This test is flaky')
@pytest.mark.asyncio
async def test_fetch_mcp_via_stdio(
    temp_dir, runtime_cls, run_as_openhands, dynamic_port
):
    mcp_stdio_server_config = MCPStdioServerConfig(
        name='fetch', command='uvx', args=['mcp-server-fetch']
    )
    override_mcp_config = MCPConfig(stdio_servers=[mcp_stdio_server_config])
    runtime, config = _load_runtime(
        temp_dir,
        runtime_cls,
        run_as_openhands,
        override_mcp_config=override_mcp_config,
        enable_browser=True,
    )

    # Test browser server
    action_cmd = CmdRunAction(
        command=f'python3 -m http.server {dynamic_port} > server.log 2>&1 &'
    )
    logger.info(action_cmd, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action_cmd)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code == 0
    assert '[1]' in obs.content

    action_cmd = CmdRunAction(command='sleep 3 && cat server.log')
    logger.info(action_cmd, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action_cmd)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert obs.exit_code == 0

    mcp_action = MCPAction(
        name='fetch', arguments={'url': f'http://localhost:{dynamic_port}'}
    )
    obs = await runtime.call_tool_mcp(mcp_action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, MCPObservation), (
        'The observation should be a MCPObservation.'
    )

    result_json = json.loads(obs.content)
    assert not result_json['isError']
    assert len(result_json['content']) == 1
    assert result_json['content'][0]['type'] == 'text'
    assert (
        result_json['content'][0]['text']
        == f'Contents of http://localhost:{dynamic_port}/:\n---\n\n* <.downloads/>\n* <server.log>\n\n---'
    )

    runtime.close()


@pytest.mark.skip('This test is flaky')
@pytest.mark.asyncio
async def test_filesystem_mcp_via_sse(
    temp_dir, runtime_cls, run_as_openhands, sse_mcp_docker_server
):
    sse_server_info = sse_mcp_docker_server
    sse_url = sse_server_info['url']
    runtime = None
    try:
        mcp_sse_server_config = MCPSSEServerConfig(url=sse_url)
        override_mcp_config = MCPConfig(sse_servers=[mcp_sse_server_config])
        runtime, config = _load_runtime(
            temp_dir,
            runtime_cls,
            run_as_openhands,
            override_mcp_config=override_mcp_config,
        )

        mcp_action = MCPAction(name='list_directory', arguments={'path': '.'})
        obs = await runtime.call_tool_mcp(mcp_action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert isinstance(obs, MCPObservation), (
            'The observation should be a MCPObservation.'
        )
        assert '[FILE] .dockerenv' in obs.content

    finally:
        if runtime:
            runtime.close()
        # Container and log_streamer cleanup is handled by the sse_mcp_docker_server fixture


@pytest.mark.skip('This test is flaky')
@pytest.mark.asyncio
async def test_both_stdio_and_sse_mcp(
    temp_dir, runtime_cls, run_as_openhands, sse_mcp_docker_server, dynamic_port
):
    sse_server_info = sse_mcp_docker_server
    sse_url = sse_server_info['url']
    runtime = None
    try:
        mcp_sse_server_config = MCPSSEServerConfig(url=sse_url)

        # Also add stdio server
        mcp_stdio_server_config = MCPStdioServerConfig(
            name='fetch', command='uvx', args=['mcp-server-fetch']
        )

        override_mcp_config = MCPConfig(
            sse_servers=[mcp_sse_server_config], stdio_servers=[mcp_stdio_server_config]
        )
        runtime, config = _load_runtime(
            temp_dir,
            runtime_cls,
            run_as_openhands,
            override_mcp_config=override_mcp_config,
            enable_browser=True,
        )

        # ======= Test SSE server =======
        mcp_action_sse = MCPAction(name='list_directory', arguments={'path': '.'})
        obs_sse = await runtime.call_tool_mcp(mcp_action_sse)
        logger.info(obs_sse, extra={'msg_type': 'OBSERVATION'})
        assert isinstance(obs_sse, MCPObservation), (
            'The observation should be a MCPObservation.'
        )
        assert '[FILE] .dockerenv' in obs_sse.content

        # ======= Test stdio server =======
        # Test browser server
        action_cmd_http = CmdRunAction(
            command=f'python3 -m http.server {dynamic_port} > server.log 2>&1 &'
        )
        logger.info(action_cmd_http, extra={'msg_type': 'ACTION'})
        obs_http = runtime.run_action(action_cmd_http)
        logger.info(obs_http, extra={'msg_type': 'OBSERVATION'})

        assert isinstance(obs_http, CmdOutputObservation)
        assert obs_http.exit_code == 0
        assert '[1]' in obs_http.content

        action_cmd_cat = CmdRunAction(command='sleep 3 && cat server.log')
        logger.info(action_cmd_cat, extra={'msg_type': 'ACTION'})
        obs_cat = runtime.run_action(action_cmd_cat)
        logger.info(obs_cat, extra={'msg_type': 'OBSERVATION'})
        assert obs_cat.exit_code == 0

        mcp_action_fetch = MCPAction(
            # NOTE: the tool name is `fetch_fetch` because the tool name is `fetch`
            # And FastMCP Proxy will pre-pend the server name (in this case, `fetch`)
            # to the tool name, so the full tool name becomes `fetch_fetch`
            name='fetch',
            arguments={'url': f'http://localhost:{dynamic_port}'},
        )
        obs_fetch = await runtime.call_tool_mcp(mcp_action_fetch)
        logger.info(obs_fetch, extra={'msg_type': 'OBSERVATION'})
        assert isinstance(obs_fetch, MCPObservation), (
            'The observation should be a MCPObservation.'
        )

        result_json = json.loads(obs_fetch.content)
        assert not result_json['isError']
        assert len(result_json['content']) == 1
        assert result_json['content'][0]['type'] == 'text'
        assert (
            result_json['content'][0]['text']
            == f'Contents of http://localhost:{dynamic_port}/:\n---\n\n* <.downloads/>\n* <server.log>\n\n---'
        )
    finally:
        if runtime:
            runtime.close()
        # SSE Docker container cleanup is handled by the sse_mcp_docker_server fixture


@pytest.mark.skip('This test is flaky')
@pytest.mark.asyncio
async def test_microagent_and_one_stdio_mcp_in_config(
    temp_dir, runtime_cls, run_as_openhands, dynamic_port
):
    runtime = None
    try:
        filesystem_config = MCPStdioServerConfig(
            name='filesystem',
            command='npx',
            args=[
                '@modelcontextprotocol/server-filesystem@2025.8.18',
                '/',
            ],
        )
        override_mcp_config = MCPConfig(stdio_servers=[filesystem_config])
        runtime, config = _load_runtime(
            temp_dir,
            runtime_cls,
            run_as_openhands,
            override_mcp_config=override_mcp_config,
            enable_browser=True,
        )

        # NOTE: this simulate the case where the microagent adds a new stdio server to the runtime
        # but that stdio server is not in the initial config
        # Actual invocation of the microagent involves `add_mcp_tools_to_agent`
        # which will call `get_mcp_config` with the stdio server from microagent's config
        fetch_config = MCPStdioServerConfig(
            name='fetch', command='uvx', args=['mcp-server-fetch']
        )
        updated_config = runtime.get_mcp_config([fetch_config])
        logger.info(f'updated_config: {updated_config}')

        # ======= Test the stdio server in the config =======
        mcp_action_sse = MCPAction(
            name='filesystem_list_directory', arguments={'path': '/'}
        )
        obs_sse = await runtime.call_tool_mcp(mcp_action_sse)
        logger.info(obs_sse, extra={'msg_type': 'OBSERVATION'})
        assert isinstance(obs_sse, MCPObservation), (
            'The observation should be a MCPObservation.'
        )
        assert '[FILE] .dockerenv' in obs_sse.content

        # ======= Test the stdio server added by the microagent =======
        # Test browser server
        action_cmd_http = CmdRunAction(
            command=f'python3 -m http.server {dynamic_port} > server.log 2>&1 &'
        )
        logger.info(action_cmd_http, extra={'msg_type': 'ACTION'})
        obs_http = runtime.run_action(action_cmd_http)
        logger.info(obs_http, extra={'msg_type': 'OBSERVATION'})

        assert isinstance(obs_http, CmdOutputObservation)
        assert obs_http.exit_code == 0
        assert '[1]' in obs_http.content

        action_cmd_cat = CmdRunAction(command='sleep 3 && cat server.log')
        logger.info(action_cmd_cat, extra={'msg_type': 'ACTION'})
        obs_cat = runtime.run_action(action_cmd_cat)
        logger.info(obs_cat, extra={'msg_type': 'OBSERVATION'})
        assert obs_cat.exit_code == 0

        mcp_action_fetch = MCPAction(
            name='fetch_fetch', arguments={'url': f'http://localhost:{dynamic_port}'}
        )
        obs_fetch = await runtime.call_tool_mcp(mcp_action_fetch)
        logger.info(obs_fetch, extra={'msg_type': 'OBSERVATION'})
        assert isinstance(obs_fetch, MCPObservation), (
            'The observation should be a MCPObservation.'
        )

        result_json = json.loads(obs_fetch.content)
        assert not result_json['isError']
        assert len(result_json['content']) == 1
        assert result_json['content'][0]['type'] == 'text'
        assert (
            result_json['content'][0]['text']
            == f'Contents of http://localhost:{dynamic_port}/:\n---\n\n* <.downloads/>\n* <server.log>\n\n---'
        )
    finally:
        if runtime:
            runtime.close()
        # SSE Docker container cleanup is handled by the sse_mcp_docker_server fixture
