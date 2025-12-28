import os
import random
import shutil
import stat
import time

import pytest
from pytest import TempPathFactory

from openhands.core.config import MCPConfig, OpenHandsConfig, load_openhands_config
from openhands.core.logger import openhands_logger as logger
from openhands.events import EventStream
from openhands.llm.llm_registry import LLMRegistry
from openhands.runtime.base import Runtime
from openhands.runtime.impl.cli.cli_runtime import CLIRuntime
from openhands.runtime.impl.docker.docker_runtime import DockerRuntime
from openhands.runtime.impl.local.local_runtime import LocalRuntime
from openhands.runtime.impl.remote.remote_runtime import RemoteRuntime
from openhands.runtime.plugins import AgentSkillsRequirement, JupyterRequirement
from openhands.runtime.utils.port_lock import find_available_port_with_lock
from openhands.storage import get_file_store
from openhands.utils.async_utils import call_async_from_sync

TEST_IN_CI = os.getenv('TEST_IN_CI', 'False').lower() in ['true', '1', 'yes']
TEST_RUNTIME = os.getenv('TEST_RUNTIME', 'docker').lower()
RUN_AS_OPENHANDS = os.getenv('RUN_AS_OPENHANDS', 'True').lower() in ['true', '1', 'yes']
test_mount_path = ''
project_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sandbox_test_folder = '/workspace'


def _get_runtime_sid(runtime: Runtime) -> str:
    logger.debug(f'\nruntime.sid: {runtime.sid}')
    return runtime.sid


def _get_host_folder(runtime: Runtime) -> str:
    return runtime.config.workspace_mount_path


def _remove_folder(folder: str) -> bool:
    success = False
    if folder and os.path.isdir(folder):
        try:
            os.rmdir(folder)
            success = True
        except OSError:
            try:
                shutil.rmtree(folder)
                success = True
            except OSError:
                pass
        logger.debug(f'\nCleanup: `{folder}`: ' + ('[OK]' if success else '[FAILED]'))
    return success


def _close_test_runtime(runtime: Runtime) -> None:
    if isinstance(runtime, DockerRuntime):
        runtime.close(rm_all_containers=False)
    else:
        runtime.close()
    time.sleep(1)


def _reset_cwd() -> None:
    global project_dir
    # Try to change back to project directory
    try:
        os.chdir(project_dir)
        logger.info(f'Changed back to project directory `{project_dir}')
    except Exception as e:
        logger.error(f'Failed to change back to project directory: {e}')


# *****************************************************************************
# *****************************************************************************


@pytest.fixture(autouse=True)
def print_method_name(request):
    print(
        '\n\n########################################################################'
    )
    print(f'Running test: {request.node.name}')
    print(
        '########################################################################\n\n'
    )


@pytest.fixture
def temp_dir(tmp_path_factory: TempPathFactory, request) -> str:
    """Creates a unique temporary directory.

    Upon finalization, the temporary directory and its content is removed.
    The cleanup function is also called upon KeyboardInterrupt.

    Parameters:
    - tmp_path_factory (TempPathFactory): A TempPathFactory class

    Returns:
    - str: The temporary directory path that was created
    """
    temp_dir = tmp_path_factory.mktemp(
        'rt_' + str(random.randint(100000, 999999)), numbered=False
    )

    logger.info(f'\n*** {request.node.name}\n>> temp folder: {temp_dir}\n')

    # Set permissions to ensure the directory is writable and deletable
    os.chmod(temp_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # 0777 permissions

    def cleanup():
        global project_dir
        os.chdir(project_dir)
        _remove_folder(temp_dir)

    request.addfinalizer(cleanup)

    return str(temp_dir)


# Depending on TEST_RUNTIME, feed the appropriate box class(es) to the test.
def get_runtime_classes() -> list[type[Runtime]]:
    runtime = TEST_RUNTIME
    if runtime.lower() == 'docker' or runtime.lower() == 'eventstream':
        return [DockerRuntime]
    elif runtime.lower() == 'local':
        return [LocalRuntime]
    elif runtime.lower() == 'remote':
        return [RemoteRuntime]
    elif runtime.lower() == 'cli':
        return [CLIRuntime]
    else:
        raise ValueError(f'Invalid runtime: {runtime}')


def get_run_as_openhands() -> list[bool]:
    print(
        '\n\n########################################################################'
    )
    print('USER: ' + 'openhands' if RUN_AS_OPENHANDS else 'root')
    print(
        '########################################################################\n\n'
    )
    return [RUN_AS_OPENHANDS]


@pytest.fixture(scope='module')  # for xdist
def runtime_setup_module():
    _reset_cwd()
    yield
    _reset_cwd()


@pytest.fixture(scope='session')  # not for xdist
def runtime_setup_session():
    _reset_cwd()
    yield
    _reset_cwd()


# This assures that all tests run together per runtime, not alternating between them,
# which cause errors (especially outside GitHub actions).
@pytest.fixture(scope='module', params=get_runtime_classes())
def runtime_cls(request):
    time.sleep(1)
    return request.param


# TODO: We will change this to `run_as_user` when `ServerRuntime` is deprecated.
# since `DockerRuntime` supports running as an arbitrary user.
@pytest.fixture(scope='module', params=get_run_as_openhands())
def run_as_openhands(request):
    time.sleep(1)
    return request.param


@pytest.fixture(scope='module', params=None)
def base_container_image(request):
    time.sleep(1)
    env_image = os.environ.get('SANDBOX_BASE_CONTAINER_IMAGE')
    if env_image:
        request.param = env_image
    else:
        if not hasattr(request, 'param'):  # prevent runtime AttributeError
            request.param = None
        if request.param is None and hasattr(request.config, 'sandbox'):
            try:
                request.param = request.config.sandbox.getoption(
                    '--base_container_image'
                )
            except ValueError:
                request.param = None
        if request.param is None:
            request.param = pytest.param(
                'nikolaik/python-nodejs:python3.12-nodejs22',
                'golang:1.23-bookworm',
            )
    print(f'Container image: {request.param}')
    return request.param


def _load_runtime(
    temp_dir: str | None,
    runtime_cls: str,
    run_as_openhands: bool = True,
    enable_auto_lint: bool = False,
    base_container_image: str | None = None,
    browsergym_eval_env: str | None = None,
    use_workspace: bool | None = None,
    force_rebuild_runtime: bool = False,
    runtime_startup_env_vars: dict[str, str] | None = None,
    docker_runtime_kwargs: dict[str, str] | None = None,
    override_mcp_config: MCPConfig | None = None,
    enable_browser: bool = False,
) -> tuple[Runtime, OpenHandsConfig]:
    sid = 'rt_' + str(random.randint(100000, 999999))

    # AgentSkills need to be initialized **before** Jupyter
    # otherwise Jupyter will not access the proper dependencies installed by AgentSkills
    plugins = [AgentSkillsRequirement(), JupyterRequirement()]

    config = load_openhands_config()
    config.run_as_openhands = run_as_openhands
    config.enable_browser = enable_browser
    config.sandbox.force_rebuild_runtime = force_rebuild_runtime
    config.sandbox.keep_runtime_alive = False
    config.sandbox.docker_runtime_kwargs = docker_runtime_kwargs
    # Folder where all tests create their own folder
    global test_mount_path
    if use_workspace:
        test_mount_path = os.path.join(config.workspace_base, 'rt')
    elif temp_dir is not None:
        test_mount_path = str(temp_dir)
    else:
        test_mount_path = None
    config.workspace_base = test_mount_path
    config.workspace_mount_path = test_mount_path

    # Mounting folder specific for this test inside the sandbox
    config.workspace_mount_path_in_sandbox = f'{sandbox_test_folder}'
    print('\nPaths used:')
    print(f'use_host_network: {config.sandbox.use_host_network}')
    print(f'workspace_base: {config.workspace_base}')
    print(f'workspace_mount_path: {config.workspace_mount_path}')
    print(
        f'workspace_mount_path_in_sandbox: {config.workspace_mount_path_in_sandbox}\n'
    )

    config.sandbox.browsergym_eval_env = browsergym_eval_env
    config.sandbox.enable_auto_lint = enable_auto_lint
    if runtime_startup_env_vars is not None:
        config.sandbox.runtime_startup_env_vars = runtime_startup_env_vars

    if base_container_image is not None:
        config.sandbox.base_container_image = base_container_image
        config.sandbox.runtime_container_image = None

    if override_mcp_config is not None:
        config.mcp = override_mcp_config

    file_store = file_store = get_file_store(
        file_store_type=config.file_store,
        file_store_path=config.file_store_path,
        file_store_web_hook_url=config.file_store_web_hook_url,
        file_store_web_hook_headers=config.file_store_web_hook_headers,
        file_store_web_hook_batch=config.file_store_web_hook_batch,
    )
    event_stream = EventStream(sid, file_store)

    # Create a LLMRegistry instance for the runtime
    llm_registry = LLMRegistry(config=OpenHandsConfig())

    runtime = runtime_cls(
        config=config,
        event_stream=event_stream,
        llm_registry=llm_registry,
        sid=sid,
        plugins=plugins,
    )

    # For CLIRuntime, the tests' assertions should be based on the physical workspace path,
    # not the logical "/workspace". So, we adjust config.workspace_mount_path_in_sandbox
    # to reflect the actual physical path used by CLIRuntime's OHEditor.
    if isinstance(runtime, CLIRuntime):
        config.workspace_mount_path_in_sandbox = str(runtime.workspace_root)
        logger.info(
            f'Adjusted workspace_mount_path_in_sandbox for CLIRuntime to: {config.workspace_mount_path_in_sandbox}'
        )

    call_async_from_sync(runtime.connect)
    time.sleep(2)
    return runtime, runtime.config


# Port range for test HTTP servers (separate from runtime ports to avoid conflicts)
TEST_HTTP_SERVER_PORT_RANGE = (18000, 18999)


@pytest.fixture
def dynamic_port(request):
    """Allocate a dynamic port with locking to prevent race conditions in parallel tests.

    This fixture uses the existing port locking system to ensure that parallel test
    workers don't try to use the same port for HTTP servers.

    Returns:
        int: An available port number that is locked for this test
    """
    result = find_available_port_with_lock(
        min_port=TEST_HTTP_SERVER_PORT_RANGE[0],
        max_port=TEST_HTTP_SERVER_PORT_RANGE[1],
        max_attempts=20,
        bind_address='0.0.0.0',
        lock_timeout=2.0,
    )

    if result is None:
        pytest.fail(
            f'Could not allocate a dynamic port in range {TEST_HTTP_SERVER_PORT_RANGE}'
        )

    port, port_lock = result
    logger.info(f'Allocated dynamic port {port} for test {request.node.name}')

    def cleanup():
        if port_lock:
            port_lock.release()
            logger.info(f'Released dynamic port {port} for test {request.node.name}')

    request.addfinalizer(cleanup)
    return port


# Export necessary function
__all__ = [
    '_load_runtime',
    '_get_host_folder',
    '_remove_folder',
    'dynamic_port',
]
