import os
import platform
import typing
from functools import lru_cache
from typing import Callable
from uuid import UUID

import docker
import httpx
import tenacity
from docker.models.containers import Container
from docker.types import DriverConfig, Mount

from openhands.core.config import OpenHandsConfig
from openhands.core.exceptions import (
    AgentRuntimeDisconnectedError,
    AgentRuntimeNotFoundError,
)
from openhands.core.logger import DEBUG, DEBUG_RUNTIME
from openhands.core.logger import openhands_logger as logger
from openhands.events import EventStream
from openhands.integrations.provider import PROVIDER_TOKEN_TYPE
from openhands.llm.llm_registry import LLMRegistry
from openhands.runtime.builder import DockerRuntimeBuilder
from openhands.runtime.impl.action_execution.action_execution_client import (
    ActionExecutionClient,
)
from openhands.runtime.impl.docker.containers import stop_all_containers
from openhands.runtime.plugins import PluginRequirement
from openhands.runtime.runtime_status import RuntimeStatus
from openhands.runtime.utils import find_available_tcp_port
from openhands.runtime.utils.command import (
    DEFAULT_MAIN_MODULE,
    get_action_execution_server_startup_command,
)
from openhands.runtime.utils.log_streamer import LogStreamer
from openhands.runtime.utils.port_lock import PortLock, find_available_port_with_lock
from openhands.runtime.utils.runtime_build import build_runtime_image
from openhands.utils.async_utils import call_sync_from_async
from openhands.utils.shutdown_listener import add_shutdown_listener
from openhands.utils.tenacity_stop import stop_if_should_exit

CONTAINER_NAME_PREFIX = 'openhands-runtime-'

EXECUTION_SERVER_PORT_RANGE = (30000, 39999)
VSCODE_PORT_RANGE = (40000, 49999)
APP_PORT_RANGE_1 = (50000, 54999)
APP_PORT_RANGE_2 = (55000, 59999)

if os.name == 'nt' or platform.release().endswith('microsoft-standard-WSL2'):
    EXECUTION_SERVER_PORT_RANGE = (30000, 34999)
    VSCODE_PORT_RANGE = (35000, 39999)
    APP_PORT_RANGE_1 = (40000, 44999)
    APP_PORT_RANGE_2 = (45000, 49151)


def _is_retryablewait_until_alive_error(exception: Exception) -> bool:
    if isinstance(exception, tenacity.RetryError):
        cause = exception.last_attempt.exception()
        return _is_retryablewait_until_alive_error(cause)

    return isinstance(
        exception,
        (
            ConnectionError,
            httpx.ConnectTimeout,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
            httpx.HTTPStatusError,
            httpx.ReadTimeout,
        ),
    )


class DockerRuntime(ActionExecutionClient):
    """This runtime will subscribe the event stream.

    When receive an event, it will send the event to runtime-client which run inside the docker environment.

    Args:
        config (OpenHandsConfig): The application configuration.
        event_stream (EventStream): The event stream to subscribe to.
        sid (str, optional): The session ID. Defaults to 'default'.
        plugins (list[PluginRequirement] | None, optional): List of plugin requirements. Defaults to None.
        env_vars (dict[str, str] | None, optional): Environment variables to set. Defaults to None.
    """

    _shutdown_listener_id: UUID | None = None

    def __init__(
        self,
        config: OpenHandsConfig,
        event_stream: EventStream,
        llm_registry: LLMRegistry,
        sid: str = 'default',
        plugins: list[PluginRequirement] | None = None,
        env_vars: dict[str, str] | None = None,
        status_callback: Callable | None = None,
        attach_to_existing: bool = False,
        headless_mode: bool = True,
        user_id: str | None = None,
        git_provider_tokens: PROVIDER_TOKEN_TYPE | None = None,
        main_module: str = DEFAULT_MAIN_MODULE,
    ):
        if not DockerRuntime._shutdown_listener_id:
            DockerRuntime._shutdown_listener_id = add_shutdown_listener(
                lambda: stop_all_containers(CONTAINER_NAME_PREFIX)
            )

        self.config = config
        self.status_callback = status_callback

        self._host_port = -1
        self._container_port = -1
        self._vscode_port = -1
        self._app_ports: list[int] = []

        # Port locks to prevent race conditions
        self._host_port_lock: PortLock | None = None
        self._vscode_port_lock: PortLock | None = None
        self._app_port_locks: list[PortLock] = []

        if os.environ.get('DOCKER_HOST_ADDR'):
            logger.info(
                f'Using DOCKER_HOST_IP: {os.environ["DOCKER_HOST_ADDR"]} for local_runtime_url'
            )
            self.config.sandbox.local_runtime_url = (
                f'http://{os.environ["DOCKER_HOST_ADDR"]}'
            )

        self.docker_client: docker.DockerClient = self._init_docker_client()
        self.api_url = f'{self.config.sandbox.local_runtime_url}:{self._container_port}'

        self.base_container_image = self.config.sandbox.base_container_image
        self.runtime_container_image = self.config.sandbox.runtime_container_image
        self.container_name = CONTAINER_NAME_PREFIX + sid
        self.container: Container | None = None
        self.main_module = main_module

        self.runtime_builder = DockerRuntimeBuilder(self.docker_client)

        # Buffer for container logs
        self.log_streamer: LogStreamer | None = None

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

        # Log runtime_extra_deps after base class initialization so self.sid is available
        if self.config.sandbox.runtime_extra_deps:
            self.log(
                'debug',
                f'Installing extra user-provided dependencies in the runtime image: {self.config.sandbox.runtime_extra_deps}',
            )

    @property
    def action_execution_server_url(self) -> str:
        return self.api_url

    async def connect(self) -> None:
        self.set_runtime_status(RuntimeStatus.STARTING_RUNTIME)
        try:
            await call_sync_from_async(self._attach_to_container)
        except docker.errors.NotFound as e:
            if self.attach_to_existing:
                self.log(
                    'warning',
                    f'Container {self.container_name} not found.',
                )
                raise AgentRuntimeDisconnectedError from e
            self.maybe_build_runtime_container_image()
            self.log(
                'info', f'Starting runtime with image: {self.runtime_container_image}'
            )
            await call_sync_from_async(self.init_container)
            self.log(
                'info',
                f'Container started: {self.container_name}. VSCode URL: {self.vscode_url}',
            )

        if DEBUG_RUNTIME and self.container:
            self.log_streamer = LogStreamer(self.container, self.log)
        else:
            self.log_streamer = None

        if not self.attach_to_existing:
            self.log('info', f'Waiting for client to become ready at {self.api_url}...')
            self.set_runtime_status(RuntimeStatus.STARTING_RUNTIME)

        await call_sync_from_async(self.wait_until_alive)

        if not self.attach_to_existing:
            self.log('info', 'Runtime is ready.')

        if not self.attach_to_existing:
            await call_sync_from_async(self.setup_initial_env)

        self.log(
            'debug',
            f'Container initialized with plugins: {[plugin.name for plugin in self.plugins]}. VSCode URL: {self.vscode_url}',
        )
        if not self.attach_to_existing:
            self.set_runtime_status(RuntimeStatus.READY)
        self._runtime_initialized = True

        for network_name in self.config.sandbox.additional_networks:
            try:
                network = self.docker_client.networks.get(network_name)
                if self.container is not None:
                    network.connect(self.container)
                else:
                    self.log(
                        'warning',
                        f'Container not available to connect to network {network_name}',
                    )
            except Exception as e:
                self.log(
                    'error',
                    f'Error: Failed to connect instance {self.container_name} to network {network_name}',
                )
                self.log('error', str(e))

    def maybe_build_runtime_container_image(self):
        if self.runtime_container_image is None:
            if self.base_container_image is None:
                raise ValueError(
                    'Neither runtime container image nor base container image is set'
                )
            self.set_runtime_status(RuntimeStatus.BUILDING_RUNTIME)
            self.runtime_container_image = build_runtime_image(
                self.base_container_image,
                self.runtime_builder,
                platform=self.config.sandbox.platform,
                extra_deps=self.config.sandbox.runtime_extra_deps,
                force_rebuild=self.config.sandbox.force_rebuild_runtime,
                extra_build_args=self.config.sandbox.runtime_extra_build_args,
                enable_browser=self.config.enable_browser,
            )

    @staticmethod
    @lru_cache(maxsize=1)
    def _init_docker_client() -> docker.DockerClient:
        try:
            return docker.from_env()
        except Exception as ex:
            logger.error(
                'Launch docker client failed. Please make sure you have installed docker and started docker desktop/daemon.',
            )
            raise ex

    def _process_volumes(self) -> dict[str, dict[str, str]]:
        """Process volume mounts based on configuration.

        Returns:
            A dictionary mapping host paths to container bind mounts with their modes.
        """
        # Initialize volumes dictionary
        volumes: dict[str, dict[str, str]] = {}

        # Process volumes (comma-delimited)
        if self.config.sandbox.volumes is not None:
            # Handle multiple mounts with comma delimiter
            mounts = self.config.sandbox.volumes.split(',')

            for mount in mounts:
                parts = mount.split(':')
                if len(parts) >= 2:
                    # Support both bind mounts (absolute paths) and Docker named volumes.
                    # Named volume syntax:
                    #   volume:<name>   (explicit)
                    #   <name>          (implicit when not starting with '/')
                    raw_host_part = parts[0]

                    if raw_host_part.startswith('volume:'):
                        host_path = raw_host_part.split('volume:', 1)[1]
                    elif not os.path.isabs(raw_host_part):
                        host_path = raw_host_part  # treat as named volume
                    else:
                        host_path = os.path.abspath(raw_host_part)
                    container_path = parts[1]
                    # Default mode is 'rw' if not specified
                    mount_mode = parts[2] if len(parts) > 2 else 'rw'
                    # Skip overlay mounts here; they will be handled separately via Mount objects
                    if 'overlay' in mount_mode:
                        continue

                    volumes[host_path] = {
                        'bind': container_path,
                        'mode': mount_mode,
                    }
                    logger.debug(
                        f'Mount dir (sandbox.volumes): {host_path} to {container_path} with mode: {mount_mode}'
                    )

        # Legacy mounting with workspace_* parameters
        elif (
            self.config.workspace_mount_path is not None
            and self.config.workspace_mount_path_in_sandbox is not None
        ):
            mount_mode = 'rw'  # Default mode

            # e.g. result would be: {"/home/user/openhands/workspace": {'bind': "/workspace", 'mode': 'rw'}}
            # Add os.path.abspath() here so that relative paths can be used when workspace_mount_path is configured in config.toml
            volumes[os.path.abspath(self.config.workspace_mount_path)] = {
                'bind': self.config.workspace_mount_path_in_sandbox,
                'mode': mount_mode,
            }
            logger.debug(
                f'Mount dir (legacy): {self.config.workspace_mount_path} with mode: {mount_mode}'
            )

        return volumes

    def _process_overlay_mounts(self) -> list[Mount]:
        """Process overlay mounts specified in sandbox.volumes with mode containing 'overlay'.

        Returns:
            List of docker.types.Mount objects configured with overlay driver providing
            read-only lowerdir with per-container copy-on-write upper/work layers.
        """
        overlay_mounts: list[Mount] = []

        # No volumes configured
        if self.config.sandbox.volumes is None:
            return overlay_mounts

        # Base directory for overlay upper/work layers from env var
        overlay_base = os.environ.get('SANDBOX_VOLUME_OVERLAYS')
        if not overlay_base:
            # If no base path provided, skip overlay processing
            return overlay_mounts

        os.makedirs(overlay_base, exist_ok=True)

        mount_specs = self.config.sandbox.volumes.split(',')

        for idx, mount_spec in enumerate(mount_specs):
            parts = mount_spec.split(':')
            if len(parts) < 2:
                continue
            host_path = os.path.abspath(parts[0])
            container_path = parts[1]
            mount_mode = parts[2] if len(parts) > 2 else 'rw'

            # Only consider overlay mounts for host-bind paths (absolute)
            if (not os.path.isabs(parts[0])) or ('overlay' not in mount_mode):
                continue

            # Prepare upper and work directories unique to this container and mount
            overlay_dir = os.path.join(overlay_base, self.container_name, f'{idx}')
            upper_dir = os.path.join(overlay_dir, 'upper')
            work_dir = os.path.join(overlay_dir, 'work')
            os.makedirs(upper_dir, exist_ok=True)
            os.makedirs(work_dir, exist_ok=True)

            driver_cfg = DriverConfig(
                name='local',
                options={
                    'type': 'overlay',
                    'device': 'overlay',
                    'o': f'lowerdir={host_path},upperdir={upper_dir},workdir={work_dir}',
                },
            )

            mount = Mount(
                target=container_path,
                source='',  # Anonymous volume
                type='volume',
                labels={
                    'app': 'openhands',
                    'role': 'worker',
                    'container': self.container_name,
                },
                driver_config=driver_cfg,
            )

            overlay_mounts.append(mount)

        return overlay_mounts

    def init_container(self) -> None:
        self.log('debug', 'Preparing to start container...')
        self.set_runtime_status(RuntimeStatus.STARTING_RUNTIME)

        # Allocate host port with locking to prevent race conditions
        self._host_port, self._host_port_lock = self._find_available_port_with_lock(
            EXECUTION_SERVER_PORT_RANGE
        )
        self._container_port = self._host_port

        # Use the configured vscode_port if provided, otherwise find an available port
        if self.config.sandbox.vscode_port:
            self._vscode_port = self.config.sandbox.vscode_port
            self._vscode_port_lock = None  # No lock needed for configured port
        else:
            self._vscode_port, self._vscode_port_lock = (
                self._find_available_port_with_lock(VSCODE_PORT_RANGE)
            )

        # Allocate app ports with locking
        app_port_1, app_lock_1 = self._find_available_port_with_lock(APP_PORT_RANGE_1)
        app_port_2, app_lock_2 = self._find_available_port_with_lock(APP_PORT_RANGE_2)

        self._app_ports = [app_port_1, app_port_2]
        self._app_port_locks = [
            lock for lock in [app_lock_1, app_lock_2] if lock is not None
        ]

        self.api_url = f'{self.config.sandbox.local_runtime_url}:{self._container_port}'

        use_host_network = self.config.sandbox.use_host_network
        network_mode: typing.Literal['host'] | None = (
            'host' if use_host_network else None
        )

        # Initialize port mappings
        port_mapping: dict[str, list[dict[str, str]]] | None = None
        if not use_host_network:
            port_mapping = {
                f'{self._container_port}/tcp': [
                    {
                        'HostPort': str(self._host_port),
                        'HostIp': self.config.sandbox.runtime_binding_address,
                    }
                ],
            }

            if self.vscode_enabled:
                port_mapping[f'{self._vscode_port}/tcp'] = [
                    {
                        'HostPort': str(self._vscode_port),
                        'HostIp': self.config.sandbox.runtime_binding_address,
                    }
                ]

            for port in self._app_ports:
                port_mapping[f'{port}/tcp'] = [
                    {
                        'HostPort': str(port),
                        'HostIp': self.config.sandbox.runtime_binding_address,
                    }
                ]
        else:
            self.log(
                'warn',
                'Using host network mode. If you are using MacOS, please make sure you have the latest version of Docker Desktop and enabled host network feature: https://docs.docker.com/network/drivers/host/#docker-desktop',
            )

        # Combine environment variables
        environment = dict(**self.initial_env_vars)
        environment.update(
            {
                'port': str(self._container_port),
                'PYTHONUNBUFFERED': '1',
                # Passing in the ports means nested runtimes do not come up with their own ports!
                'VSCODE_PORT': str(self._vscode_port),
                'APP_PORT_1': str(self._app_ports[0]),
                'APP_PORT_2': str(self._app_ports[1]),
                'OPENHANDS_SESSION_ID': str(self.sid),
                'PIP_BREAK_SYSTEM_PACKAGES': '1',
            }
        )
        if self.config.debug or DEBUG:
            environment['DEBUG'] = 'true'
        # also update with runtime_startup_env_vars
        environment.update(self.config.sandbox.runtime_startup_env_vars)

        self.log('debug', f'Workspace Base: {self.config.workspace_base}')

        # Process volumes for mounting
        volumes = self._process_volumes()

        # If no volumes were configured, set to None
        if not volumes:
            logger.debug(
                'Mount dir is not set, will not mount the workspace directory to the container'
            )
            volumes = {}  # Empty dict instead of None to satisfy mypy
        self.log(
            'debug',
            f'Sandbox workspace: {self.config.workspace_mount_path_in_sandbox}',
        )

        command = self.get_action_execution_server_startup_command()
        self.log('info', f'Starting server with command: {command}')

        if self.config.sandbox.enable_gpu:
            gpu_ids = self.config.sandbox.cuda_visible_devices
            if gpu_ids is None:
                device_requests = [
                    docker.types.DeviceRequest(capabilities=[['gpu']], count=-1)
                ]
            else:
                device_requests = [
                    docker.types.DeviceRequest(
                        capabilities=[['gpu']],
                        device_ids=[str(i) for i in gpu_ids.split(',')],
                    )
                ]
        else:
            device_requests = None
        try:
            if self.runtime_container_image is None:
                raise ValueError('Runtime container image is not set')
            # Process overlay mounts (read-only lower with per-container COW)
            overlay_mounts = self._process_overlay_mounts()

            self.container = self.docker_client.containers.run(
                self.runtime_container_image,
                command=command,
                # Override the default 'bash' entrypoint because the command is a binary.
                entrypoint=[],
                network_mode=network_mode,
                ports=port_mapping,
                working_dir='/openhands/code/',  # do not change this!
                name=self.container_name,
                detach=True,
                environment=environment,
                volumes=volumes,  # type: ignore
                mounts=overlay_mounts,  # type: ignore
                device_requests=device_requests,
                **(self.config.sandbox.docker_runtime_kwargs or {}),
            )
            self.log('debug', f'Container started. Server url: {self.api_url}')
            self.set_runtime_status(RuntimeStatus.RUNTIME_STARTED)
        except Exception as e:
            self.log(
                'error',
                f'Error: Instance {self.container_name} FAILED to start container!\n',
            )
            self.close()
            raise e

    def _attach_to_container(self) -> None:
        self.container = self.docker_client.containers.get(self.container_name)
        if self.container.status == 'exited':
            self.container.start()

        config = self.container.attrs['Config']
        for env_var in config['Env']:
            if env_var.startswith('port='):
                self._host_port = int(env_var.split('port=')[1])
                self._container_port = self._host_port
            elif env_var.startswith('VSCODE_PORT='):
                self._vscode_port = int(env_var.split('VSCODE_PORT=')[1])

        self._app_ports = []
        exposed_ports = config.get('ExposedPorts')
        if exposed_ports:
            for exposed_port in exposed_ports.keys():
                exposed_port = int(exposed_port.split('/tcp')[0])
                if (
                    exposed_port != self._host_port
                    and exposed_port != self._vscode_port
                ):
                    self._app_ports.append(exposed_port)

        self.api_url = f'{self.config.sandbox.local_runtime_url}:{self._container_port}'
        self.log(
            'debug',
            f'attached to container: {self.container_name} {self._container_port} {self.api_url}',
        )

    @tenacity.retry(
        stop=tenacity.stop_after_delay(120) | stop_if_should_exit(),
        retry=tenacity.retry_if_exception(_is_retryablewait_until_alive_error),
        reraise=True,
        wait=tenacity.wait_fixed(2),
    )
    def wait_until_alive(self) -> None:
        try:
            container = self.docker_client.containers.get(self.container_name)
            if container.status == 'exited':
                raise AgentRuntimeDisconnectedError(
                    f'Container {self.container_name} has exited.'
                )
        except docker.errors.NotFound:
            raise AgentRuntimeNotFoundError(
                f'Container {self.container_name} not found.'
            )

        self.check_if_alive()

    def close(self, rm_all_containers: bool | None = None) -> None:
        """Closes the DockerRuntime and associated objects

        Parameters:
        - rm_all_containers (bool): Whether to remove all containers with the 'openhands-sandbox-' prefix
        """
        super().close()
        if self.log_streamer:
            self.log_streamer.close()

        if rm_all_containers is None:
            rm_all_containers = self.config.sandbox.rm_all_containers

        if self.config.sandbox.keep_runtime_alive or self.attach_to_existing:
            return
        close_prefix = (
            CONTAINER_NAME_PREFIX if rm_all_containers else self.container_name
        )
        stop_all_containers(close_prefix)
        self._release_port_locks()

    def _release_port_locks(self) -> None:
        """Release all acquired port locks."""
        if self._host_port_lock:
            self._host_port_lock.release()
            self._host_port_lock = None
            logger.debug(f'Released host port lock for port {self._host_port}')

        if self._vscode_port_lock:
            self._vscode_port_lock.release()
            self._vscode_port_lock = None
            logger.debug(f'Released VSCode port lock for port {self._vscode_port}')

        for i, lock in enumerate(self._app_port_locks):
            if lock:
                lock.release()
                logger.debug(
                    f'Released app port lock for port {self._app_ports[i] if i < len(self._app_ports) else "unknown"}'
                )

        self._app_port_locks.clear()

    def _is_port_in_use_docker(self, port: int) -> bool:
        containers = self.docker_client.containers.list()
        for container in containers:
            container_ports = container.ports
            if str(port) in str(container_ports):
                return True
        return False

    def _find_available_port_with_lock(
        self, port_range: tuple[int, int], max_attempts: int = 5
    ) -> tuple[int, PortLock | None]:
        """Find an available port with race condition protection.

        This method uses file-based locking to prevent multiple workers
        from allocating the same port simultaneously.

        Args:
            port_range: Tuple of (min_port, max_port)
            max_attempts: Maximum number of attempts to find a port

        Returns:
            Tuple of (port_number, port_lock) where port_lock may be None if locking failed
        """
        # Try to find and lock an available port
        result = find_available_port_with_lock(
            min_port=port_range[0],
            max_port=port_range[1],
            max_attempts=max_attempts,
            bind_address='0.0.0.0',
            lock_timeout=1.0,
        )

        if result is None:
            # Fallback to original method if port locking fails
            logger.warning(
                f'Port locking failed for range {port_range}, falling back to original method'
            )
            port = port_range[1]
            for _ in range(max_attempts):
                port = find_available_tcp_port(port_range[0], port_range[1])
                if not self._is_port_in_use_docker(port):
                    return port, None
            return port, None

        port, port_lock = result

        # Additional check with Docker to ensure port is not in use
        if self._is_port_in_use_docker(port):
            port_lock.release()
            # Try again with a different port
            logger.debug(f'Port {port} is in use by Docker, trying again')
            return self._find_available_port_with_lock(port_range, max_attempts - 1)

        return port, port_lock

    def _find_available_port(
        self, port_range: tuple[int, int], max_attempts: int = 5
    ) -> int:
        """Find an available port (legacy method for backward compatibility)."""
        port, _ = self._find_available_port_with_lock(port_range, max_attempts)
        return port

    @property
    def vscode_url(self) -> str | None:
        token = super().get_vscode_token()
        if not token:
            return None

        vscode_url = f'http://localhost:{self._vscode_port}/?tkn={token}&folder={self.config.workspace_mount_path_in_sandbox}'
        return vscode_url

    @property
    def web_hosts(self) -> dict[str, int]:
        hosts: dict[str, int] = {}

        host_addr = os.environ.get('DOCKER_HOST_ADDR', 'localhost')
        for port in self._app_ports:
            hosts[f'http://{host_addr}:{port}'] = port

        return hosts

    def pause(self) -> None:
        """Pause the runtime by stopping the container.
        This is different from container.stop() as it ensures environment variables are properly preserved.
        """
        if not self.container:
            raise RuntimeError('Container not initialized')

        # First, ensure all environment variables are properly persisted in .bashrc
        # This is already handled by add_env_vars in base.py

        # Stop the container
        self.container.stop()
        self.log('debug', f'Container {self.container_name} paused')

    def resume(self) -> None:
        """Resume the runtime by starting the container.
        This is different from container.start() as it ensures environment variables are properly restored.
        """
        if not self.container:
            raise RuntimeError('Container not initialized')

        # Start the container
        self.container.start()
        self.log('debug', f'Container {self.container_name} resumed')

        # Wait for the container to be ready
        self.wait_until_alive()

    @classmethod
    async def delete(cls, conversation_id: str) -> None:
        docker_client = cls._init_docker_client()
        try:
            container_name = CONTAINER_NAME_PREFIX + conversation_id
            container = docker_client.containers.get(container_name)
            container.remove(force=True)
        except docker.errors.APIError:
            pass
        except docker.errors.NotFound:
            pass
        finally:
            docker_client.close()

    def get_action_execution_server_startup_command(self) -> list[str]:
        return get_action_execution_server_startup_command(
            server_port=self._container_port,
            plugins=self.plugins,
            app_config=self.config,
            main_module=self.main_module,
        )
