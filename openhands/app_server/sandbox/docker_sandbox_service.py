import asyncio
import logging
import os
import shutil
import socket
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import AsyncGenerator

import base62
import docker
import httpx
from docker.errors import APIError, NotFound
from fastapi import Request
from pydantic import BaseModel, ConfigDict, Field

from openhands.agent_server.utils import utc_now
from openhands.app_server.errors import SandboxError
from openhands.app_server.sandbox.docker_sandbox_spec_service import get_docker_client
from openhands.app_server.sandbox.sandbox_models import (
    AGENT_SERVER,
    VSCODE,
    WORKER_1,
    WORKER_2,
    WORKER_3,
    ExposedUrl,
    SandboxInfo,
    SandboxPage,
    SandboxStatus,
)
from openhands.app_server.sandbox.sandbox_service import (
    ALLOW_CORS_ORIGINS_VARIABLE,
    SESSION_API_KEY_VARIABLE,
    WEBHOOK_CALLBACK_VARIABLE,
    SandboxService,
    SandboxServiceInjector,
)
from openhands.app_server.sandbox.sandbox_spec_service import SandboxSpecService
from openhands.app_server.services.injector import InjectorState
from openhands.app_server.utils.docker_utils import (
    replace_localhost_hostname_for_docker,
)

_logger = logging.getLogger(__name__)
STARTUP_GRACE_SECONDS = 60

_DOCKER_SOCKET = '/var/run/docker.sock'
_PACKAGE_CACHE_PATH = '/opt/package-cache'


def _docker_socket_group(volumes: dict) -> list[int]:
    """Return group IDs to add when /var/run/docker.sock is mounted.

    If the Docker socket is among the mounted volumes, return its owning
    group so the non-root container user can access it.
    """
    for host_path, bind_info in volumes.items():
        container_path = (
            bind_info.get('bind', '') if isinstance(bind_info, dict) else ''
        )
        if container_path == _DOCKER_SOCKET or host_path == _DOCKER_SOCKET:
            try:
                gid = os.stat(_DOCKER_SOCKET).st_gid
                if gid != 0:
                    return [gid]
            except OSError:
                pass
    return []


class VolumeMount(BaseModel):
    """Mounted volume within the container."""

    host_path: str
    container_path: str
    mode: str = 'rw'

    model_config = ConfigDict(frozen=True)


class ExposedPort(BaseModel):
    """Exposed port within container to be matched to a free port on the host."""

    name: str
    description: str
    container_port: int = 8000

    model_config = ConfigDict(frozen=True)


@dataclass
class DockerSandboxService(SandboxService):
    """Sandbox service built on docker.

    The Docker API does not currently support async operations, so some of these operations will block.
    Given that the docker API is intended for local use on a single machine, this is probably acceptable.
    """

    sandbox_spec_service: SandboxSpecService
    container_name_prefix: str
    host_port: int
    container_url_pattern: str
    mounts: list[VolumeMount]
    exposed_ports: list[ExposedPort]
    health_check_path: str | None
    httpx_client: httpx.AsyncClient
    max_num_sandboxes: int
    resource_prefix: str = 'openhands'
    web_url: str | None = None
    extra_hosts: dict[str, str] = field(default_factory=dict)
    network: str | None = None
    app_hostname: str | None = None
    container_labels: dict[str, str] = field(default_factory=dict)
    privileged: bool = False
    registry_mirror_url: str | None = None
    registry_mirrors: dict[str, str] = field(default_factory=dict)
    traefik_network: str | None = None
    traefik_domain: str | None = None
    traefik_entrypoints: str = 'web'
    traefik_certresolver: str | None = None
    traefik_worker_ports: list[str] | None = None
    traefik_subdomain_prefix: str | None = None
    traefik_scheme: str = 'https'
    docker_client: docker.DockerClient = field(default_factory=get_docker_client)
    startup_grace_seconds: int = STARTUP_GRACE_SECONDS

    def _build_traefik_labels(
        self,
        container_name: str,
        sandbox_id: str,
        exposed_ports: list[ExposedPort],
    ) -> dict[str, str]:
        """Build Traefik labels for automatic subdomain routing of worker ports.

        Only generates labels for WORKER_* ports. Returns an empty dict when
        Traefik integration is not configured.

        When ``traefik_subdomain_prefix`` is set the subdomain is
        ``{prefix}-{sandbox_id}.{domain}`` (e.g. ``mystack-abc123.example.com``).
        Otherwise it falls back to ``{container_name}-{worker_name}.{domain}``.
        """
        if not self.traefik_network or not self.traefik_domain:
            return {}

        labels: dict[str, str] = {
            'traefik.enable': 'true',
            'traefik.docker.network': self.traefik_network,
        }
        allowed = self.traefik_worker_ports
        for ep in exposed_ports:
            if not ep.name.startswith('WORKER_'):
                continue
            if allowed and ep.name not in allowed:
                continue
            if self.traefik_subdomain_prefix:
                subdomain = f'{self.traefik_subdomain_prefix}-{sandbox_id}'
            else:
                worker_name = ep.name.lower().replace('_', '-')
                subdomain = f'{container_name}-{worker_name}'
            router_name = subdomain
            labels[f'traefik.http.routers.{router_name}.rule'] = (
                f'Host(`{subdomain}.{self.traefik_domain}`)'
            )
            labels[f'traefik.http.routers.{router_name}.entrypoints'] = (
                self.traefik_entrypoints
            )
            labels[f'traefik.http.services.{router_name}.loadbalancer.server.port'] = (
                str(ep.container_port)
            )
            if self.traefik_certresolver:
                labels[f'traefik.http.routers.{router_name}.tls'] = 'true'
                labels[f'traefik.http.routers.{router_name}.tls.certresolver'] = (
                    self.traefik_certresolver
                )
        return labels

    async def _start_dockerd(self, container) -> None:
        """Start dockerd inside a privileged container for DinD support."""
        _logger.info(f'Starting dockerd in container {container.name}...')

        # In DinD the container's root filesystem is itself an overlayfs
        # mount.  Running Docker inside creates nested overlayfs which fails
        # when extracting image layers that contain whiteout files ("operation
        # not permitted").  Mounting a tmpfs at /var/lib/docker gives the
        # inner Docker a real filesystem so its overlayfs works correctly.
        # This also lets us keep the containerd image store active (needed
        # for per-registry hosts.toml mirror configs).
        container.exec_run('mkdir -p /var/lib/docker', user='root')
        container.exec_run(
            'mount -t tmpfs -o size=8G tmpfs /var/lib/docker', user='root'
        )

        dockerd_args = ''

        if self.registry_mirrors:
            # Write per-registry mirror configs (hosts.toml).  Docker 29+
            # with the containerd image store reads host configs from
            # /etc/docker/certs.d/<host>/hosts.toml.
            #
            # We only grant the "pull" capability (blob downloads) to the
            # cache — NOT "resolve".  The containerd image store resolves
            # manifests differently from the legacy store and the registry:2
            # pull-through cache returns manifest lists where platform-
            # specific manifests are expected, causing size-validation
            # failures.  By omitting "resolve", containerd resolves tags
            # directly against the upstream registry (authenticated via
            # ``docker login`` below) and only routes blob fetches through
            # the cache.
            container.exec_run('mkdir -p /etc/docker/certs.d', user='root')
            for upstream_host, mirror_url in self.registry_mirrors.items():
                host_dir = f'/etc/docker/certs.d/{upstream_host}'
                container.exec_run(f'mkdir -p {host_dir}', user='root')
                hosts_toml = (
                    f'server = "https://{upstream_host}"\n\n'
                    f'[host."{mirror_url}"]\n'
                    f'  capabilities = ["pull"]\n'
                    f'  skip_verify = true\n'
                )
                container.exec_run(
                    [
                        'bash',
                        '-c',
                        f"cat > {host_dir}/hosts.toml << 'TOML'\n{hosts_toml}TOML",
                    ],
                    user='root',
                )

        if self.registry_mirror_url:
            dockerd_args += (
                f' --registry-mirror={self.registry_mirror_url}'
                f' --insecure-registry={self.registry_mirror_url.split("//", 1)[-1]}'
            )

        container.exec_run(
            f'bash -c "dockerd {dockerd_args} > /tmp/dockerd.log 2>&1 &"',
            user='root',
            detach=True,
        )
        # Wait for dockerd to become ready
        loop = asyncio.get_running_loop()
        for i in range(30):
            result = await loop.run_in_executor(
                None,
                lambda: container.exec_run('docker info', user='root'),
            )
            if result.exit_code == 0:
                _logger.info(
                    f'dockerd ready in container {container.name} after {i + 1}s'
                )
                # Make the socket accessible to non-root users
                await loop.run_in_executor(
                    None,
                    lambda: container.exec_run(
                        'chmod 666 /var/run/docker.sock', user='root'
                    ),
                )

                # Authenticate to ECR if AWS credentials are available
                await self._login_ecr(container)
                return
            await asyncio.sleep(1)
        # Capture dockerd log to help diagnose startup failures.
        log_result = container.exec_run('cat /tmp/dockerd.log', user='root')
        dockerd_log = (
            log_result.output.decode(errors='replace')
            if log_result.output
            else '(empty)'
        )
        _logger.warning(
            f'dockerd failed to start in container {container.name}. '
            f'dockerd log:\n{dockerd_log}'
        )

    @staticmethod
    def _get_codeartifact_token(env_vars: dict[str, str]) -> str | None:
        """Fetch a CodeArtifact authorization token using AWS credentials.

        Uses B1_ACCESS_KEY_ID / B1_SECRET_ACCESS_KEY from env_vars if
        available, otherwise falls back to the server's default AWS
        credential chain (IAM role, environment, etc.).

        Returns the token string, or None if the API call fails.
        """
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError

        region = os.environ.get('AWS_REGION', 'eu-central-1')
        access_key = env_vars.get('B1_ACCESS_KEY_ID', '')
        secret_key = env_vars.get('B1_SECRET_ACCESS_KEY', '')

        try:
            if access_key and secret_key:
                session = boto3.Session(
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    region_name=region,
                )
            else:
                session = boto3.Session(region_name=region)
            ca_client = session.client('codeartifact')
            ca_response = ca_client.get_authorization_token(
                domain='buildone',
                domainOwner='653306034207',
            )
            _logger.info('CodeArtifact auth token fetched successfully')
            return ca_response['authorizationToken']
        except (BotoCoreError, ClientError) as e:
            _logger.warning(f'Failed to fetch CodeArtifact auth token: {e}')
            return None

    async def _login_ecr(self, container) -> None:
        """Authenticate the sandbox's Docker daemon to AWS ECR.

        Uses the container's B1_ACCESS_KEY_ID and B1_SECRET_ACCESS_KEY
        environment variables to obtain an ECR authorization token via
        boto3, then runs ``docker login`` inside the container.
        """
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError

        # Read credentials from the container's environment
        result = container.exec_run('bash -c "echo $B1_ACCESS_KEY_ID"', user='root')
        access_key = result.output.decode().strip() if result.output else ''

        result = container.exec_run('bash -c "echo $B1_SECRET_ACCESS_KEY"', user='root')
        secret_key = result.output.decode().strip() if result.output else ''

        if not access_key or not secret_key:
            _logger.debug(
                f'No AWS credentials in container {container.name}, skipping ECR login'
            )
            return

        region = os.environ.get('AWS_REGION', 'eu-central-1')
        try:
            session = boto3.Session(
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=region,
            )
            ecr_client = session.client('ecr')
            token_response = ecr_client.get_authorization_token()
            auth_data = token_response['authorizationData'][0]
            token = auth_data['authorizationToken']
            endpoint = auth_data['proxyEndpoint']

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: container.exec_run(
                    f'bash -c "echo {token} | base64 -d | cut -d: -f2 | '
                    f'docker login --username AWS --password-stdin {endpoint}"',
                    user='root',
                ),
            )
            _logger.info(f'ECR login successful in container {container.name}')
        except (BotoCoreError, ClientError) as e:
            _logger.warning(f'ECR login failed in container {container.name}: {e}')

    def _schedule_codespace_port_visibility(
        self, port_mappings: dict[int, int]
    ) -> None:
        """Fire-and-forget background task to make sandbox ports public.

        Codespaces auto-forwards docker-proxy ports but defaults them to
        ``private``.  Auto-detection can take up to ~60 s to register a new
        port with the tunnel service, so this runs in the background with
        retries — it never blocks sandbox startup.
        """
        codespace_name = os.getenv('CODESPACE_NAME')
        if not codespace_name:
            return

        gh = shutil.which('gh')
        if not gh:
            _logger.debug('gh CLI not found — skipping Codespace port visibility')
            return

        github_token = os.getenv('CODESPACE_GITHUB_TOKEN')
        if not github_token:
            _logger.debug(
                'CODESPACE_GITHUB_TOKEN not set — skipping Codespace port visibility'
            )
            return

        worker_container_ports = {
            ep.container_port
            for ep in self.exposed_ports
            if ep.name.startswith('WORKER_')
        }

        ports_to_set = [
            host_port
            for container_port, host_port in port_mappings.items()
            if container_port in worker_container_ports
        ]
        if not ports_to_set:
            return

        async def _set_visibility(ports: list[int]) -> None:
            env = {**os.environ, 'GITHUB_TOKEN': github_token}
            max_attempts = 20  # ~60 s total (20 × 3 s)
            for attempt in range(max_attempts):
                if attempt > 0:
                    await asyncio.sleep(3)

                remaining: list[int] = []
                for host_port in ports:
                    try:
                        proc = await asyncio.create_subprocess_exec(
                            gh,
                            'codespace',
                            'ports',
                            'visibility',
                            f'{host_port}:public',
                            '-c',
                            codespace_name,
                            stdout=asyncio.subprocess.DEVNULL,
                            stderr=asyncio.subprocess.PIPE,
                            env=env,
                        )
                        _, stderr_bytes = await asyncio.wait_for(
                            proc.communicate(), timeout=10
                        )
                        if proc.returncode != 0:
                            err_msg = stderr_bytes.decode().strip()
                            if '404' in err_msg and attempt < max_attempts - 1:
                                remaining.append(host_port)
                            elif '404' in err_msg:
                                # Port never registered in the Codespace tunnel
                                # service. This is expected when devcontainer.json
                                # configures auto-forwarding (portsAttributes) for
                                # the port range — Codespaces will handle visibility
                                # once it detects the listener.
                                _logger.info(
                                    f'Codespace tunnel did not register port {host_port} '
                                    f'after {max_attempts} attempts — relying on '
                                    f'devcontainer.json auto-forward settings'
                                )
                            else:
                                _logger.warning(
                                    f'Failed to set port {host_port} public: {err_msg}'
                                )
                        else:
                            _logger.info(f'Codespace port {host_port} set to public')
                    except asyncio.TimeoutError:
                        if attempt < max_attempts - 1:
                            remaining.append(host_port)
                        else:
                            _logger.warning(
                                f'Timed out setting Codespace port {host_port} visibility'
                            )
                    except Exception as exc:
                        _logger.warning(
                            f'Error setting Codespace port {host_port} visibility: {exc}'
                        )

                ports = remaining
                if not ports:
                    break

        asyncio.create_task(_set_visibility(ports_to_set))

    def _find_unused_port(self) -> int:
        """Find an unused port on the host machine."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    def _docker_status_to_sandbox_status(self, docker_status: str) -> SandboxStatus:
        """Convert Docker container status to SandboxStatus."""
        status_mapping = {
            'running': SandboxStatus.RUNNING,
            'paused': SandboxStatus.PAUSED,
            # The stop button was pressed in the docker console
            'exited': SandboxStatus.PAUSED,
            'created': SandboxStatus.STARTING,
            'restarting': SandboxStatus.STARTING,
            'removing': SandboxStatus.MISSING,
            'dead': SandboxStatus.ERROR,
        }
        return status_mapping.get(docker_status.lower(), SandboxStatus.ERROR)

    def _get_container_env_vars(self, container) -> dict[str, str | None]:
        env_vars_list = container.attrs['Config']['Env']
        result = {}
        for env_var in env_vars_list:
            if '=' in env_var:
                key, value = env_var.split('=', 1)
                result[key] = value
            else:
                # Handle cases where an environment variable might not have a value
                result[env_var] = None
        return result

    async def _container_to_sandbox_info(self, container) -> SandboxInfo | None:
        """Convert Docker container to SandboxInfo."""
        # Convert Docker status to runtime status
        status = self._docker_status_to_sandbox_status(container.status)

        # Parse creation time
        created_str = container.attrs.get('Created', '')
        try:
            created_at = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            created_at = utc_now()

        # Get URL and session key for running containers
        exposed_urls = None
        session_api_key = None

        if status == SandboxStatus.RUNNING:
            # Get session API key first
            env = self._get_container_env_vars(container)
            session_api_key = env.get(SESSION_API_KEY_VARIABLE)

            # Get the first exposed port mapping
            exposed_urls = []
            port_bindings = container.attrs.get('NetworkSettings', {}).get('Ports', {})
            if port_bindings:
                for container_port, host_bindings in port_bindings.items():
                    if host_bindings:
                        host_port = host_bindings[0]['HostPort']
                        exposed_port = next(
                            (
                                exposed_port
                                for exposed_port in self.exposed_ports
                                if container_port
                                == f'{exposed_port.container_port}/tcp'
                            ),
                            None,
                        )
                        if exposed_port:
                            traefik_allowed = self.traefik_worker_ports
                            if (
                                self.traefik_domain
                                and self.traefik_network
                                and exposed_port.name.startswith('WORKER_')
                                and (
                                    not traefik_allowed
                                    or exposed_port.name in traefik_allowed
                                )
                            ):
                                if self.traefik_subdomain_prefix:
                                    sid = container.labels.get(
                                        'sandbox_id', container.name
                                    )
                                    subdomain = f'{self.traefik_subdomain_prefix}-{sid}'
                                else:
                                    worker_name = exposed_port.name.lower().replace(
                                        '_', '-'
                                    )
                                    subdomain = f'{container.name}-{worker_name}'
                                scheme = self.traefik_scheme
                                url = f'{scheme}://{subdomain}.{self.traefik_domain}'
                            else:
                                url = self.container_url_pattern.format(port=host_port)

                            # VSCode URLs require the api_key and working dir
                            if exposed_port.name == VSCODE:
                                url += f'/?tkn={session_api_key}&folder={container.attrs["Config"]["WorkingDir"]}'

                            # Compute internal URL for container-to-container communication
                            internal_url = None
                            if self.network:
                                internal_url = f'http://{container.name}:{exposed_port.container_port}'

                            exposed_urls.append(
                                ExposedUrl(
                                    name=exposed_port.name,
                                    url=url,
                                    port=host_port,
                                    internal_url=internal_url,
                                )
                            )

        return SandboxInfo(
            id=container.name,
            created_by_user_id=None,
            sandbox_spec_id=container.image.tags[0],
            status=status,
            session_api_key=session_api_key,
            exposed_urls=exposed_urls,
            created_at=created_at,
        )

    async def _container_to_checked_sandbox_info(self, container) -> SandboxInfo | None:
        """Convert container to SandboxInfo with health check validation."""
        sandbox_info = await self._container_to_sandbox_info(container)
        if (
            sandbox_info
            and self.health_check_path is not None
            and sandbox_info.exposed_urls
        ):
            agent_server_eu = next(
                exposed_url
                for exposed_url in sandbox_info.exposed_urls
                if exposed_url.name == AGENT_SERVER
            )
            app_server_url = agent_server_eu.internal_url or agent_server_eu.url
            try:
                # When running in Docker, replace localhost hostname with host.docker.internal for internal requests
                if not agent_server_eu.internal_url:
                    app_server_url = replace_localhost_hostname_for_docker(
                        app_server_url
                    )

                response = await self.httpx_client.get(
                    f'{app_server_url}{self.health_check_path}'
                )
                response.raise_for_status()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # If the server is
                if sandbox_info.created_at < utc_now() - timedelta(
                    seconds=self.startup_grace_seconds
                ):
                    _logger.info(
                        f'Sandbox server not running: {app_server_url} : {exc}'
                    )
                    sandbox_info.status = SandboxStatus.ERROR
                else:
                    sandbox_info.status = SandboxStatus.STARTING
                sandbox_info.exposed_urls = None
                sandbox_info.session_api_key = None
        return sandbox_info

    async def search_sandboxes(
        self,
        page_id: str | None = None,
        limit: int = 100,
    ) -> SandboxPage:
        """Search for sandboxes."""
        try:
            # Get all containers with our prefix
            all_containers = self.docker_client.containers.list(all=True)
            sandboxes = []

            for container in all_containers:
                if container.name and container.name.startswith(
                    self.container_name_prefix
                ):
                    sandbox_info = await self._container_to_checked_sandbox_info(
                        container
                    )
                    if sandbox_info:
                        sandboxes.append(sandbox_info)

            # Sort by creation time (newest first)
            sandboxes.sort(key=lambda x: x.created_at, reverse=True)

            # Apply pagination
            start_idx = 0
            if page_id:
                try:
                    start_idx = int(page_id)
                except ValueError:
                    start_idx = 0

            end_idx = start_idx + limit
            paginated_containers = sandboxes[start_idx:end_idx]

            # Determine next page ID
            next_page_id = None
            if end_idx < len(sandboxes):
                next_page_id = str(end_idx)

            return SandboxPage(items=paginated_containers, next_page_id=next_page_id)

        except APIError:
            return SandboxPage(items=[], next_page_id=None)

    async def get_sandbox(self, sandbox_id: str) -> SandboxInfo | None:
        """Get a single sandbox info."""
        try:
            if not sandbox_id.startswith(self.container_name_prefix):
                return None
            container = self.docker_client.containers.get(sandbox_id)
            return await self._container_to_checked_sandbox_info(container)
        except (NotFound, APIError):
            return None

    async def get_sandbox_by_session_api_key(
        self, session_api_key: str
    ) -> SandboxInfo | None:
        """Get a single sandbox by session API key."""
        try:
            # Get all containers with our prefix
            all_containers = self.docker_client.containers.list(all=True)

            for container in all_containers:
                if container.name and container.name.startswith(
                    self.container_name_prefix
                ):
                    # Check if this container has the matching session API key
                    env_vars = self._get_container_env_vars(container)
                    container_session_key = env_vars.get(SESSION_API_KEY_VARIABLE)

                    if container_session_key == session_api_key:
                        return await self._container_to_checked_sandbox_info(container)

            return None
        except (NotFound, APIError):
            return None

    async def start_sandbox(
        self,
        sandbox_spec_id: str | None = None,
        sandbox_id: str | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> SandboxInfo:
        """Start a new sandbox."""
        _logger.info(
            f'Starting sandbox: sandbox_id={sandbox_id}, spec_id={sandbox_spec_id}'
        )
        # Enforce sandbox limits by cleaning up old sandboxes
        await self.pause_old_sandboxes(self.max_num_sandboxes - 1)

        if sandbox_spec_id is None:
            sandbox_spec = await self.sandbox_spec_service.get_default_sandbox_spec()
        else:
            sandbox_spec_maybe = await self.sandbox_spec_service.get_sandbox_spec(
                sandbox_spec_id
            )
            if sandbox_spec_maybe is None:
                raise ValueError('Sandbox Spec not found')
            sandbox_spec = sandbox_spec_maybe

        # Generate a sandbox id if none was provided
        if sandbox_id is None:
            sandbox_id = base62.encodebytes(os.urandom(16))

        # Generate container name and session api key
        container_name = f'{self.container_name_prefix}{sandbox_id}'
        session_api_key = base62.encodebytes(os.urandom(32))

        # Prepare environment variables
        env_vars = sandbox_spec.initial_env.copy()
        if extra_env:
            env_vars.update(extra_env)
        env_vars[SESSION_API_KEY_VARIABLE] = session_api_key

        # Fetch CodeArtifact auth token if AWS credentials are available
        ca_token = self._get_codeartifact_token(env_vars)
        if ca_token:
            env_vars['CODEARTIFACT_AUTH_TOKEN'] = ca_token
        webhook_host = self.app_hostname or 'host.docker.internal'
        env_vars[WEBHOOK_CALLBACK_VARIABLE] = (
            f'http://{webhook_host}:{self.host_port}/api/v1/webhooks'
        )

        # Set CORS origins for remote browser access when web_url is configured.
        # This allows the agent-server container to accept requests from the
        # frontend when running OpenHands on a remote machine.
        if self.web_url:
            env_vars[ALLOW_CORS_ORIGINS_VARIABLE] = self.web_url

        # Set the base path for openvscode-server so it generates correct URLs
        # when proxied through the app server's /vscode-proxy/ route.
        if self.network:
            env_vars['OPENVSCODE_SERVER_BASE_PATH'] = f'/vscode-proxy/{container_name}'

        # Prepare port mappings and add port environment variables
        port_mappings = {}
        for exposed_port in self.exposed_ports:
            host_port = self._find_unused_port()
            port_mappings[exposed_port.container_port] = host_port
            # Add port as environment variable
            env_vars[exposed_port.name] = str(host_port)

        # Compute APP_URL from a WORKER port using the same URL logic
        # as _container_to_sandbox_info so sandbox scripts can reference the app URL.
        # Prefer a Traefik-routed WORKER port when Traefik is configured;
        # fall back to the first WORKER port otherwise.
        traefik_allowed = self.traefik_worker_ports
        app_url_fallback: str | None = None
        for ep in self.exposed_ports:
            if not ep.name.startswith('WORKER_'):
                continue
            worker_host_port = port_mappings.get(ep.container_port)
            if worker_host_port is None:
                continue
            if (
                self.traefik_domain
                and self.traefik_network
                and (not traefik_allowed or ep.name in traefik_allowed)
            ):
                if self.traefik_subdomain_prefix:
                    subdomain = f'{self.traefik_subdomain_prefix}-{sandbox_id}'
                else:
                    worker_name = ep.name.lower().replace('_', '-')
                    subdomain = f'{container_name}-{worker_name}'
                env_vars['APP_URL'] = (
                    f'{self.traefik_scheme}://{subdomain}.{self.traefik_domain}'
                )
                break
            else:
                if app_url_fallback is None:
                    app_url_fallback = self.container_url_pattern.format(
                        port=worker_host_port
                    )
        if 'APP_URL' not in env_vars and app_url_fallback:
            env_vars['APP_URL'] = app_url_fallback

        # Prepare labels
        labels = {
            'sandbox_spec_id': sandbox_spec.id,
            'sandbox_id': sandbox_id,
            **self.container_labels,
        }
        labels.update(
            self._build_traefik_labels(container_name, sandbox_id, self.exposed_ports)
        )

        # Prepare volumes
        volumes = {
            mount.host_path: {
                'bind': mount.container_path,
                'mode': mount.mode,
            }
            for mount in self.mounts
        }

        # Mount the shared package cache volume
        package_cache_volume = f'{self.resource_prefix}-package-cache'
        volumes[package_cache_volume] = {
            'bind': _PACKAGE_CACHE_PATH,
            'mode': 'rw',
        }

        try:
            # If /var/run/docker.sock is mounted, grant the container user
            # access to the socket by adding its owning group.
            group_add = _docker_socket_group(volumes)

            # Create and start the container
            _logger.info(
                f'Creating container: name={container_name}, image={sandbox_spec.id}, ports={port_mappings}'
            )
            container = self.docker_client.containers.run(  # type: ignore[call-overload, misc]
                image=sandbox_spec.id,
                command=sandbox_spec.command,  # Use default command from image
                remove=False,
                name=container_name,
                environment=env_vars,
                ports=port_mappings,
                volumes=volumes,
                working_dir=sandbox_spec.working_dir,
                labels=labels,
                detach=True,
                # Use Docker's tini init process to ensure proper signal handling and reaping of
                # zombie child processes.
                init=True,
                # Allow agent-server containers to resolve host.docker.internal
                # and other custom hostnames for LAN deployments
                extra_hosts=self.extra_hosts if self.extra_hosts else None,
                # Join shared network for container-to-container communication
                network=self.network if self.network else None,
                group_add=group_add if group_add else None,
                privileged=self.privileged if self.privileged else None,
            )

            # Ensure the package cache is writable by the non-root container
            # user (named volumes start root-owned).
            container.exec_run(
                f'chown openhands:openhands {_PACKAGE_CACHE_PATH}', user='root'
            )

            # Connect to the Traefik network so Traefik can discover
            # and route to this container's worker ports.
            if self.traefik_network:
                try:
                    traefik_net = self.docker_client.networks.get(self.traefik_network)
                    traefik_net.connect(container)
                except Exception as e:
                    _logger.error(
                        f'Failed to connect {container_name} to Traefik network: {e}'
                    )

            # When running in privileged mode, start dockerd inside the
            # container so that Docker-in-Docker works out of the box.
            if self.privileged:
                await self._start_dockerd(container)

            sandbox_info = await self._container_to_sandbox_info(container)
            assert sandbox_info is not None
            _logger.info(
                f'Container started: name={container_name}, status={sandbox_info.status}'
            )

            # In GitHub Codespaces, make WORKER ports publicly accessible
            # so that Codespace URLs work without authentication.
            self._schedule_codespace_port_visibility(port_mappings)

            # In GitHub Codespaces with a shared Docker network, start TCP
            # forwarders inside the devcontainer so the Codespace agent
            # detects the worker ports and auto-forwards them.
            if self.network and os.getenv('CODESPACE_NAME'):
                from openhands.app_server.sandbox.tcp_port_forwarder import (
                    get_tcp_port_forwarder_manager,
                )

                fwd_manager = get_tcp_port_forwarder_manager()
                if fwd_manager:
                    fwd_mappings = [
                        (
                            port_mappings[ep.container_port],
                            container_name,
                            ep.container_port,
                        )
                        for ep in self.exposed_ports
                        if ep.name.startswith('WORKER_')
                        and ep.container_port in port_mappings
                    ]
                    if fwd_mappings:
                        await fwd_manager.start_forwarders(container_name, fwd_mappings)

            # Record initial activity for idle-timeout tracking
            from openhands.app_server.idle_timeout_manager import (
                get_idle_timeout_manager,
            )

            manager = get_idle_timeout_manager()
            if manager:
                manager.touch(sandbox_info.id)

            return sandbox_info

        except APIError as e:
            _logger.error(f'Failed to start container {container_name}: {e}')
            raise SandboxError(f'Failed to start container: {e}')

    async def resume_sandbox(self, sandbox_id: str) -> bool:
        """Resume a paused sandbox."""
        # Enforce sandbox limits by cleaning up old sandboxes
        await self.pause_old_sandboxes(self.max_num_sandboxes - 1)

        try:
            if not sandbox_id.startswith(self.container_name_prefix):
                return False
            container = self.docker_client.containers.get(sandbox_id)

            if container.status == 'paused':
                container.unpause()
            elif container.status == 'exited':
                container.start()

            # Restart TCP port forwarders for resumed sandboxes
            if self.network and os.getenv('CODESPACE_NAME'):
                from openhands.app_server.sandbox.tcp_port_forwarder import (
                    get_tcp_port_forwarder_manager,
                )

                fwd_manager = get_tcp_port_forwarder_manager()
                if fwd_manager:
                    # Reload container attrs to get current port mappings
                    container.reload()
                    port_bindings = container.attrs.get('NetworkSettings', {}).get(
                        'Ports', {}
                    )
                    fwd_mappings = []
                    for ep in self.exposed_ports:
                        if not ep.name.startswith('WORKER_'):
                            continue
                        key = f'{ep.container_port}/tcp'
                        bindings = port_bindings.get(key)
                        if bindings:
                            host_port = int(bindings[0]['HostPort'])
                            fwd_mappings.append(
                                (host_port, sandbox_id, ep.container_port)
                            )
                    if fwd_mappings:
                        await fwd_manager.start_forwarders(sandbox_id, fwd_mappings)

            # Record activity for idle-timeout tracking on resume
            from openhands.app_server.idle_timeout_manager import (
                get_idle_timeout_manager,
            )

            manager = get_idle_timeout_manager()
            if manager:
                manager.touch(sandbox_id)

            return True
        except (NotFound, APIError):
            return False

    async def pause_sandbox(self, sandbox_id: str) -> bool:
        """Pause a running sandbox."""
        try:
            if not sandbox_id.startswith(self.container_name_prefix):
                return False
            container = self.docker_client.containers.get(sandbox_id)

            if container.status == 'running':
                container.pause()

            # Stop TCP port forwarders for paused sandboxes
            from openhands.app_server.sandbox.tcp_port_forwarder import (
                get_tcp_port_forwarder_manager,
            )

            fwd_manager = get_tcp_port_forwarder_manager()
            if fwd_manager:
                await fwd_manager.stop_forwarders(sandbox_id)

            # Stop idle-timeout tracking for paused sandboxes
            from openhands.app_server.idle_timeout_manager import (
                get_idle_timeout_manager,
            )

            manager = get_idle_timeout_manager()
            if manager:
                manager.remove(sandbox_id)

            return True
        except (NotFound, APIError):
            return False

    async def delete_sandbox(self, sandbox_id: str) -> bool:
        """Delete a sandbox."""
        try:
            if not sandbox_id.startswith(self.container_name_prefix):
                return False
            container = self.docker_client.containers.get(sandbox_id)

            # Stop TCP port forwarders before removing the container
            from openhands.app_server.sandbox.tcp_port_forwarder import (
                get_tcp_port_forwarder_manager,
            )

            fwd_manager = get_tcp_port_forwarder_manager()
            if fwd_manager:
                await fwd_manager.stop_forwarders(sandbox_id)

            # Stop the container if it's running
            if container.status in ['running', 'paused']:
                container.stop(timeout=10)

            # Remove the container
            container.remove()

            # Remove associated volume
            try:
                volume_name = f'{self.resource_prefix}-workspace-{sandbox_id}'
                volume = self.docker_client.volumes.get(volume_name)
                volume.remove()
            except (NotFound, APIError):
                # Volume might not exist or already removed
                pass

            return True
        except (NotFound, APIError):
            return False


class DockerSandboxServiceInjector(SandboxServiceInjector):
    """Dependency injector for docker sandbox services."""

    container_url_pattern: str = Field(
        default='http://localhost:{port}',
        description=(
            'URL pattern for exposed sandbox ports. Use {port} as placeholder. '
            'For remote access, set to your server IP (e.g., http://192.168.1.100:{port}). '
            'Configure via OH_SANDBOX_CONTAINER_URL_PATTERN environment variable.'
        ),
    )
    host_port: int = Field(
        default=3000,
        description=(
            'The port on which the main OpenHands app server is running. '
            'Used for webhook callbacks from agent-server containers. '
            'If running OpenHands on a non-default port, set this to match. '
            'Configure via OH_SANDBOX_HOST_PORT environment variable.'
        ),
    )
    resource_prefix: str = Field(
        default='openhands',
        description=(
            'Prefix for Docker resource names (volumes, containers) created by '
            'the sandbox service. Use a unique value per deployment when multiple '
            'OpenHands instances share the same Docker daemon. '
            'Configure via OH_SANDBOX__RESOURCE_PREFIX environment variable.'
        ),
    )
    container_name_prefix: str = 'oh-agent-server-'
    max_num_sandboxes: int = Field(
        default=5,
        description='Maximum number of sandboxes allowed to run simultaneously',
    )
    mounts: list[VolumeMount] = Field(default_factory=list)
    exposed_ports: list[ExposedPort] = Field(
        default_factory=lambda: [
            ExposedPort(
                name=AGENT_SERVER,
                description=(
                    'The port on which the agent server runs within the container'
                ),
                container_port=8000,
            ),
            ExposedPort(
                name=VSCODE,
                description=(
                    'The port on which the VSCode server runs within the container'
                ),
                container_port=8001,
            ),
            ExposedPort(
                name=WORKER_1,
                description=(
                    'The first port on which the agent should start application servers.'
                ),
                container_port=8080,
            ),
            ExposedPort(
                name=WORKER_2,
                description=(
                    'The second port on which the agent should start application servers.'
                ),
                container_port=8011,
            ),
            ExposedPort(
                name=WORKER_3,
                description=(
                    'The third port on which the agent should start application servers.'
                ),
                container_port=8012,
            ),
        ]
    )
    health_check_path: str | None = Field(
        default='/health',
        description=(
            'The url path in the sandbox agent server to check to '
            'determine whether the server is running'
        ),
    )
    extra_hosts: dict[str, str] = Field(
        default_factory=lambda: {'host.docker.internal': 'host-gateway'},
        description=(
            'Extra hostname mappings to add to agent-server containers. '
            'This allows containers to resolve hostnames like host.docker.internal '
            'for LAN deployments and MCP connections. '
            'Format: {"hostname": "ip_or_gateway"}'
        ),
    )
    network: str | None = Field(
        default=None,
        description=(
            'Docker network to attach sandbox containers to for container-to-container '
            'communication. When set, sandbox containers join this network and internal '
            'URLs are computed for server-to-server communication. '
            'Configure via OH_SANDBOX__NETWORK environment variable.'
        ),
    )
    app_hostname: str | None = Field(
        default=None,
        description=(
            'Hostname of the app server on the shared Docker network. '
            'Used for webhook callbacks from sandbox containers instead of host.docker.internal. '
            'Configure via OH_SANDBOX__APP_HOSTNAME environment variable.'
        ),
    )
    container_labels: dict[str, str] = Field(
        default_factory=dict,
        description=(
            'Additional labels to apply to sandbox containers. '
            'Useful for grouping containers in tools like Portainer or VS Code Docker extension. '
            'Configure via OH_SANDBOX_CONTAINER_LABELS environment variable. '
            'Format: {"com.docker.compose.project": "openhands"}'
        ),
    )
    startup_grace_seconds: int = Field(
        default=STARTUP_GRACE_SECONDS,
        description=(
            'Number of seconds were no response from the agent server is acceptable'
            'before it is considered an error'
        ),
    )
    idle_timeout_seconds: int = Field(
        default=1800,
        description=(
            'Seconds of inactivity before a sandbox is automatically paused. '
            'Set to 0 to disable idle timeout. Default is 1800 (30 minutes). '
            'Configure via OH_SANDBOX__IDLE_TIMEOUT_SECONDS environment variable.'
        ),
    )
    idle_warning_seconds: int = Field(
        default=300,
        description=(
            'Seconds before the idle timeout to send a warning to the frontend. '
            'Default is 300 (5 minutes). '
            'Configure via OH_SANDBOX__IDLE_WARNING_SECONDS environment variable.'
        ),
    )
    privileged: bool = Field(
        default=False,
        description=(
            'Run sandbox containers in privileged mode. '
            'Required for Docker-in-Docker (DinD) so that the sandbox can '
            'run its own Docker daemon and use volume mounts that reference '
            'the sandbox filesystem. '
            'Configure via OH_SANDBOX__PRIVILEGED environment variable.'
        ),
    )
    dind_registry_cache: bool = Field(
        default=False,
        description=(
            'Enable a pull-through registry cache for Docker-in-Docker sandboxes. '
            'Requires privileged mode. '
            'Configure via OH_SANDBOX__DIND_REGISTRY_CACHE environment variable.'
        ),
    )
    dind_registry_port: int = Field(
        default=5555,
        description=(
            'Port for the local pull-through registry cache. '
            'Configure via OH_SANDBOX__DIND_REGISTRY_PORT environment variable.'
        ),
    )
    dind_registry_mirror_url: str | None = Field(
        default=None,
        description=(
            'URL of an external pull-through registry cache (e.g. one defined '
            'in docker-compose.yml). When set, the programmatic registry cache '
            'is skipped and this URL is used directly as the --registry-mirror '
            'for dockerd inside privileged sandboxes. '
            'Configure via OH_SANDBOX__DIND_REGISTRY_MIRROR_URL environment variable.'
        ),
    )
    dind_registry_mirrors: dict[str, str] = Field(
        default_factory=dict,
        description=(
            'Mapping of upstream registry hostnames to local pull-through cache URLs. '
            'Used to configure containerd host mirrors inside privileged sandboxes '
            'so that pulls from non-Docker-Hub registries are cached locally. '
            'Example: {"myregistry.example.com": "http://openhands-registry-cache:5000"} '
            'Configure via OH_SANDBOX__DIND_REGISTRY_MIRRORS environment variable '
            'as comma-separated host=url pairs.'
        ),
    )
    traefik_network: str | None = Field(
        default=None,
        description=(
            'Traefik Docker network name to attach sandbox containers to for '
            'automatic Traefik discovery and subdomain routing. '
            'Configure via OH_SANDBOX__TRAEFIK_NETWORK environment variable.'
        ),
    )
    traefik_domain: str | None = Field(
        default=None,
        description=(
            'Base domain for Traefik subdomain routing of sandbox worker ports. '
            'Each worker gets a subdomain like {container}-worker-2.{domain}. '
            'Requires a wildcard DNS record pointing to the Traefik host. '
            'Configure via OH_SANDBOX__TRAEFIK_DOMAIN environment variable.'
        ),
    )
    traefik_entrypoints: str = Field(
        default='web',
        description=(
            'Traefik entrypoints for sandbox container routers. '
            'Configure via OH_SANDBOX__TRAEFIK_ENTRYPOINTS environment variable.'
        ),
    )
    traefik_certresolver: str | None = Field(
        default=None,
        description=(
            'Traefik certificate resolver for TLS on sandbox container routers. '
            'When set, TLS is enabled and certificates are automatically provisioned. '
            'Configure via OH_SANDBOX__TRAEFIK_CERTRESOLVER environment variable.'
        ),
    )
    traefik_worker_ports: list[str] | None = Field(
        default=None,
        description=(
            'List of WORKER port names to route via Traefik '
            '(e.g. WORKER_2). When not set, all WORKER_* ports get Traefik routes. '
            'Configure via OH_SANDBOX__TRAEFIK_WORKER_PORTS environment variable.'
        ),
    )
    traefik_subdomain_prefix: str | None = Field(
        default=None,
        description=(
            'Prefix for Traefik subdomain routing. When set, sandbox URLs use '
            'the pattern {prefix}-{sandbox_id}.{domain} instead of '
            '{container_name}-{worker_name}.{domain}. '
            'Configure via OH_SANDBOX__TRAEFIK_SUBDOMAIN_PREFIX environment variable.'
        ),
    )
    traefik_scheme: str = Field(
        default='https',
        description=(
            'URL scheme for Traefik-routed sandbox URLs (http or https). '
            'Defaults to https. '
            'Configure via OH_SANDBOX__TRAEFIK_SCHEME environment variable.'
        ),
    )

    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[SandboxService, None]:
        # Define inline to prevent circular lookup
        from openhands.app_server.config import (
            get_global_config,
            get_httpx_client,
            get_sandbox_spec_service,
        )

        # Get web_url from global config for CORS support
        config = get_global_config()
        web_url = config.web_url

        async with (
            get_httpx_client(state) as httpx_client,
            get_sandbox_spec_service(state) as sandbox_spec_service,
        ):
            yield DockerSandboxService(
                sandbox_spec_service=sandbox_spec_service,
                container_name_prefix=self.container_name_prefix,
                host_port=self.host_port,
                container_url_pattern=self.container_url_pattern,
                mounts=self.mounts,
                exposed_ports=self.exposed_ports,
                health_check_path=self.health_check_path,
                httpx_client=httpx_client,
                max_num_sandboxes=self.max_num_sandboxes,
                resource_prefix=self.resource_prefix,
                web_url=web_url,
                extra_hosts=self.extra_hosts,
                network=self.network,
                app_hostname=self.app_hostname,
                container_labels=self.container_labels,
                startup_grace_seconds=self.startup_grace_seconds,
                privileged=self.privileged,
                registry_mirror_url=self.dind_registry_mirror_url
                or getattr(self, '_registry_mirror_url', None),
                registry_mirrors=self.dind_registry_mirrors,
                traefik_network=self.traefik_network,
                traefik_domain=self.traefik_domain,
                traefik_entrypoints=self.traefik_entrypoints,
                traefik_certresolver=self.traefik_certresolver,
                traefik_worker_ports=self.traefik_worker_ports,
                traefik_subdomain_prefix=self.traefik_subdomain_prefix,
                traefik_scheme=self.traefik_scheme,
            )
