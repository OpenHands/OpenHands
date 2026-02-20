"""Pull-through registry cache for Docker-in-Docker sandboxes.

Runs a ``registry:2`` container configured as a pull-through cache for Docker
Hub.  Each sandbox's internal ``dockerd`` is pointed at this mirror so that
images are fetched from the local cache whenever possible.
"""

import logging

import docker
from docker.errors import APIError, NotFound

_logger = logging.getLogger(__name__)

_CONTAINER_NAME = 'openhands-registry-cache'
_VOLUME_NAME = 'openhands-registry-cache'
_IMAGE = 'registry:2'


class RegistryCacheManager:
    """Manages a local pull-through Docker registry cache container."""

    def __init__(self, port: int = 5555) -> None:
        self.port = port
        self._docker: docker.DockerClient | None = None

    @property
    def _client(self) -> docker.DockerClient:
        if self._docker is None:
            self._docker = docker.from_env()
        return self._docker

    def ensure_running(self) -> str:
        """Ensure the registry cache container is running.

        Returns the mirror URL (e.g. ``http://host.docker.internal:5555``).
        If the container already exists and is running, this is a no-op.
        """
        try:
            container = self._client.containers.get(_CONTAINER_NAME)
            if container.status == 'running':
                _logger.info('Registry cache already running')
                return self._mirror_url()
            # Exists but not running — remove and recreate
            container.remove(force=True)
        except NotFound:
            pass

        # Pull image if needed
        try:
            self._client.images.get(_IMAGE)
        except docker.errors.ImageNotFound:
            _logger.info(f'Pulling {_IMAGE} for registry cache...')
            self._client.images.pull(_IMAGE)

        _logger.info(f'Starting registry cache container on port {self.port}...')
        self._client.containers.run(  # type: ignore[call-overload]
            image=_IMAGE,
            name=_CONTAINER_NAME,
            detach=True,
            restart_policy={'Name': 'unless-stopped'},
            ports={'5000/tcp': self.port},
            environment={
                'REGISTRY_PROXY_REMOTEURL': 'https://registry-1.docker.io',
            },
            volumes={
                _VOLUME_NAME: {'bind': '/var/lib/registry', 'mode': 'rw'},
            },
        )
        _logger.info('Registry cache container started')
        return self._mirror_url()

    def stop(self) -> None:
        """Stop and remove the registry cache container."""
        try:
            container = self._client.containers.get(_CONTAINER_NAME)
            container.remove(force=True)
            _logger.info('Registry cache container removed')
        except (NotFound, APIError):
            pass

    def _mirror_url(self) -> str:
        return f'http://host.docker.internal:{self.port}'
