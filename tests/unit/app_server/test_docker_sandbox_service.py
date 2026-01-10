"""Tests for DockerSandboxService.

This module tests the Docker sandbox service implementation, focusing on:
- Container lifecycle management (start, pause, resume, delete)
- Container search and retrieval with filtering and pagination
- Data transformation from Docker containers to SandboxInfo objects
- Health checking and URL generation
- Error handling for Docker API failures
- Edge cases with malformed container data
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from docker.errors import APIError, NotFound

from openhands.app_server.errors import SandboxError
from openhands.app_server.sandbox.docker_sandbox_service import (
    DockerSandboxService,
    ExposedPort,
    VolumeMount,
)
from openhands.app_server.sandbox.sandbox_models import (
    AGENT_SERVER,
    VSCODE,
    SandboxPage,
    SandboxStatus,
)


@pytest.fixture
def mock_docker_client():
    """Mock Docker client for testing."""
    mock_client = MagicMock()
    return mock_client


@pytest.fixture
def mock_sandbox_spec_service():
    """Mock SandboxSpecService for testing."""
    mock_service = AsyncMock()
    mock_spec = MagicMock()
    mock_spec.id = 'test-image:latest'
    mock_spec.initial_env = {'TEST_VAR': 'test_value'}
    mock_spec.working_dir = '/workspace'
    mock_service.get_default_sandbox_spec.return_value = mock_spec
    mock_service.get_sandbox_spec.return_value = mock_spec
    return mock_service


@pytest.fixture
def mock_httpx_client():
    """Mock httpx AsyncClient for testing."""
    client = AsyncMock(spec=httpx.AsyncClient)
    # Configure the mock response
    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock()
    client.get.return_value = mock_response
    return client


@pytest.fixture
def service(mock_sandbox_spec_service, mock_httpx_client, mock_docker_client):
    """Create DockerSandboxService instance for testing."""
    return DockerSandboxService(
        sandbox_spec_service=mock_sandbox_spec_service,
        container_name_prefix='oh-test-',
        host_port=3000,
        container_url_pattern='http://localhost:{port}',
        mounts=[
            VolumeMount(host_path='/tmp/test', container_path='/workspace', mode='rw')
        ],
        exposed_ports=[
            ExposedPort(
                name=AGENT_SERVER, description='Agent server', container_port=8000
            ),
            ExposedPort(name=VSCODE, description='VSCode server', container_port=8001),
        ],
        health_check_path='/health',
        httpx_client=mock_httpx_client,
        max_num_sandboxes=3,
        docker_client=mock_docker_client,
    )


@pytest.fixture
def mock_running_container():
    """Create a mock running Docker container."""
    container = MagicMock()
    container.name = 'oh-test-abc123'
    container.status = 'running'
    container.image.tags = ['spec456']
    container.attrs = {
        'Created': '2024-01-15T10:30:00.000000000Z',
        'Config': {
            'Env': ['OH_SESSION_API_KEYS_0=session_key_123', 'OTHER_VAR=other_value'],
            'WorkingDir': '/workspace',
        },
        'NetworkSettings': {
            'Ports': {
                '8000/tcp': [{'HostPort': '12345'}],
                '8001/tcp': [{'HostPort': '12346'}],
            }
        },
    }
    return container


@pytest.fixture
def mock_paused_container():
    """Create a mock paused Docker container."""
    container = MagicMock()
    container.name = 'oh-test-def456'
    container.status = 'paused'
    container.image.tags = ['spec456']
    container.attrs = {
        'Created': '2024-01-15T10:30:00.000000000Z',
        'Config': {'Env': []},
        'NetworkSettings': {'Ports': {}},
    }
    return container


@pytest.fixture
def mock_exited_container():
    """Create a mock exited Docker container."""
    container = MagicMock()
    container.name = 'oh-test-ghi789'
    container.status = 'exited'
    container.labels = {'created_by_user_id': 'user123', 'sandbox_spec_id': 'spec456'}
    container.attrs = {
        'Created': '2024-01-15T10:30:00.000000000Z',
        'Config': {'Env': []},
        'NetworkSettings': {'Ports': {}},
    }
    return container


class TestDockerSandboxService:
    """Test cases for DockerSandboxService."""

    async def test_search_sandboxes_success(
        self, service, mock_running_container, mock_paused_container
    ):
        """Test successful search for sandboxes."""
        # Setup
        service.docker_client.containers.list.return_value = [
            mock_running_container,
            mock_paused_container,
        ]
        service.httpx_client.get.return_value.raise_for_status.return_value = None

        # Execute
        result = await service.search_sandboxes()

        # Verify
        assert isinstance(result, SandboxPage)
        assert len(result.items) == 2
        assert result.next_page_id is None

        # Verify running container
        running_sandbox = next(
            s for s in result.items if s.status == SandboxStatus.RUNNING
        )
        assert running_sandbox.id == 'oh-test-abc123'
        assert running_sandbox.created_by_user_id is None
        assert running_sandbox.sandbox_spec_id == 'spec456'
        assert running_sandbox.session_api_key == 'session_key_123'
        assert len(running_sandbox.exposed_urls) == 2

        # Verify paused container
        paused_sandbox = next(
            s for s in result.items if s.status == SandboxStatus.PAUSED
        )
        assert paused_sandbox.id == 'oh-test-def456'
        assert paused_sandbox.session_api_key is None
        assert paused_sandbox.exposed_urls is None

    async def test_search_sandboxes_pagination(self, service):
        """Test pagination functionality."""
        # Setup - create multiple containers
        containers = []
        for i in range(5):
            container = MagicMock()
            container.name = f'oh-test-container{i}'
            container.status = 'running'
            container.image.tags = ['spec456']
            container.attrs = {
                'Created': f'2024-01-{15 + i:02d}T10:30:00.000000000Z',
                'Config': {
                    'Env': [
                        f'OH_SESSION_API_KEYS_0=session_key_{i}',
                        f'OTHER_VAR=value_{i}',
                    ]
                },
                'NetworkSettings': {'Ports': {}},
            }
            containers.append(container)

        service.docker_client.containers.list.return_value = containers
        service.httpx_client.get.return_value.raise_for_status.return_value = None

        # Execute - first page
        result = await service.search_sandboxes(limit=3)

        # Verify first page
        assert len(result.items) == 3
        assert result.next_page_id == '3'

        # Execute - second page
        result = await service.search_sandboxes(page_id='3', limit=3)

        # Verify second page
        assert len(result.items) == 2
        assert result.next_page_id is None

    async def test_search_sandboxes_invalid_page_id(
        self, service, mock_running_container
    ):
        """Test handling of invalid page ID."""
        # Setup
        service.docker_client.containers.list.return_value = [mock_running_container]
        service.httpx_client.get.return_value.raise_for_status.return_value = None

        # Execute
        result = await service.search_sandboxes(page_id='invalid')

        # Verify - should start from beginning
        assert len(result.items) == 1

    async def test_search_sandboxes_docker_api_error(self, service):
        """Test handling of Docker API errors."""
        # Setup
        service.docker_client.containers.list.side_effect = APIError(
            'Docker daemon error'
        )

        # Execute
        result = await service.search_sandboxes()

        # Verify
        assert isinstance(result, SandboxPage)
        assert len(result.items) == 0
        assert result.next_page_id is None

    async def test_search_sandboxes_filters_by_prefix(self, service):
        """Test that search filters containers by name prefix."""
        # Setup
        matching_container = MagicMock()
        matching_container.name = 'oh-test-abc123'
        matching_container.status = 'running'
        matching_container.image.tags = ['spec456']
        matching_container.attrs = {
            'Created': '2024-01-15T10:30:00.000000000Z',
            'Config': {
                'Env': [
                    'OH_SESSION_API_KEYS_0=matching_session_key',
                    'OTHER_VAR=matching_value',
                ]
            },
            'NetworkSettings': {'Ports': {}},
        }

        non_matching_container = MagicMock()
        non_matching_container.name = 'other-container'
        non_matching_container.status = 'running'
        non_matching_container.image.tags = (['other'],)

        service.docker_client.containers.list.return_value = [
            matching_container,
            non_matching_container,
        ]
        service.httpx_client.get.return_value.raise_for_status.return_value = None

        # Execute
        result = await service.search_sandboxes()

        # Verify - only matching container should be included
        assert len(result.items) == 1
        assert result.items[0].id == 'oh-test-abc123'

    async def test_get_sandbox_success(self, service, mock_running_container):
        """Test successful retrieval of specific sandbox."""
        # Setup
        service.docker_client.containers.get.return_value = mock_running_container
        service.httpx_client.get.return_value.raise_for_status.return_value = None

        # Execute
        result = await service.get_sandbox('oh-test-abc123')

        # Verify
        assert result is not None
        assert result.id == 'oh-test-abc123'
        assert result.status == SandboxStatus.RUNNING

        # Verify Docker client was called correctly
        service.docker_client.containers.get.assert_called_once_with('oh-test-abc123')

    async def test_get_sandbox_not_found(self, service):
        """Test handling when sandbox is not found."""
        # Setup
        service.docker_client.containers.get.side_effect = NotFound(
            'Container not found'
        )

        # Execute
        result = await service.get_sandbox('oh-test-nonexistent')

        # Verify
        assert result is None

    async def test_get_sandbox_wrong_prefix(self, service):
        """Test handling when sandbox ID doesn't match prefix."""
        # Execute
        result = await service.get_sandbox('wrong-prefix-abc123')

        # Verify
        assert result is None
        service.docker_client.containers.get.assert_not_called()

    async def test_get_sandbox_api_error(self, service):
        """Test handling of Docker API errors during get."""
        # Setup
        service.docker_client.containers.get.side_effect = APIError(
            'Docker daemon error'
        )

        # Execute
        result = await service.get_sandbox('oh-test-abc123')

        # Verify
        assert result is None

    @patch('openhands.app_server.sandbox.docker_sandbox_service.base62.encodebytes')
    @patch('os.urandom')
    async def test_start_sandbox_success(self, mock_urandom, mock_encodebytes, service):
        """Test successful sandbox startup."""
        # Setup
        mock_urandom.side_effect = [b'container_id', b'session_key']
        mock_encodebytes.side_effect = ['test_container_id', 'test_session_key']

        mock_container = MagicMock()
        mock_container.name = 'oh-test-test_container_id'
        mock_container.status = 'running'
        mock_container.image.tags = ['test-image:latest']
        mock_container.attrs = {
            'Created': '2024-01-15T10:30:00.000000000Z',
            'Config': {
                'Env': ['OH_SESSION_API_KEYS_0=test_session_key', 'TEST_VAR=test_value']
            },
            'NetworkSettings': {'Ports': {}},
        }

        service.docker_client.containers.run.return_value = mock_container

        with (
            patch.object(service, '_find_unused_port', side_effect=[12345, 12346]),
            patch.object(
                service, 'pause_old_sandboxes', return_value=[]
            ) as mock_cleanup,
        ):
            # Execute
            result = await service.start_sandbox()

        # Verify
        assert result is not None
        assert result.id == 'oh-test-test_container_id'

        # Verify cleanup was called with the correct limit
        mock_cleanup.assert_called_once_with(2)

        # Verify container was created with correct parameters
        service.docker_client.containers.run.assert_called_once()
        call_args = service.docker_client.containers.run.call_args

        assert call_args[1]['image'] == 'test-image:latest'
        assert call_args[1]['name'] == 'oh-test-test_container_id'
        assert 'OH_SESSION_API_KEYS_0' in call_args[1]['environment']
        assert (
            call_args[1]['environment']['OH_SESSION_API_KEYS_0'] == 'test_session_key'
        )
        assert call_args[1]['ports'] == {8000: 12345, 8001: 12346}
        assert call_args[1]['working_dir'] == '/workspace'
        assert call_args[1]['detach'] is True

    async def test_start_sandbox_with_spec_id(self, service, mock_sandbox_spec_service):
        """Test starting sandbox with specific spec ID."""
        # Setup
        mock_container = MagicMock()
        mock_container.name = 'oh-test-abc123'
        mock_container.status = 'running'
        mock_container.image.tags = ['spec456']
        mock_container.attrs = {
            'Created': '2024-01-15T10:30:00.000000000Z',
            'Config': {
                'Env': [
                    'OH_SESSION_API_KEYS_0=test_session_key',
                    'OTHER_VAR=test_value',
                ]
            },
            'NetworkSettings': {'Ports': {}},
        }
        service.docker_client.containers.run.return_value = mock_container

        with (
            patch.object(service, '_find_unused_port', return_value=12345),
            patch.object(service, 'pause_old_sandboxes', return_value=[]),
        ):
            # Execute
            await service.start_sandbox(sandbox_spec_id='custom-spec')

        # Verify
        mock_sandbox_spec_service.get_sandbox_spec.assert_called_once_with(
            'custom-spec'
        )

    async def test_start_sandbox_spec_not_found(
        self, service, mock_sandbox_spec_service
    ):
        """Test starting sandbox with non-existent spec ID."""
        # Setup
        mock_sandbox_spec_service.get_sandbox_spec.return_value = None

        # Execute & Verify
        with (
            patch.object(service, 'pause_old_sandboxes', return_value=[]),
            pytest.raises(ValueError, match='Sandbox Spec not found'),
        ):
            await service.start_sandbox(sandbox_spec_id='nonexistent')

    async def test_start_sandbox_docker_error(self, service):
        """Test handling of Docker errors during sandbox startup."""
        # Setup
        service.docker_client.containers.run.side_effect = APIError(
            'Failed to create container'
        )

        with (
            patch.object(service, '_find_unused_port', return_value=12345),
            patch.object(service, 'pause_old_sandboxes', return_value=[]),
            pytest.raises(SandboxError, match='Failed to start container'),
        ):
            await service.start_sandbox()

    @patch('openhands.app_server.sandbox.docker_sandbox_service.base62.encodebytes')
    @patch('os.urandom')
    async def test_start_sandbox_with_extra_hosts(
        self,
        mock_urandom,
        mock_encodebytes,
        mock_sandbox_spec_service,
        mock_httpx_client,
        mock_docker_client,
    ):
        """Test that extra_hosts are passed to container creation."""
        # Setup
        mock_urandom.side_effect = [b'container_id', b'session_key']
        mock_encodebytes.side_effect = ['test_container_id', 'test_session_key']

        mock_container = MagicMock()
        mock_container.name = 'oh-test-test_container_id'
        mock_container.status = 'running'
        mock_container.image.tags = ['test-image:latest']
        mock_container.attrs = {
            'Created': '2024-01-15T10:30:00.000000000Z',
            'Config': {
                'Env': ['OH_SESSION_API_KEYS_0=test_session_key', 'TEST_VAR=test_value']
            },
            'NetworkSettings': {'Ports': {}},
        }
        mock_docker_client.containers.run.return_value = mock_container

        # Create service with extra_hosts
        service_with_extra_hosts = DockerSandboxService(
            sandbox_spec_service=mock_sandbox_spec_service,
            container_name_prefix='oh-test-',
            host_port=3000,
            container_url_pattern='http://localhost:{port}',
            mounts=[],
            exposed_ports=[
                ExposedPort(
                    name=AGENT_SERVER, description='Agent server', container_port=8000
                ),
            ],
            health_check_path='/health',
            httpx_client=mock_httpx_client,
            max_num_sandboxes=3,
            extra_hosts={
                'host.docker.internal': 'host-gateway',
                'custom.host': '192.168.1.100',
            },
            docker_client=mock_docker_client,
        )

        with (
            patch.object(
                service_with_extra_hosts, '_find_unused_port', return_value=12345
            ),
            patch.object(
                service_with_extra_hosts, 'pause_old_sandboxes', return_value=[]
            ),
        ):
            # Execute
            await service_with_extra_hosts.start_sandbox()

        # Verify extra_hosts was passed to container creation
        mock_docker_client.containers.run.assert_called_once()
        call_args = mock_docker_client.containers.run.call_args
        assert call_args[1]['extra_hosts'] == {
            'host.docker.internal': 'host-gateway',
            'custom.host': '192.168.1.100',
        }

    @patch('openhands.app_server.sandbox.docker_sandbox_service.base62.encodebytes')
    @patch('os.urandom')
    async def test_start_sandbox_without_extra_hosts(
        self,
        mock_urandom,
        mock_encodebytes,
        mock_sandbox_spec_service,
        mock_httpx_client,
        mock_docker_client,
    ):
        """Test that extra_hosts is None when not configured."""
        # Setup
        mock_urandom.side_effect = [b'container_id', b'session_key']
        mock_encodebytes.side_effect = ['test_container_id', 'test_session_key']

        mock_container = MagicMock()
        mock_container.name = 'oh-test-test_container_id'
        mock_container.status = 'running'
        mock_container.image.tags = ['test-image:latest']
        mock_container.attrs = {
            'Created': '2024-01-15T10:30:00.000000000Z',
            'Config': {
                'Env': ['OH_SESSION_API_KEYS_0=test_session_key', 'TEST_VAR=test_value']
            },
            'NetworkSettings': {'Ports': {}},
        }
        mock_docker_client.containers.run.return_value = mock_container

        # Create service without extra_hosts (empty dict)
        service_without_extra_hosts = DockerSandboxService(
            sandbox_spec_service=mock_sandbox_spec_service,
            container_name_prefix='oh-test-',
            host_port=3000,
            container_url_pattern='http://localhost:{port}',
            mounts=[],
            exposed_ports=[
                ExposedPort(
                    name=AGENT_SERVER, description='Agent server', container_port=8000
                ),
            ],
            health_check_path='/health',
            httpx_client=mock_httpx_client,
            max_num_sandboxes=3,
            extra_hosts={},
            docker_client=mock_docker_client,
        )

        with (
            patch.object(
                service_without_extra_hosts, '_find_unused_port', return_value=12345
            ),
            patch.object(
                service_without_extra_hosts, 'pause_old_sandboxes', return_value=[]
            ),
        ):
            # Execute
            await service_without_extra_hosts.start_sandbox()

        # Verify extra_hosts is None when empty dict is provided
        mock_docker_client.containers.run.assert_called_once()
        call_args = mock_docker_client.containers.run.call_args
        assert call_args[1]['extra_hosts'] is None

    async def test_resume_sandbox_from_paused(self, service):
        """Test resuming a paused sandbox."""
        # Setup
        mock_container = MagicMock()
        mock_container.status = 'paused'
        service.docker_client.containers.get.return_value = mock_container

        with patch.object(
            service, 'pause_old_sandboxes', return_value=[]
        ) as mock_cleanup:
            # Execute
            result = await service.resume_sandbox('oh-test-abc123')

        # Verify
        assert result is True
        mock_container.unpause.assert_called_once()
        mock_container.start.assert_not_called()
        # Verify cleanup was called with the correct limit
        mock_cleanup.assert_called_once_with(2)

    async def test_resume_sandbox_from_exited(self, service):
        """Test resuming an exited sandbox."""
        # Setup
        mock_container = MagicMock()
        mock_container.status = 'exited'
        service.docker_client.containers.get.return_value = mock_container

        with patch.object(
            service, 'pause_old_sandboxes', return_value=[]
        ) as mock_cleanup:
            # Execute
            result = await service.resume_sandbox('oh-test-abc123')

        # Verify
        assert result is True
        mock_container.start.assert_called_once()
        mock_container.unpause.assert_not_called()
        # Verify cleanup was called with the correct limit
        mock_cleanup.assert_called_once_with(2)

    async def test_resume_sandbox_wrong_prefix(self, service):
        """Test resuming sandbox with wrong prefix."""
        with patch.object(
            service, 'pause_old_sandboxes', return_value=[]
        ) as mock_cleanup:
            # Execute
            result = await service.resume_sandbox('wrong-prefix-abc123')

        # Verify
        assert result is False
        service.docker_client.containers.get.assert_not_called()
        # Verify cleanup was still called
        mock_cleanup.assert_called_once_with(2)

    async def test_resume_sandbox_not_found(self, service):
        """Test resuming non-existent sandbox."""
        # Setup
        service.docker_client.containers.get.side_effect = NotFound(
            'Container not found'
        )

        with patch.object(
            service, 'pause_old_sandboxes', return_value=[]
        ) as mock_cleanup:
            # Execute
            result = await service.resume_sandbox('oh-test-abc123')

        # Verify
        assert result is False
        # Verify cleanup was still called
        mock_cleanup.assert_called_once_with(2)

    async def test_pause_sandbox_success(self, service):
        """Test pausing a running sandbox."""
        # Setup
        mock_container = MagicMock()
        mock_container.status = 'running'
        service.docker_client.containers.get.return_value = mock_container

        # Execute
        result = await service.pause_sandbox('oh-test-abc123')

        # Verify
        assert result is True
        mock_container.pause.assert_called_once()

    async def test_pause_sandbox_not_running(self, service):
        """Test pausing a non-running sandbox."""
        # Setup
        mock_container = MagicMock()
        mock_container.status = 'paused'
        service.docker_client.containers.get.return_value = mock_container

        # Execute
        result = await service.pause_sandbox('oh-test-abc123')

        # Verify
        assert result is True
        mock_container.pause.assert_not_called()

    async def test_delete_sandbox_success(self, service):
        """Test successful sandbox deletion."""
        # Setup
        mock_container = MagicMock()
        mock_container.status = 'running'
        service.docker_client.containers.get.return_value = mock_container

        mock_volume = MagicMock()
        service.docker_client.volumes.get.return_value = mock_volume

        # Execute
        result = await service.delete_sandbox('oh-test-abc123')

        # Verify
        assert result is True
        mock_container.stop.assert_called_once_with(timeout=10)
        mock_container.remove.assert_called_once()
        service.docker_client.volumes.get.assert_called_once_with(
            'openhands-workspace-oh-test-abc123'
        )
        mock_volume.remove.assert_called_once()

    async def test_delete_sandbox_volume_not_found(self, service):
        """Test sandbox deletion when volume doesn't exist."""
        # Setup
        mock_container = MagicMock()
        mock_container.status = 'exited'
        service.docker_client.containers.get.return_value = mock_container
        service.docker_client.volumes.get.side_effect = NotFound('Volume not found')

        # Execute
        result = await service.delete_sandbox('oh-test-abc123')

        # Verify
        assert result is True
        mock_container.stop.assert_not_called()  # Already stopped
        mock_container.remove.assert_called_once()

    def test_find_unused_port(self, service):
        """Test finding an unused port."""
        # Execute
        port = service._find_unused_port()

        # Verify
        assert isinstance(port, int)
        assert 1024 <= port <= 65535

    def test_docker_status_to_sandbox_status(self, service):
        """Test Docker status to SandboxStatus conversion."""
        # Test all mappings
        assert (
            service._docker_status_to_sandbox_status('running') == SandboxStatus.RUNNING
        )
        assert (
            service._docker_status_to_sandbox_status('paused') == SandboxStatus.PAUSED
        )
        assert (
            service._docker_status_to_sandbox_status('exited') == SandboxStatus.PAUSED
        )
        assert (
            service._docker_status_to_sandbox_status('created')
            == SandboxStatus.STARTING
        )
        assert (
            service._docker_status_to_sandbox_status('restarting')
            == SandboxStatus.STARTING
        )
        assert (
            service._docker_status_to_sandbox_status('removing')
            == SandboxStatus.MISSING
        )
        assert service._docker_status_to_sandbox_status('dead') == SandboxStatus.ERROR
        assert (
            service._docker_status_to_sandbox_status('unknown') == SandboxStatus.ERROR
        )

    def test_get_container_env_vars(self, service):
        """Test environment variable extraction from container."""
        # Setup
        mock_container = MagicMock()
        mock_container.attrs = {
            'Config': {
                'Env': [
                    'VAR1=value1',
                    'VAR2=value2',
                    'VAR_NO_VALUE',
                    'VAR3=value=with=equals',
                ]
            }
        }

        # Execute
        result = service._get_container_env_vars(mock_container)

        # Verify
        assert result == {
            'VAR1': 'value1',
            'VAR2': 'value2',
            'VAR_NO_VALUE': None,
            'VAR3': 'value=with=equals',
        }

    async def test_container_to_sandbox_info_running(
        self, service, mock_running_container
    ):
        """Test conversion of running container to SandboxInfo."""
        # Execute
        result = await service._container_to_sandbox_info(mock_running_container)

        # Verify
        assert result is not None
        assert result.id == 'oh-test-abc123'
        assert result.created_by_user_id is None
        assert result.sandbox_spec_id == 'spec456'
        assert result.status == SandboxStatus.RUNNING
        assert result.session_api_key == 'session_key_123'
        assert len(result.exposed_urls) == 2

        # Check exposed URLs
        agent_url = next(url for url in result.exposed_urls if url.name == AGENT_SERVER)
        assert agent_url.url == 'http://localhost:12345'

        vscode_url = next(url for url in result.exposed_urls if url.name == VSCODE)
        assert (
            vscode_url.url
            == 'http://localhost:12346/?tkn=session_key_123&folder=/workspace'
        )

    async def test_container_to_sandbox_info_invalid_created_time(self, service):
        """Test conversion with invalid creation timestamp."""
        # Setup
        container = MagicMock()
        container.name = 'oh-test-abc123'
        container.status = 'running'
        container.image.tags = ['spec456']
        container.attrs = {
            'Created': 'invalid-timestamp',
            'Config': {
                'Env': [
                    'OH_SESSION_API_KEYS_0=test_session_key',
                    'OTHER_VAR=test_value',
                ]
            },
            'NetworkSettings': {'Ports': {}},
        }

        # Execute
        result = await service._container_to_sandbox_info(container)

        # Verify - should use current time as fallback
        assert result is not None
        assert isinstance(result.created_at, datetime)

    @patch(
        'openhands.app_server.utils.docker_utils.is_running_in_docker',
        return_value=True,
    )
    async def test_container_to_checked_sandbox_info_health_check_success(
        self, mock_is_docker, service, mock_running_container
    ):
        """Test health check success when running in Docker."""
        # Setup
        service.httpx_client.get.return_value.raise_for_status.return_value = None

        # Execute
        result = await service._container_to_checked_sandbox_info(
            mock_running_container
        )

        # Verify
        assert result is not None
        assert result.status == SandboxStatus.RUNNING
        assert result.exposed_urls is not None
        assert result.session_api_key == 'session_key_123'

        # Verify health check was called with Docker-internal URL
        service.httpx_client.get.assert_called_once_with(
            'http://host.docker.internal:12345/health'
        )

    @patch(
        'openhands.app_server.utils.docker_utils.is_running_in_docker',
        return_value=False,
    )
    async def test_container_to_checked_sandbox_info_health_check_success_not_in_docker(
        self, mock_is_docker, service, mock_running_container
    ):
        """Test health check success when not running in Docker."""
        # Setup
        service.httpx_client.get.return_value.raise_for_status.return_value = None

        # Execute
        result = await service._container_to_checked_sandbox_info(
            mock_running_container
        )

        # Verify
        assert result is not None
        assert result.status == SandboxStatus.RUNNING
        assert result.exposed_urls is not None
        assert result.session_api_key == 'session_key_123'

        # Verify health check was called with original localhost URL
        service.httpx_client.get.assert_called_once_with(
            'http://localhost:12345/health'
        )

    async def test_container_to_checked_sandbox_info_health_check_failure(
        self, service, mock_running_container
    ):
        """Test health check failure."""
        # Setup
        service.httpx_client.get.side_effect = httpx.HTTPError('Health check failed')

        # Execute
        result = await service._container_to_checked_sandbox_info(
            mock_running_container
        )

        # Verify
        assert result is not None
        assert result.status == SandboxStatus.ERROR
        assert result.exposed_urls is None
        assert result.session_api_key is None

    async def test_container_to_checked_sandbox_info_no_health_check(
        self, service, mock_running_container
    ):
        """Test when health check is disabled."""
        # Setup
        service.health_check_path = None

        # Execute
        result = await service._container_to_checked_sandbox_info(
            mock_running_container
        )

        # Verify
        assert result is not None
        assert result.status == SandboxStatus.RUNNING
        service.httpx_client.get.assert_not_called()

    async def test_container_to_checked_sandbox_info_no_exposed_urls(
        self, service, mock_paused_container
    ):
        """Test health check when no exposed URLs."""
        # Execute
        result = await service._container_to_checked_sandbox_info(mock_paused_container)

        # Verify
        assert result is not None
        assert result.status == SandboxStatus.PAUSED
        service.httpx_client.get.assert_not_called()


class TestVolumeMount:
    """Test cases for VolumeMount model."""

    def test_volume_mount_creation(self):
        """Test VolumeMount creation with default mode."""
        mount = VolumeMount(host_path='/host', container_path='/container')
        assert mount.host_path == '/host'
        assert mount.container_path == '/container'
        assert mount.mode == 'rw'

    def test_volume_mount_custom_mode(self):
        """Test VolumeMount creation with custom mode."""
        mount = VolumeMount(host_path='/host', container_path='/container', mode='ro')
        assert mount.mode == 'ro'

    def test_volume_mount_immutable(self):
        """Test that VolumeMount is immutable."""
        mount = VolumeMount(host_path='/host', container_path='/container')
        with pytest.raises(ValueError):  # Should raise validation error
            mount.host_path = '/new_host'


class TestExposedPort:
    """Test cases for ExposedPort model."""

    def test_exposed_port_creation(self):
        """Test ExposedPort creation with default port."""
        port = ExposedPort(name='test', description='Test port')
        assert port.name == 'test'
        assert port.description == 'Test port'
        assert port.container_port == 8000

    def test_exposed_port_custom_port(self):
        """Test ExposedPort creation with custom port."""
        port = ExposedPort(name='test', description='Test port', container_port=9000)
        assert port.container_port == 9000

    def test_exposed_port_immutable(self):
        """Test that ExposedPort is immutable."""
        port = ExposedPort(name='test', description='Test port')
        with pytest.raises(ValueError):  # Should raise validation error
            port.name = 'new_name'


class TestDockerSandboxServiceInjector:
    """Test cases for DockerSandboxServiceInjector configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        from openhands.app_server.sandbox.docker_sandbox_service import (
            DockerSandboxServiceInjector,
        )

        injector = DockerSandboxServiceInjector()
        assert injector.host_port == 3000
        assert injector.container_url_pattern == 'http://localhost:{port}'

    def test_custom_host_port(self):
        """Test custom host_port configuration."""
        from openhands.app_server.sandbox.docker_sandbox_service import (
            DockerSandboxServiceInjector,
        )

        injector = DockerSandboxServiceInjector(host_port=4000)
        assert injector.host_port == 4000

    def test_custom_container_url_pattern(self):
        """Test custom container_url_pattern configuration."""
        from openhands.app_server.sandbox.docker_sandbox_service import (
            DockerSandboxServiceInjector,
        )

        injector = DockerSandboxServiceInjector(
            container_url_pattern='http://192.168.1.100:{port}'
        )
        assert injector.container_url_pattern == 'http://192.168.1.100:{port}'

    def test_custom_configuration_combined(self):
        """Test combined custom configuration for remote access."""
        from openhands.app_server.sandbox.docker_sandbox_service import (
            DockerSandboxServiceInjector,
        )

        injector = DockerSandboxServiceInjector(
            host_port=4000,
            container_url_pattern='http://192.168.1.100:{port}',
        )
        assert injector.host_port == 4000
        assert injector.container_url_pattern == 'http://192.168.1.100:{port}'


class TestDockerSandboxServiceInjectorFromEnv:
    """Test cases for DockerSandboxServiceInjector environment variable configuration."""

    def test_config_from_env_with_sandbox_host_port(self):
        """Test that SANDBOX_HOST_PORT environment variable is respected."""
        import os
        from unittest.mock import patch

        env_vars = {
            'SANDBOX_HOST_PORT': '4000',
        }

        with patch.dict(os.environ, env_vars, clear=False):
            # Clear the global config to force reload
            import openhands.app_server.config as config_module
            from openhands.app_server.config import config_from_env

            config_module._global_config = None

            config = config_from_env()
            assert config.sandbox is not None
            assert config.sandbox.host_port == 4000

    def test_config_from_env_with_sandbox_container_url_pattern(self):
        """Test that SANDBOX_CONTAINER_URL_PATTERN environment variable is respected."""
        import os
        from unittest.mock import patch

        env_vars = {
            'SANDBOX_CONTAINER_URL_PATTERN': 'http://192.168.1.100:{port}',
        }

        with patch.dict(os.environ, env_vars, clear=False):
            # Clear the global config to force reload
            import openhands.app_server.config as config_module
            from openhands.app_server.config import config_from_env

            config_module._global_config = None

            config = config_from_env()
            assert config.sandbox is not None
            assert config.sandbox.container_url_pattern == 'http://192.168.1.100:{port}'

    def test_config_from_env_with_both_sandbox_vars(self):
        """Test that both SANDBOX_HOST_PORT and SANDBOX_CONTAINER_URL_PATTERN work together."""
        import os
        from unittest.mock import patch

        env_vars = {
            'SANDBOX_HOST_PORT': '4000',
            'SANDBOX_CONTAINER_URL_PATTERN': 'http://192.168.1.100:{port}',
        }

        with patch.dict(os.environ, env_vars, clear=False):
            # Clear the global config to force reload
            import openhands.app_server.config as config_module
            from openhands.app_server.config import config_from_env

            config_module._global_config = None

            config = config_from_env()
            assert config.sandbox is not None
            assert config.sandbox.host_port == 4000
            assert config.sandbox.container_url_pattern == 'http://192.168.1.100:{port}'
