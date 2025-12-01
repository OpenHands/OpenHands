"""Tests for DockerSandboxSpecServiceInjector.

This module tests the Docker sandbox spec service injector implementation, focusing on:
- Initialization with default and custom specs
- Docker image pulling functionality when specs are missing
- Proper mocking of Docker client operations
- Error handling for Docker API failures
- Async generator behavior of the inject method
- Integration with PresetSandboxSpecService
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from docker.errors import APIError, ImageNotFound
from fastapi import Request
from starlette.datastructures import State

from openhands.app_server.errors import SandboxError
from openhands.app_server.sandbox.docker_sandbox_spec_service import (
    DockerSandboxSpecServiceInjector,
    get_default_sandbox_specs,
    get_docker_client,
)
from openhands.app_server.sandbox.preset_sandbox_spec_service import (
    PresetSandboxSpecService,
)
from openhands.app_server.sandbox.sandbox_spec_models import SandboxSpecInfo


@pytest.fixture
def mock_docker_client():
    """Mock Docker client for testing."""
    mock_client = MagicMock()
    mock_client.images = MagicMock()
    return mock_client


@pytest.fixture
def mock_state():
    """Mock injector state for testing."""
    return State()


@pytest.fixture
def mock_request():
    """Mock FastAPI request for testing."""
    request = MagicMock(spec=Request)
    request.state = State()
    return request


@pytest.fixture
def sample_spec():
    """Sample sandbox spec for testing."""
    return SandboxSpecInfo(
        id='test-image:latest',
        command=['/bin/bash'],
        initial_env={'TEST_VAR': 'test_value'},
        working_dir='/test/workspace',
    )


@pytest.fixture
def sample_specs(sample_spec):
    """List of sample sandbox specs for testing."""
    return [
        sample_spec,
        SandboxSpecInfo(
            id='another-image:v1.0',
            command=['/usr/bin/python'],
            initial_env={'PYTHON_ENV': 'test'},
            working_dir='/python/workspace',
        ),
    ]


class TestDockerSandboxSpecServiceInjector:
    """Test cases for DockerSandboxSpecServiceInjector."""

    def test_initialization_with_defaults(self):
        """Test initialization with default values."""
        injector = DockerSandboxSpecServiceInjector()

        # Should use default specs
        default_specs = get_default_sandbox_specs()
        assert len(injector.specs) == len(default_specs)
        assert injector.specs[0].id == default_specs[0].id

        # Should have pull_if_missing enabled by default
        assert injector.pull_if_missing is True

    def test_initialization_with_custom_specs(self, sample_specs):
        """Test initialization with custom specs."""
        injector = DockerSandboxSpecServiceInjector(
            specs=sample_specs, pull_if_missing=False
        )

        assert injector.specs == sample_specs
        assert injector.pull_if_missing is False

    @patch('openhands.app_server.sandbox.docker_sandbox_spec_service.get_docker_client')
    async def test_inject_with_pull_if_missing_true(
        self, mock_get_docker_client, sample_specs, mock_state
    ):
        """Test inject method when pull_if_missing is True."""
        # Setup
        mock_docker_client = MagicMock()
        mock_get_docker_client.return_value = mock_docker_client

        # Mock that images exist (no ImageNotFound exception)
        mock_docker_client.images.get.return_value = MagicMock()

        injector = DockerSandboxSpecServiceInjector(
            specs=sample_specs, pull_if_missing=True
        )

        # Execute
        async for service in injector.inject(mock_state):
            # Verify
            assert isinstance(service, PresetSandboxSpecService)
            assert service.specs == sample_specs

            # Should check for images
            assert mock_docker_client.images.get.call_count == len(sample_specs)
            mock_docker_client.images.get.assert_any_call('test-image:latest')
            mock_docker_client.images.get.assert_any_call('another-image:v1.0')

            # pull_if_missing should be set to False after first run
            assert injector.pull_if_missing is False
            break

    @patch('openhands.app_server.sandbox.docker_sandbox_spec_service.get_docker_client')
    async def test_inject_with_pull_if_missing_false(
        self, mock_get_docker_client, sample_specs, mock_state
    ):
        """Test inject method when pull_if_missing is False."""
        # Setup
        mock_docker_client = MagicMock()
        mock_get_docker_client.return_value = mock_docker_client

        injector = DockerSandboxSpecServiceInjector(
            specs=sample_specs, pull_if_missing=False
        )

        # Execute
        async for service in injector.inject(mock_state):
            # Verify
            assert isinstance(service, PresetSandboxSpecService)
            assert service.specs == sample_specs

            # Should not check for images
            mock_get_docker_client.assert_not_called()
            mock_docker_client.images.get.assert_not_called()
            break

    @patch('openhands.app_server.sandbox.docker_sandbox_spec_service.get_docker_client')
    async def test_inject_with_request(
        self, mock_get_docker_client, sample_specs, mock_request
    ):
        """Test inject method with request parameter."""
        # Setup
        mock_docker_client = MagicMock()
        mock_get_docker_client.return_value = mock_docker_client
        mock_docker_client.images.get.return_value = MagicMock()

        injector = DockerSandboxSpecServiceInjector(
            specs=sample_specs, pull_if_missing=True
        )

        # Execute
        async for service in injector.inject(mock_request.state, mock_request):
            # Verify
            assert isinstance(service, PresetSandboxSpecService)
            assert service.specs == sample_specs
            break

    @patch('openhands.app_server.sandbox.docker_sandbox_spec_service.get_docker_client')
    async def test_pull_missing_specs_all_exist(
        self, mock_get_docker_client, sample_specs
    ):
        """Test pull_missing_specs when all images exist."""
        # Setup
        mock_docker_client = MagicMock()
        mock_get_docker_client.return_value = mock_docker_client
        mock_docker_client.images.get.return_value = MagicMock()  # Images exist

        injector = DockerSandboxSpecServiceInjector(specs=sample_specs)

        # Execute
        await injector.pull_missing_specs()

        # Verify
        assert mock_docker_client.images.get.call_count == len(sample_specs)
        mock_docker_client.images.pull.assert_not_called()

    @patch('openhands.app_server.sandbox.docker_sandbox_spec_service.get_docker_client')
    async def test_pull_missing_specs_some_missing(
        self, mock_get_docker_client, sample_specs
    ):
        """Test pull_missing_specs when some images are missing."""
        # Setup
        mock_docker_client = MagicMock()
        mock_get_docker_client.return_value = mock_docker_client

        # First image exists, second is missing
        def mock_get_side_effect(image_id):
            if image_id == 'test-image:latest':
                return MagicMock()  # Exists
            else:
                raise ImageNotFound('Image not found')

        mock_docker_client.images.get.side_effect = mock_get_side_effect
        mock_docker_client.images.pull.return_value = MagicMock()

        injector = DockerSandboxSpecServiceInjector(specs=sample_specs)

        # Execute
        await injector.pull_missing_specs()

        # Verify
        assert mock_docker_client.images.get.call_count == len(sample_specs)
        mock_docker_client.images.pull.assert_called_once_with('another-image:v1.0')

    @patch('openhands.app_server.sandbox.docker_sandbox_spec_service.get_docker_client')
    async def test_pull_spec_if_missing_image_exists(
        self, mock_get_docker_client, sample_spec
    ):
        """Test pull_spec_if_missing when image exists."""
        # Setup
        mock_docker_client = MagicMock()
        mock_get_docker_client.return_value = mock_docker_client
        mock_docker_client.images.get.return_value = MagicMock()  # Image exists

        injector = DockerSandboxSpecServiceInjector()

        # Execute
        await injector.pull_spec_if_missing(sample_spec)

        # Verify
        mock_docker_client.images.get.assert_called_once_with('test-image:latest')
        mock_docker_client.images.pull.assert_not_called()

    @patch('openhands.app_server.sandbox.docker_sandbox_spec_service.get_docker_client')
    async def test_pull_spec_if_missing_image_not_found(
        self, mock_get_docker_client, sample_spec
    ):
        """Test pull_spec_if_missing when image is missing."""
        # Setup
        mock_docker_client = MagicMock()
        mock_get_docker_client.return_value = mock_docker_client
        mock_docker_client.images.get.side_effect = ImageNotFound('Image not found')
        mock_docker_client.images.pull.return_value = MagicMock()

        injector = DockerSandboxSpecServiceInjector()

        # Execute
        await injector.pull_spec_if_missing(sample_spec)

        # Verify
        mock_docker_client.images.get.assert_called_once_with('test-image:latest')
        mock_docker_client.images.pull.assert_called_once_with('test-image:latest')

    @patch('openhands.app_server.sandbox.docker_sandbox_spec_service.get_docker_client')
    async def test_pull_spec_if_missing_api_error(
        self, mock_get_docker_client, sample_spec
    ):
        """Test pull_spec_if_missing when Docker API error occurs."""
        # Setup
        mock_docker_client = MagicMock()
        mock_get_docker_client.return_value = mock_docker_client
        mock_docker_client.images.get.side_effect = APIError('Docker daemon error')

        injector = DockerSandboxSpecServiceInjector()

        # Execute & Verify
        with pytest.raises(
            SandboxError, match='Error Getting Docker Image: test-image:latest'
        ):
            await injector.pull_spec_if_missing(sample_spec)

    @patch('openhands.app_server.sandbox.docker_sandbox_spec_service.get_docker_client')
    async def test_pull_spec_if_missing_pull_api_error(
        self, mock_get_docker_client, sample_spec
    ):
        """Test pull_spec_if_missing when pull operation fails."""
        # Setup
        mock_docker_client = MagicMock()
        mock_get_docker_client.return_value = mock_docker_client
        mock_docker_client.images.get.side_effect = ImageNotFound('Image not found')
        mock_docker_client.images.pull.side_effect = APIError('Pull failed')

        injector = DockerSandboxSpecServiceInjector()

        # Execute & Verify
        with pytest.raises(
            SandboxError, match='Error Getting Docker Image: test-image:latest'
        ):
            await injector.pull_spec_if_missing(sample_spec)

    @patch('openhands.app_server.sandbox.docker_sandbox_spec_service.get_docker_client')
    async def test_pull_spec_if_missing_uses_executor(
        self, mock_get_docker_client, sample_spec
    ):
        """Test that pull_spec_if_missing uses executor for blocking operations."""
        # Setup
        mock_docker_client = MagicMock()
        mock_get_docker_client.return_value = mock_docker_client
        mock_docker_client.images.get.side_effect = ImageNotFound('Image not found')
        mock_docker_client.images.pull.return_value = MagicMock()

        injector = DockerSandboxSpecServiceInjector()

        # Mock the event loop and executor
        with patch('asyncio.get_running_loop') as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            mock_loop.run_in_executor.return_value = asyncio.Future()
            mock_loop.run_in_executor.return_value.set_result(MagicMock())

            # Execute
            await injector.pull_spec_if_missing(sample_spec)

            # Verify executor was used
            mock_loop.run_in_executor.assert_called_once_with(
                None, mock_docker_client.images.pull, 'test-image:latest'
            )

    @patch('openhands.app_server.sandbox.docker_sandbox_spec_service.get_docker_client')
    async def test_concurrent_pull_operations(
        self, mock_get_docker_client, sample_specs
    ):
        """Test that multiple specs are pulled concurrently."""
        # Setup
        mock_docker_client = MagicMock()
        mock_get_docker_client.return_value = mock_docker_client
        mock_docker_client.images.get.side_effect = ImageNotFound('Image not found')
        mock_docker_client.images.pull.return_value = MagicMock()

        injector = DockerSandboxSpecServiceInjector(specs=sample_specs)

        # Mock asyncio.gather to verify concurrent execution
        with patch('asyncio.gather') as mock_gather:
            mock_gather.return_value = asyncio.Future()
            mock_gather.return_value.set_result([None, None])

            # Execute
            await injector.pull_missing_specs()

            # Verify gather was called with correct number of coroutines
            mock_gather.assert_called_once()
            args = mock_gather.call_args[0]
            assert len(args) == len(sample_specs)

    def test_get_default_sandbox_specs(self):
        """Test get_default_sandbox_specs function."""
        specs = get_default_sandbox_specs()

        assert len(specs) == 1
        assert isinstance(specs[0], SandboxSpecInfo)
        assert specs[0].id.startswith('ghcr.io/openhands/agent-server:')
        assert specs[0].id.endswith('-python')
        assert specs[0].command == ['--port', '8000']
        assert 'OPENVSCODE_SERVER_ROOT' in specs[0].initial_env
        assert 'OH_ENABLE_VNC' in specs[0].initial_env
        assert 'LOG_JSON' in specs[0].initial_env
        assert specs[0].working_dir == '/workspace/project'

    @patch(
        'openhands.app_server.sandbox.docker_sandbox_spec_service._global_docker_client',
        None,
    )
    @patch('docker.from_env')
    def test_get_docker_client_creates_new_client(self, mock_from_env):
        """Test get_docker_client creates new client when none exists."""
        mock_client = MagicMock()
        mock_from_env.return_value = mock_client

        result = get_docker_client()

        assert result == mock_client
        mock_from_env.assert_called_once()

    @patch(
        'openhands.app_server.sandbox.docker_sandbox_spec_service._global_docker_client'
    )
    @patch('docker.from_env')
    def test_get_docker_client_reuses_existing_client(
        self, mock_from_env, mock_global_client
    ):
        """Test get_docker_client reuses existing client."""
        mock_client = MagicMock()

        # Import and patch the global variable properly
        import openhands.app_server.sandbox.docker_sandbox_spec_service as module

        module._global_docker_client = mock_client

        result = get_docker_client()

        assert result == mock_client
        mock_from_env.assert_not_called()

    async def test_inject_yields_single_service(self, sample_specs, mock_state):
        """Test that inject method yields exactly one service."""
        injector = DockerSandboxSpecServiceInjector(
            specs=sample_specs, pull_if_missing=False
        )

        services = []
        async for service in injector.inject(mock_state):
            services.append(service)

        assert len(services) == 1
        assert isinstance(services[0], PresetSandboxSpecService)

    @patch('openhands.app_server.sandbox.docker_sandbox_spec_service.get_docker_client')
    async def test_pull_if_missing_flag_reset_after_first_inject(
        self, mock_get_docker_client, sample_specs, mock_state
    ):
        """Test that pull_if_missing flag is reset to False after first inject call."""
        # Setup
        mock_docker_client = MagicMock()
        mock_get_docker_client.return_value = mock_docker_client
        mock_docker_client.images.get.return_value = MagicMock()

        injector = DockerSandboxSpecServiceInjector(
            specs=sample_specs, pull_if_missing=True
        )

        # First inject call
        async for _ in injector.inject(mock_state):
            break

        # Verify flag was reset
        assert injector.pull_if_missing is False

        # Reset mock call counts
        mock_get_docker_client.reset_mock()
        mock_docker_client.images.get.reset_mock()

        # Second inject call
        async for _ in injector.inject(mock_state):
            break

        # Verify no Docker operations were performed
        mock_get_docker_client.assert_not_called()
        mock_docker_client.images.get.assert_not_called()

    @patch('openhands.app_server.sandbox.docker_sandbox_spec_service.get_docker_client')
    @patch('openhands.app_server.sandbox.docker_sandbox_spec_service._logger')
    async def test_pull_with_progress_logging(
        self, mock_logger, mock_get_docker_client, sample_spec
    ):
        """Test that periodic progress logging occurs during image pull."""
        # Setup
        mock_docker_client = MagicMock()
        mock_get_docker_client.return_value = mock_docker_client
        mock_docker_client.images.get.side_effect = ImageNotFound('Image not found')

        # Create a future that will be resolved after some delay to simulate slow pull
        pull_future = asyncio.Future()

        async def delayed_pull_completion():
            # Wait for multiple logging intervals to pass
            await asyncio.sleep(12)  # 12 seconds = 2 logging intervals (5s each)
            pull_future.set_result(MagicMock())

        # Start the delayed completion task
        asyncio.create_task(delayed_pull_completion())

        # Mock the executor to return our delayed future
        with patch('asyncio.get_running_loop') as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            mock_loop.run_in_executor.return_value = pull_future

            injector = DockerSandboxSpecServiceInjector()

            # Execute
            await injector.pull_spec_if_missing(sample_spec)

            # Verify that progress logging occurred
            # Should have initial pull message, progress messages, and completion message
            progress_calls = [
                call
                for call in mock_logger.info.call_args_list
                if '🔄 Downloading Docker Image:' in str(call)
            ]

            # Should have at least 2 progress log messages (every 5 seconds for 12 seconds)
            assert len(progress_calls) >= 2

            # Verify the progress message format
            for call in progress_calls:
                assert '🔄 Downloading Docker Image: test-image:latest...' in str(call)

    @patch('openhands.app_server.sandbox.docker_sandbox_spec_service.get_docker_client')
    @patch('openhands.app_server.sandbox.docker_sandbox_spec_service._logger')
    async def test_pull_with_progress_logging_fast_pull(
        self, mock_logger, mock_get_docker_client, sample_spec
    ):
        """Test that no progress logging occurs for fast pulls (< 5 seconds)."""
        # Setup
        mock_docker_client = MagicMock()
        mock_get_docker_client.return_value = mock_docker_client
        mock_docker_client.images.get.side_effect = ImageNotFound('Image not found')

        # Mock fast pull (completes immediately)
        with patch('asyncio.get_running_loop') as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            fast_future = asyncio.Future()
            fast_future.set_result(MagicMock())
            mock_loop.run_in_executor.return_value = fast_future

            injector = DockerSandboxSpecServiceInjector()

            # Execute
            await injector.pull_spec_if_missing(sample_spec)

            # Verify that no progress logging occurred (only start/end messages)
            progress_calls = [
                call
                for call in mock_logger.info.call_args_list
                if '🔄 Downloading Docker Image:' in str(call)
            ]

            # Should have no progress log messages for fast pulls
            assert len(progress_calls) == 0
