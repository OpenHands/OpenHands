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
import time
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
            SandboxError, match="Error getting Docker image 'test-image:latest'"
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
            SandboxError, match="Error getting Docker image 'test-image:latest'"
        ):
            await injector.pull_spec_if_missing(sample_spec)

    @patch('openhands.app_server.sandbox.docker_sandbox_spec_service.get_docker_client')
    async def test_pull_spec_if_missing_uses_executor(
        self, mock_get_docker_client, sample_spec
    ):
        """Test that pull_spec_if_missing uses executor for blocking operations.

        Both images.get (preflight check) and images.pull must be dispatched
        via run_in_executor so they don't block the event loop.
        """
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

            def executor_side_effect(executor, fn, *args):
                """Invoke fn so that mock side_effects/return_values are honoured."""
                fut = asyncio.Future()
                try:
                    result = fn(*args)
                    fut.set_result(result)
                except Exception as exc:
                    fut.set_exception(exc)
                return fut

            mock_loop.run_in_executor.side_effect = executor_side_effect

            # Execute
            await injector.pull_spec_if_missing(sample_spec)

            # Verify executor was used for both the preflight get and the pull
            assert mock_loop.run_in_executor.call_count == 2
            mock_loop.run_in_executor.assert_any_call(
                None, mock_docker_client.images.get, 'test-image:latest'
            )
            mock_loop.run_in_executor.assert_any_call(
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

        # images.pull blocks its executor thread long enough for the logger to fire twice.
        # Using a real thread executor means the event loop stays free to run periodic_logger.
        mock_docker_client.images.pull.side_effect = lambda _id: time.sleep(12)

        injector = DockerSandboxSpecServiceInjector()

        # Execute
        await injector.pull_spec_if_missing(sample_spec)

        # Verify that progress logging occurred
        progress_calls = [
            call
            for call in mock_logger.info.call_args_list
            if 'Downloading Docker Image:' in str(call)
        ]

        # Should have at least 2 progress log messages (every 5 seconds for 12 seconds)
        assert len(progress_calls) >= 2

        # Verify the progress message format
        for call in progress_calls:
            assert 'Downloading Docker Image:' in str(call)
            assert 'test-image:latest' in str(call)

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

        # images.pull returns immediately (default MagicMock -- no side_effect needed)
        injector = DockerSandboxSpecServiceInjector()

        # Execute
        await injector.pull_spec_if_missing(sample_spec)

        # Verify that no progress logging occurred (only start/end messages)
        progress_calls = [
            call
            for call in mock_logger.info.call_args_list
            if 'Downloading Docker Image:' in str(call)
        ]

        # Should have no progress log messages for fast pulls
        assert len(progress_calls) == 0

    @patch('openhands.app_server.sandbox.docker_sandbox_spec_service.get_docker_client')
    async def test_pull_logger_task_cancel_timeout_is_handled(
        self, mock_get_docker_client, sample_spec
    ):
        """Test that a stuck logger_task cancellation does not hang pull_spec_if_missing.

        After Fix D the cleanup block uses asyncio.wait_for(logger_task, timeout=5.0)
        so that a logger task that ignores CancelledError cannot block the caller.
        We verify the fix by simulating asyncio.TimeoutError being raised ONLY for
        Task-based wait_for calls (the logger_task cleanup in the finally block).
        Coroutine-based wait_for calls (periodic_logger's pull_complete.wait()) use
        the real implementation so the pull can complete normally.
        """
        mock_docker_client = MagicMock()
        mock_get_docker_client.return_value = mock_docker_client
        mock_docker_client.images.get.side_effect = ImageNotFound('not found')
        # Pull completes instantly
        mock_docker_client.images.pull.return_value = MagicMock()

        original_wait_for = asyncio.wait_for

        async def selective_wait_for(coro_or_future, timeout):
            # Only intercept Task-based calls (the logger_task cleanup in finally).
            # Coroutine-based calls (pull_complete.wait() inside periodic_logger)
            # fall through to the real wait_for so the pull can complete normally.
            if isinstance(coro_or_future, asyncio.Task):
                try:
                    coro_or_future.cancel()
                except Exception:
                    pass
                raise asyncio.TimeoutError
            return await original_wait_for(coro_or_future, timeout)

        injector = DockerSandboxSpecServiceInjector()

        with patch('asyncio.wait_for', new=selective_wait_for):
            # Should complete without raising — TimeoutError is caught by Fix D
            await injector.pull_spec_if_missing(sample_spec)

    @patch('openhands.app_server.sandbox.docker_sandbox_spec_service.get_docker_client')
    async def test_pull_spec_if_missing_concurrent_no_double_pull(
        self, mock_get_docker_client, sample_spec
    ):
        """Test that two concurrent callers for the same image only pull once.

        The per-image asyncio.Lock ensures that when multiple coroutines race to
        pull the same image, only the first performs the actual pull while the other
        waits.  The second coroutine acquires the lock *after* the first has finished
        and then finds the image already present (images.get succeeds), so images.pull
        is only ever called once.
        """
        mock_docker_client = MagicMock()
        mock_get_docker_client.return_value = mock_docker_client

        pull_count = 0
        first_pull_allowed = asyncio.Event()

        def get_side_effect(image_id):
            # Before the first pull completes raise ImageNotFound; after, return ok.
            if not first_pull_allowed.is_set():
                raise ImageNotFound('not found')
            return MagicMock()

        async def pull_side_effect_async(image_id):
            nonlocal pull_count
            pull_count += 1
            await asyncio.sleep(0.05)
            first_pull_allowed.set()

        mock_docker_client.images.get.side_effect = get_side_effect

        injector = DockerSandboxSpecServiceInjector()
        original_pull_spec = injector.pull_spec_if_missing

        async def patched_pull_spec(spec):
            loop = asyncio.get_running_loop()

            def sync_pull(image_id):
                fut = asyncio.run_coroutine_threadsafe(
                    pull_side_effect_async(image_id), loop
                )
                fut.result()

            mock_docker_client.images.pull.side_effect = sync_pull
            await original_pull_spec(spec)

        await asyncio.gather(
            patched_pull_spec(sample_spec),
            patched_pull_spec(sample_spec),
        )

        # Despite two concurrent callers, Docker pull must only be invoked once.
        assert pull_count == 1
