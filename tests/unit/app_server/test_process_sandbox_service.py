"""Tests for ProcessSandboxService."""

import os
import tempfile
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import psutil
import pytest

from openhands.app_server.sandbox.process_sandbox_service import (
    ProcessInfo,
    ProcessSandboxService,
    ProcessSandboxServiceInjector,
)
from openhands.app_server.sandbox.sandbox_models import SandboxStatus


class MockSandboxSpec:
    """Mock sandbox specification."""

    def __init__(self):
        self.id = 'test-spec'
        self.initial_env = {'TEST_VAR': 'test_value'}
        self.plugins = []


class MockSandboxSpecService:
    """Mock sandbox spec service."""

    async def get_default_sandbox_spec(self):
        return MockSandboxSpec()

    async def get_sandbox_spec(self, spec_id: str):
        if spec_id == 'test-spec':
            return MockSandboxSpec()
        return None


@pytest.fixture
def mock_httpx_client():
    """Mock httpx client."""
    client = AsyncMock(spec=httpx.AsyncClient)
    return client


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def process_sandbox_service(mock_httpx_client, temp_dir):
    """Create a ProcessSandboxService instance for testing."""
    return ProcessSandboxService(
        user_id='test-user-id',
        sandbox_spec_service=MockSandboxSpecService(),
        base_working_dir=temp_dir,
        base_port=9000,
        python_executable='python',
        agent_server_module='openhands.agent_server',
        health_check_path='/alive',
        httpx_client=mock_httpx_client,
    )


class TestProcessSandboxService:
    """Test cases for ProcessSandboxService."""

    def test_find_unused_port(self, process_sandbox_service):
        """Test finding an unused port."""
        port = process_sandbox_service._find_unused_port()
        assert port >= process_sandbox_service.base_port
        assert port < process_sandbox_service.base_port + 10000

    @patch('os.makedirs')
    def test_create_sandbox_directory(self, mock_makedirs, process_sandbox_service):
        """Test creating a sandbox directory."""
        sandbox_dir = process_sandbox_service._create_sandbox_directory('test-id')

        expected_dir = os.path.join(process_sandbox_service.base_working_dir, 'test-id')
        assert sandbox_dir == expected_dir
        mock_makedirs.assert_called_once_with(expected_dir, exist_ok=True)

    @pytest.mark.asyncio
    async def test_wait_for_server_ready_success(self, process_sandbox_service):
        """Test waiting for server to be ready - success case."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'status': 'ok'}
        process_sandbox_service.httpx_client.get.return_value = mock_response

        result = await process_sandbox_service._wait_for_server_ready(9000, timeout=1)
        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_server_ready_timeout(self, process_sandbox_service):
        """Test waiting for server to be ready - timeout case."""
        # Mock failed response
        process_sandbox_service.httpx_client.get.side_effect = Exception(
            'Connection failed'
        )

        result = await process_sandbox_service._wait_for_server_ready(9000, timeout=1)
        assert result is False

    @patch('psutil.Process')
    def test_get_process_status_running(
        self, mock_process_class, process_sandbox_service
    ):
        """Test getting process status for running process."""
        mock_process = MagicMock()
        mock_process.is_running.return_value = True
        mock_process.status.return_value = psutil.STATUS_RUNNING
        mock_process_class.return_value = mock_process

        process_info = ProcessInfo(
            pid=1234,
            port=9000,
            user_id='test-user-id',
            working_dir='/tmp/test',
            session_api_key='test-key',
            created_at=datetime.now(),
            sandbox_spec_id='test-spec',
        )

        status = process_sandbox_service._get_process_status(process_info)
        assert status == SandboxStatus.RUNNING

    @patch('psutil.Process')
    def test_get_process_status_missing(
        self, mock_process_class, process_sandbox_service
    ):
        """Test getting process status for missing process."""
        import psutil

        mock_process_class.side_effect = psutil.NoSuchProcess(1234)

        process_info = ProcessInfo(
            pid=1234,
            port=9000,
            user_id='test-user-id',
            working_dir='/tmp/test',
            session_api_key='test-key',
            created_at=datetime.now(),
            sandbox_spec_id='test-spec',
        )

        status = process_sandbox_service._get_process_status(process_info)
        assert status == SandboxStatus.MISSING

    @pytest.mark.asyncio
    async def test_search_sandboxes_empty(self, process_sandbox_service):
        """Test searching sandboxes when none exist."""
        result = await process_sandbox_service.search_sandboxes()

        assert len(result.items) == 0
        assert result.next_page_id is None

    @pytest.mark.asyncio
    async def test_get_sandbox_not_found(self, process_sandbox_service):
        """Test getting a sandbox that doesn't exist."""
        result = await process_sandbox_service.get_sandbox('nonexistent')
        assert result is None

    @pytest.mark.asyncio
    async def test_resume_sandbox_not_found(self, process_sandbox_service):
        """Test resuming a sandbox that doesn't exist."""
        result = await process_sandbox_service.resume_sandbox('nonexistent')
        assert result is False

    @pytest.mark.asyncio
    async def test_pause_sandbox_not_found(self, process_sandbox_service):
        """Test pausing a sandbox that doesn't exist."""
        result = await process_sandbox_service.pause_sandbox('nonexistent')
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_sandbox_not_found(self, process_sandbox_service):
        """Test deleting a sandbox that doesn't exist."""
        result = await process_sandbox_service.delete_sandbox('nonexistent')
        assert result is False

    @patch('psutil.Process')
    def test_get_process_status_paused(
        self, mock_process_class, process_sandbox_service
    ):
        """Test getting process status for paused process."""
        mock_process = MagicMock()
        mock_process.is_running.return_value = True
        mock_process.status.return_value = psutil.STATUS_STOPPED
        mock_process_class.return_value = mock_process

        process_info = ProcessInfo(
            pid=1234,
            port=9000,
            user_id='test-user-id',
            working_dir='/tmp/test',
            session_api_key='test-key',
            created_at=datetime.now(),
            sandbox_spec_id='test-spec',
        )

        status = process_sandbox_service._get_process_status(process_info)
        assert status == SandboxStatus.PAUSED

    @patch('psutil.Process')
    def test_get_process_status_starting(
        self, mock_process_class, process_sandbox_service
    ):
        """Test getting process status for starting process."""
        mock_process = MagicMock()
        mock_process.is_running.return_value = True
        mock_process.status.return_value = psutil.STATUS_SLEEPING
        mock_process_class.return_value = mock_process

        process_info = ProcessInfo(
            pid=1234,
            port=9000,
            user_id='test-user-id',
            working_dir='/tmp/test',
            session_api_key='test-key',
            created_at=datetime.now(),
            sandbox_spec_id='test-spec',
        )

        status = process_sandbox_service._get_process_status(process_info)
        assert status == SandboxStatus.STARTING

    @patch('psutil.Process')
    def test_get_process_status_access_denied(
        self, mock_process_class, process_sandbox_service
    ):
        """Test getting process status when access is denied."""
        mock_process_class.side_effect = psutil.AccessDenied(1234)

        process_info = ProcessInfo(
            pid=1234,
            port=9000,
            user_id='test-user-id',
            working_dir='/tmp/test',
            session_api_key='test-key',
            created_at=datetime.now(),
            sandbox_spec_id='test-spec',
        )

        status = process_sandbox_service._get_process_status(process_info)
        assert status == SandboxStatus.MISSING

    @pytest.mark.asyncio
    async def test_process_to_sandbox_info_error_status(self, process_sandbox_service):
        """Test converting process info to sandbox info when server is not responding."""
        # Mock a process that's running but server is not responding
        with patch.object(
            process_sandbox_service,
            '_get_process_status',
            return_value=SandboxStatus.RUNNING,
        ):
            # Mock httpx client to return error response
            mock_response = MagicMock()
            mock_response.status_code = 500
            process_sandbox_service.httpx_client.get.return_value = mock_response

            process_info = ProcessInfo(
                pid=1234,
                port=9000,
                user_id='test-user-id',
                working_dir='/tmp/test',
                session_api_key='test-key',
                created_at=datetime.now(),
                sandbox_spec_id='test-spec',
            )

            sandbox_info = await process_sandbox_service._process_to_sandbox_info(
                'test-sandbox', process_info
            )

            assert sandbox_info.status == SandboxStatus.ERROR
            assert sandbox_info.session_api_key is None
            assert sandbox_info.exposed_urls is None

    @pytest.mark.asyncio
    async def test_process_to_sandbox_info_exception(self, process_sandbox_service):
        """Test converting process info to sandbox info when httpx raises exception."""
        # Mock a process that's running but httpx raises exception
        with patch.object(
            process_sandbox_service,
            '_get_process_status',
            return_value=SandboxStatus.RUNNING,
        ):
            # Mock httpx client to raise exception
            process_sandbox_service.httpx_client.get.side_effect = Exception(
                'Connection failed'
            )

            process_info = ProcessInfo(
                pid=1234,
                port=9000,
                user_id='test-user-id',
                working_dir='/tmp/test',
                session_api_key='test-key',
                created_at=datetime.now(),
                sandbox_spec_id='test-spec',
            )

            sandbox_info = await process_sandbox_service._process_to_sandbox_info(
                'test-sandbox', process_info
            )

            assert sandbox_info.status == SandboxStatus.ERROR
            assert sandbox_info.session_api_key is None
            assert sandbox_info.exposed_urls is None


class TestProcessSandboxServiceInjector:
    """Test cases for ProcessSandboxServiceInjector."""

    def test_default_values(self):
        """Test default configuration values."""
        injector = ProcessSandboxServiceInjector()

        assert injector.base_working_dir == '/tmp/openhands-sandboxes'
        assert injector.base_port == 8000
        assert injector.health_check_path == '/alive'
        assert injector.agent_server_module == 'openhands.agent_server'

    def test_custom_values(self):
        """Test custom configuration values."""
        injector = ProcessSandboxServiceInjector(
            base_working_dir='/custom/path',
            base_port=9000,
            health_check_path='/health',
            agent_server_module='custom.agent.module',
        )

        assert injector.base_working_dir == '/custom/path'
        assert injector.base_port == 9000
        assert injector.health_check_path == '/health'
        assert injector.agent_server_module == 'custom.agent.module'
