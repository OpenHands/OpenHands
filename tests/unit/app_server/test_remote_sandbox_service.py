"""Tests for RemoteSandboxService.

This module tests the RemoteSandboxService implementation, focusing on:
- Remote runtime API communication and error handling
- Sandbox lifecycle management (start, pause, resume, delete)
- Status mapping from remote runtime to internal sandbox statuses
- Environment variable injection for CORS and webhooks
- Data transformation from remote runtime to SandboxInfo objects
- User-scoped sandbox operations and security
- Pagination and search functionality
- Error handling for HTTP failures and edge cases
"""

import asyncio
import json
from contextlib import ExitStack, asynccontextmanager
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from openhands.app_server.errors import SandboxDeleteRetryError, SandboxError
from openhands.app_server.sandbox.remote_sandbox_service import (
    ALLOW_CORS_ORIGINS_VARIABLE,
    STATUS_MAPPING,
    WEBHOOK_CALLBACK_VARIABLE,
    RemoteSandboxService,
    StoredRemoteSandbox,
)
from openhands.app_server.sandbox.sandbox_models import (
    AGENT_SERVER,
    VSCODE,
    WORKER_1,
    WORKER_2,
    SandboxInfo,
    SandboxStatus,
)
from openhands.app_server.sandbox.sandbox_spec_models import SandboxSpecInfo
from openhands.app_server.settings.settings_models import SandboxGroupingStrategy
from openhands.app_server.user.user_context import UserContext


@pytest.fixture
def mock_sandbox_spec_service():
    """Mock SandboxSpecService for testing."""
    mock_service = AsyncMock()
    mock_spec = SandboxSpecInfo(
        id='test-image:latest',
        command=['/usr/local/bin/openhands-agent-server', '--port', '60000'],
        initial_env={'TEST_VAR': 'test_value'},
        working_dir='/workspace/project',
    )
    mock_service.get_default_sandbox_spec.return_value = mock_spec
    mock_service.get_sandbox_spec.return_value = mock_spec
    return mock_service


@pytest.fixture
def mock_user_context():
    """Mock UserContext for testing."""
    mock_context = AsyncMock(spec=UserContext)
    mock_context.get_user_id.return_value = 'test-user-123'
    return mock_context


@pytest.fixture
def mock_httpx_client():
    """Mock httpx.AsyncClient for testing."""
    return AsyncMock(spec=httpx.AsyncClient)


@pytest.fixture
def mock_db_session():
    """Mock database session for testing."""
    return AsyncMock(spec=AsyncSession)


@pytest.fixture
def remote_sandbox_service(
    mock_sandbox_spec_service, mock_user_context, mock_httpx_client, mock_db_session
):
    """Create RemoteSandboxService instance with mocked dependencies."""
    return RemoteSandboxService(
        sandbox_spec_service=mock_sandbox_spec_service,
        api_url='https://api.example.com',
        api_key='test-api-key',
        web_url='https://web.example.com',
        resource_factor=1,
        runtime_class='gvisor',
        start_sandbox_timeout=120,
        max_num_sandboxes=10,
        user_context=mock_user_context,
        httpx_client=mock_httpx_client,
        db_session=mock_db_session,
    )


def _make_stream_response(
    status_code: int, content: bytes = b'', headers: dict | None = None
):
    """A fake httpx streaming response: status/headers + async aiter_bytes()."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.stream_closed = False

    async def _aiter_bytes():
        yield content

    resp.aiter_bytes = _aiter_bytes
    return resp


def _stream_client(resp_or_map):
    """AsyncMock httpx client whose .stream(...) yields a fake response.

    ``resp_or_map`` is a single fake response or a ``{format: response}`` map.
    """
    client = AsyncMock()

    @asynccontextmanager
    async def _stream(method, url, **kwargs):
        if isinstance(resp_or_map, dict):
            response = resp_or_map[kwargs['params']['format']]
        else:
            response = resp_or_map
        try:
            yield response
        finally:
            response.stream_closed = True

    client.stream = MagicMock(side_effect=_stream)
    return client


def create_runtime_data(
    session_id: str = 'test-sandbox-123',
    status: str = 'running',
    url: str = 'https://sandbox.example.com',
    session_api_key: str = 'test-session-key',
    runtime_id: str = 'runtime-456',
) -> dict[str, Any]:
    """Helper function to create runtime data for testing."""
    return {
        'session_id': session_id,
        'status': status,
        'url': url,
        'session_api_key': session_api_key,
        'runtime_id': runtime_id,
    }


def create_stored_sandbox(
    sandbox_id: str = 'test-sandbox-123',
    user_id: str = 'test-user-123',
    spec_id: str = 'test-image:latest',
    created_at: datetime | None = None,
    session_api_key_hash: str | None = None,
) -> StoredRemoteSandbox:
    """Helper function to create StoredRemoteSandbox for testing."""
    if created_at is None:
        created_at = datetime.now(timezone.utc)

    return StoredRemoteSandbox(
        id=sandbox_id,
        created_by_user_id=user_id,
        sandbox_spec_id=spec_id,
        session_api_key_hash=session_api_key_hash,
        created_at=created_at,
    )


class TestRemoteSandboxService:
    """Test cases for RemoteSandboxService core functionality."""

    @pytest.mark.asyncio
    async def test_send_runtime_api_request_success(self, remote_sandbox_service):
        """Test successful API request to remote runtime."""
        # Setup
        mock_response = MagicMock()
        mock_response.json.return_value = {'result': 'success'}
        remote_sandbox_service.httpx_client.request.return_value = mock_response

        # Execute
        response = await remote_sandbox_service._send_runtime_api_request(
            'GET', '/test-endpoint', json={'test': 'data'}
        )

        # Verify
        assert response == mock_response
        remote_sandbox_service.httpx_client.request.assert_called_once_with(
            'GET',
            'https://api.example.com/test-endpoint',
            headers={'X-API-Key': 'test-api-key'},
            json={'test': 'data'},
        )

    @pytest.mark.asyncio
    async def test_send_runtime_api_request_timeout(self, remote_sandbox_service):
        """Test API request timeout handling."""
        # Setup
        remote_sandbox_service.httpx_client.request.side_effect = (
            httpx.TimeoutException('Request timeout')
        )

        # Execute & Verify
        with pytest.raises(httpx.TimeoutException):
            await remote_sandbox_service._send_runtime_api_request('GET', '/test')

    @pytest.mark.asyncio
    async def test_send_runtime_api_request_http_error(self, remote_sandbox_service):
        """Test API request HTTP error handling."""
        # Setup
        remote_sandbox_service.httpx_client.request.side_effect = httpx.HTTPError(
            'HTTP error'
        )

        # Execute & Verify
        with pytest.raises(httpx.HTTPError):
            await remote_sandbox_service._send_runtime_api_request('GET', '/test')


class TestStatusMapping:
    """Test cases for status mapping functionality."""

    @pytest.mark.asyncio
    async def test_get_sandbox_status_from_runtime_with_status(
        self, remote_sandbox_service
    ):
        """Test status mapping using status field."""
        runtime_data = create_runtime_data(status='running')

        status = remote_sandbox_service._get_sandbox_status_from_runtime(runtime_data)

        assert status == SandboxStatus.RUNNING

    @pytest.mark.asyncio
    async def test_get_sandbox_status_from_runtime_no_runtime(
        self, remote_sandbox_service
    ):
        """Test status mapping with no runtime data."""
        status = remote_sandbox_service._get_sandbox_status_from_runtime(None)

        assert status == SandboxStatus.MISSING

    @pytest.mark.asyncio
    async def test_get_sandbox_status_from_runtime_unknown_status(
        self, remote_sandbox_service
    ):
        """Test status mapping with unknown status values."""
        runtime_data = create_runtime_data(status='unknown_status')

        status = remote_sandbox_service._get_sandbox_status_from_runtime(runtime_data)

        assert status == SandboxStatus.MISSING

    @pytest.mark.asyncio
    async def test_get_sandbox_status_from_runtime_empty_status(
        self, remote_sandbox_service
    ):
        """Test status mapping with empty status field."""
        runtime_data = create_runtime_data(status='')

        status = remote_sandbox_service._get_sandbox_status_from_runtime(runtime_data)

        assert status == SandboxStatus.MISSING

    @pytest.mark.asyncio
    async def test_status_mapping_coverage(self, remote_sandbox_service):
        """Test all status mappings are handled correctly."""
        test_cases = [
            ('running', SandboxStatus.RUNNING),
            ('paused', SandboxStatus.PAUSED),
            ('stopped', SandboxStatus.MISSING),
            ('starting', SandboxStatus.STARTING),
            ('error', SandboxStatus.ERROR),
        ]

        for status, expected_status in test_cases:
            runtime_data = create_runtime_data(status=status)
            result = remote_sandbox_service._get_sandbox_status_from_runtime(
                runtime_data
            )
            assert result == expected_status, f'Failed for status: {status}'

    @pytest.mark.asyncio
    async def test_status_mapping_case_insensitive(self, remote_sandbox_service):
        """Test that status mapping is case-insensitive."""
        test_cases = [
            ('RUNNING', SandboxStatus.RUNNING),
            ('Running', SandboxStatus.RUNNING),
            ('PAUSED', SandboxStatus.PAUSED),
            ('Starting', SandboxStatus.STARTING),
        ]

        for status, expected_status in test_cases:
            runtime_data = create_runtime_data(status=status)
            result = remote_sandbox_service._get_sandbox_status_from_runtime(
                runtime_data
            )
            assert result == expected_status, f'Failed for status: {status}'


class TestEnvironmentInitialization:
    """Test cases for environment variable initialization."""

    @pytest.mark.asyncio
    async def test_init_environment_with_web_url(self, remote_sandbox_service):
        """Test environment initialization with web_url set."""
        # Setup
        sandbox_spec = SandboxSpecInfo(
            id='test-image',
            command=['test'],
            initial_env={'EXISTING_VAR': 'existing_value'},
            working_dir='/workspace',
        )
        sandbox_id = 'test-sandbox-123'

        # Execute
        environment = await remote_sandbox_service._init_environment(
            sandbox_spec, sandbox_id
        )

        # Verify
        expected_webhook_url = 'https://web.example.com/api/v1/webhooks'
        assert environment['EXISTING_VAR'] == 'existing_value'
        assert environment[WEBHOOK_CALLBACK_VARIABLE] == expected_webhook_url
        assert environment[ALLOW_CORS_ORIGINS_VARIABLE] == 'https://web.example.com'
        # Verify worker port environment variables are set
        assert environment[WORKER_1] == '12000'
        assert environment[WORKER_2] == '12001'

    @pytest.mark.asyncio
    async def test_init_environment_without_web_url(self, remote_sandbox_service):
        """Test environment initialization without web_url."""
        # Setup
        remote_sandbox_service.web_url = None
        sandbox_spec = SandboxSpecInfo(
            id='test-image',
            command=['test'],
            initial_env={'EXISTING_VAR': 'existing_value'},
            working_dir='/workspace',
        )
        sandbox_id = 'test-sandbox-123'

        # Execute
        environment = await remote_sandbox_service._init_environment(
            sandbox_spec, sandbox_id
        )

        # Verify
        assert environment['EXISTING_VAR'] == 'existing_value'
        assert WEBHOOK_CALLBACK_VARIABLE not in environment
        assert ALLOW_CORS_ORIGINS_VARIABLE not in environment
        # Worker port environment variables should still be set regardless of web_url
        assert environment[WORKER_1] == '12000'
        assert environment[WORKER_2] == '12001'


class TestSandboxInfoConversion:
    """Test cases for converting stored sandbox and runtime data to SandboxInfo."""

    @pytest.mark.asyncio
    async def test_to_sandbox_info_with_running_runtime(self, remote_sandbox_service):
        """Test conversion to SandboxInfo with running runtime."""
        # Setup
        stored_sandbox = create_stored_sandbox()
        runtime_data = create_runtime_data(status='running')

        # Execute
        sandbox_info = remote_sandbox_service._to_sandbox_info(
            stored_sandbox, runtime_data
        )

        # Verify
        assert sandbox_info.id == 'test-sandbox-123'
        assert sandbox_info.created_by_user_id == 'test-user-123'
        assert sandbox_info.sandbox_spec_id == 'test-image:latest'
        assert sandbox_info.status == SandboxStatus.RUNNING
        assert sandbox_info.session_api_key == 'test-session-key'
        assert len(sandbox_info.exposed_urls) == 4

        # Check exposed URLs
        url_names = [url.name for url in sandbox_info.exposed_urls]
        assert AGENT_SERVER in url_names
        assert VSCODE in url_names
        assert WORKER_1 in url_names
        assert WORKER_2 in url_names

    @pytest.mark.asyncio
    async def test_to_sandbox_info_with_starting_runtime(self, remote_sandbox_service):
        """Test conversion to SandboxInfo with starting runtime."""
        # Setup
        stored_sandbox = create_stored_sandbox()
        runtime_data = create_runtime_data(status='starting')

        # Execute
        sandbox_info = remote_sandbox_service._to_sandbox_info(
            stored_sandbox, runtime_data
        )

        # Verify
        assert sandbox_info.status == SandboxStatus.STARTING
        assert sandbox_info.session_api_key == 'test-session-key'
        assert sandbox_info.exposed_urls is None

    @pytest.mark.asyncio
    async def test_to_sandbox_info_loads_runtime_when_none_provided(
        self, remote_sandbox_service
    ):
        """Test that runtime data is loaded when not provided."""
        # Setup
        stored_sandbox = create_stored_sandbox()

        # Execute
        sandbox_info = remote_sandbox_service._to_sandbox_info(stored_sandbox, None)

        # Verify
        assert sandbox_info.status == SandboxStatus.MISSING


class TestSandboxLifecycle:
    """Test cases for sandbox lifecycle operations."""

    @pytest.mark.asyncio
    async def test_start_sandbox_success(
        self, remote_sandbox_service, mock_sandbox_spec_service
    ):
        """Test successful sandbox start."""
        # Setup
        mock_response = MagicMock()
        mock_response.json.return_value = create_runtime_data(status='running')
        remote_sandbox_service.httpx_client.request.return_value = mock_response
        remote_sandbox_service.pause_old_sandboxes = AsyncMock(return_value=[])

        # Mock database operations
        remote_sandbox_service.db_session.add = MagicMock()
        remote_sandbox_service.db_session.commit = AsyncMock()

        # Execute
        with patch('base62.encodebytes', return_value='test-sandbox-123'):
            sandbox_info = await remote_sandbox_service.start_sandbox()

        # Verify
        assert sandbox_info.id == 'test-sandbox-123'
        assert sandbox_info.status == SandboxStatus.RUNNING
        remote_sandbox_service.pause_old_sandboxes.assert_called_once_with(
            9
        )  # max_num_sandboxes - 1
        remote_sandbox_service.db_session.add.assert_called_once()
        remote_sandbox_service.db_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_sandbox_with_specific_spec(
        self, remote_sandbox_service, mock_sandbox_spec_service
    ):
        """Test starting sandbox with specific sandbox spec."""
        # Setup
        mock_response = MagicMock()
        mock_response.json.return_value = create_runtime_data()
        remote_sandbox_service.httpx_client.request.return_value = mock_response
        remote_sandbox_service.pause_old_sandboxes = AsyncMock(return_value=[])
        remote_sandbox_service.db_session.add = MagicMock()
        remote_sandbox_service.db_session.commit = AsyncMock()

        # Execute
        with patch('base62.encodebytes', return_value='test-sandbox-123'):
            await remote_sandbox_service.start_sandbox('custom-spec-id')

        # Verify
        mock_sandbox_spec_service.get_sandbox_spec.assert_called_once_with(
            'custom-spec-id'
        )

    @pytest.mark.asyncio
    async def test_start_sandbox_spec_not_found(
        self, remote_sandbox_service, mock_sandbox_spec_service
    ):
        """Test starting sandbox with non-existent spec."""
        # Setup
        mock_sandbox_spec_service.get_sandbox_spec.return_value = None
        remote_sandbox_service.pause_old_sandboxes = AsyncMock(return_value=[])

        # Execute & Verify
        with pytest.raises(ValueError, match=r"Sandbox Spec '.*' not found"):
            await remote_sandbox_service.start_sandbox('non-existent-spec')

    @pytest.mark.asyncio
    async def test_start_sandbox_with_sandbox_id(
        self, remote_sandbox_service, mock_sandbox_spec_service
    ):
        """Test starting sandbox with a specified sandbox_id."""
        # Setup
        mock_response = MagicMock()
        mock_response.json.return_value = create_runtime_data(
            session_id='custom_sandbox_id'
        )
        remote_sandbox_service.httpx_client.request.return_value = mock_response
        remote_sandbox_service.pause_old_sandboxes = AsyncMock(return_value=[])

        # Mock database operations
        remote_sandbox_service.db_session.add = MagicMock()
        remote_sandbox_service.db_session.commit = AsyncMock()

        # Execute with custom sandbox_id - should not need base62 encoding
        sandbox_info = await remote_sandbox_service.start_sandbox(
            sandbox_id='custom_sandbox_id'
        )

        # Verify the custom sandbox_id is used
        assert sandbox_info.id == 'custom_sandbox_id'
        # Verify the stored sandbox used the custom ID
        add_call_args = remote_sandbox_service.db_session.add.call_args[0][0]
        assert add_call_args.id == 'custom_sandbox_id'

    @pytest.mark.asyncio
    async def test_start_sandbox_http_error(self, remote_sandbox_service):
        """Test sandbox start with HTTP error."""
        # Setup
        remote_sandbox_service.httpx_client.request.side_effect = httpx.HTTPError(
            'API Error'
        )
        remote_sandbox_service.pause_old_sandboxes = AsyncMock(return_value=[])
        remote_sandbox_service.db_session.add = MagicMock()
        remote_sandbox_service.db_session.commit = AsyncMock()

        # Execute & Verify
        with patch('base62.encodebytes', return_value='test-sandbox-123'):
            with pytest.raises(SandboxError, match='Failed to start sandbox'):
                await remote_sandbox_service.start_sandbox()

    @pytest.mark.asyncio
    async def test_start_sandbox_with_sysbox_runtime(self, remote_sandbox_service):
        """Test sandbox start with sysbox runtime class."""
        # Setup
        remote_sandbox_service.runtime_class = 'sysbox'
        mock_response = MagicMock()
        mock_response.json.return_value = create_runtime_data()
        remote_sandbox_service.httpx_client.request.return_value = mock_response
        remote_sandbox_service.pause_old_sandboxes = AsyncMock(return_value=[])
        remote_sandbox_service.db_session.add = MagicMock()
        remote_sandbox_service.db_session.commit = AsyncMock()

        # Execute
        with patch('base62.encodebytes', return_value='test-sandbox-123'):
            await remote_sandbox_service.start_sandbox()

        # Verify runtime_class is included in request
        call_args = remote_sandbox_service.httpx_client.request.call_args
        request_data = call_args[1]['json']
        assert request_data['runtime_class'] == 'sysbox-runc'

    @pytest.mark.asyncio
    async def test_resume_sandbox_success(self, remote_sandbox_service):
        """Test successful sandbox resume."""
        # Setup
        stored_sandbox = create_stored_sandbox()
        runtime_data = create_runtime_data()

        remote_sandbox_service._get_stored_sandbox = AsyncMock(
            return_value=stored_sandbox
        )
        remote_sandbox_service._get_runtime = AsyncMock(return_value=runtime_data)
        remote_sandbox_service.pause_old_sandboxes = AsyncMock(return_value=[])

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'session_api_key': 'new-session-key-123'}
        remote_sandbox_service.httpx_client.request.return_value = mock_response

        # Execute
        result = await remote_sandbox_service.resume_sandbox('test-sandbox-123')

        # Verify
        assert result is True
        remote_sandbox_service.pause_old_sandboxes.assert_called_once_with(9)
        remote_sandbox_service.httpx_client.request.assert_called_once_with(
            'POST',
            'https://api.example.com/resume',
            headers={'X-API-Key': 'test-api-key'},
            json={'runtime_id': 'runtime-456'},
        )

    @pytest.mark.asyncio
    async def test_resume_sandbox_not_found(self, remote_sandbox_service):
        """Test resuming non-existent sandbox."""
        # Setup
        remote_sandbox_service._get_stored_sandbox = AsyncMock(return_value=None)
        remote_sandbox_service.pause_old_sandboxes = AsyncMock(return_value=[])

        # Execute
        result = await remote_sandbox_service.resume_sandbox('non-existent')

        # Verify
        assert result is False

    @pytest.mark.asyncio
    async def test_resume_sandbox_runtime_not_found(self, remote_sandbox_service):
        """Test resuming sandbox when runtime returns 404."""
        # Setup
        stored_sandbox = create_stored_sandbox()
        runtime_data = create_runtime_data()

        remote_sandbox_service._get_stored_sandbox = AsyncMock(
            return_value=stored_sandbox
        )
        remote_sandbox_service._get_runtime = AsyncMock(return_value=runtime_data)
        remote_sandbox_service.pause_old_sandboxes = AsyncMock(return_value=[])

        mock_response = MagicMock()
        mock_response.status_code = 404
        remote_sandbox_service.httpx_client.request.return_value = mock_response

        # Execute
        result = await remote_sandbox_service.resume_sandbox('test-sandbox-123')

        # Verify
        assert result is False

    @pytest.mark.asyncio
    async def test_pause_sandbox_success(self, remote_sandbox_service):
        """Test successful sandbox pause."""
        # Setup
        stored_sandbox = create_stored_sandbox()
        runtime_data = create_runtime_data()

        remote_sandbox_service._get_stored_sandbox = AsyncMock(
            return_value=stored_sandbox
        )
        remote_sandbox_service._get_runtime = AsyncMock(return_value=runtime_data)

        mock_response = MagicMock()
        mock_response.status_code = 200
        remote_sandbox_service.httpx_client.request.return_value = mock_response

        # Execute
        result = await remote_sandbox_service.pause_sandbox('test-sandbox-123')

        # Verify
        assert result is True
        remote_sandbox_service.httpx_client.request.assert_called_once_with(
            'POST',
            'https://api.example.com/pause',
            headers={'X-API-Key': 'test-api-key'},
            json={'runtime_id': 'runtime-456'},
        )

    @pytest.mark.asyncio
    async def test_delete_sandbox_success(self, remote_sandbox_service):
        """Test successful sandbox deletion."""
        # Setup
        stored_sandbox = create_stored_sandbox()
        runtime_data = create_runtime_data()

        remote_sandbox_service._get_stored_sandbox = AsyncMock(
            return_value=stored_sandbox
        )
        remote_sandbox_service._get_runtime = AsyncMock(return_value=runtime_data)
        remote_sandbox_service.db_session.delete = AsyncMock()
        remote_sandbox_service.db_session.commit = AsyncMock()

        mock_response = MagicMock()
        mock_response.status_code = 200
        remote_sandbox_service.httpx_client.request.return_value = mock_response

        # Execute
        result = await remote_sandbox_service.delete_sandbox('test-sandbox-123')

        # Verify
        assert result is True
        remote_sandbox_service.db_session.delete.assert_called_once_with(stored_sandbox)
        # delete_sandbox no longer commits internally: the session key is dropped
        # atomically with the row delete, which the caller commits.
        remote_sandbox_service.db_session.commit.assert_not_awaited()
        remote_sandbox_service.httpx_client.request.assert_called_once_with(
            'POST',
            'https://api.example.com/stop',
            headers={'X-API-Key': 'test-api-key'},
            json={'runtime_id': 'runtime-456'},
        )

    @pytest.mark.asyncio
    async def test_archive_conversation_workspace_archives_without_stopping(
        self, remote_sandbox_service
    ):
        """A grouped, non-terminal conversation's workspace is captured without
        stopping the shared sandbox (no /stop, no row delete)."""
        stored_sandbox = create_stored_sandbox()
        runtime_data = create_runtime_data()
        remote_sandbox_service._get_stored_sandbox = AsyncMock(
            return_value=stored_sandbox
        )
        remote_sandbox_service._get_runtime = AsyncMock(return_value=runtime_data)
        remote_sandbox_service.db_session.delete = AsyncMock()
        remote_sandbox_service.httpx_client.request = AsyncMock()

        with patch(
            'openhands.app_server.sandbox.remote_sandbox_service.workspace_archive'
        ) as mock_wa:
            mock_wa.archive_enabled.return_value = True
            mock_wa.archive_workspace = AsyncMock(return_value=True)
            ok = await remote_sandbox_service.archive_conversation_workspace(
                'test-sandbox-123', conversation_id='conv-1'
            )

        assert ok is True
        mock_wa.archive_workspace.assert_awaited_once()
        # Captured, but the shared sandbox was neither stopped nor deleted.
        remote_sandbox_service.httpx_client.request.assert_not_called()
        remote_sandbox_service.db_session.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_archive_conversation_workspace_noop_when_disabled(
        self, remote_sandbox_service
    ):
        """No-op (returns True) when archiving is disabled — no lookups at all."""
        remote_sandbox_service._get_stored_sandbox = AsyncMock()
        with patch(
            'openhands.app_server.sandbox.remote_sandbox_service.workspace_archive'
        ) as mock_wa:
            mock_wa.archive_enabled.return_value = False
            ok = await remote_sandbox_service.archive_conversation_workspace(
                'test-sandbox-123', conversation_id='conv-1'
            )
        assert ok is True
        remote_sandbox_service._get_stored_sandbox.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_sandbox_runtime_not_found_ignored(
        self, remote_sandbox_service
    ):
        """Test sandbox deletion when runtime returns 404 (should be ignored)."""
        # Setup
        stored_sandbox = create_stored_sandbox()
        runtime_data = create_runtime_data()

        remote_sandbox_service._get_stored_sandbox = AsyncMock(
            return_value=stored_sandbox
        )
        remote_sandbox_service._get_runtime = AsyncMock(return_value=runtime_data)
        remote_sandbox_service.db_session.delete = AsyncMock()
        remote_sandbox_service.db_session.commit = AsyncMock()

        mock_response = MagicMock()
        mock_response.status_code = 404
        remote_sandbox_service.httpx_client.request.return_value = mock_response

        # Execute
        result = await remote_sandbox_service.delete_sandbox('test-sandbox-123')

        # Verify
        assert result is True  # 404 should be ignored for delete operations

    @pytest.mark.asyncio
    async def test_delete_sandbox_runtime_gone_deletes_record(
        self, remote_sandbox_service
    ):
        """_get_runtime 404 (runtime already gone) still deletes the DB row."""
        stored_sandbox = create_stored_sandbox()

        remote_sandbox_service._get_stored_sandbox = AsyncMock(
            return_value=stored_sandbox
        )
        not_found = httpx.HTTPStatusError(
            'Not Found',
            request=httpx.Request('GET', 'https://api.example.com/sessions/x'),
            response=httpx.Response(404),
        )
        remote_sandbox_service._get_runtime = AsyncMock(side_effect=not_found)
        remote_sandbox_service.db_session.delete = AsyncMock()

        result = await remote_sandbox_service.delete_sandbox('test-sandbox-123')

        assert result is True
        remote_sandbox_service.db_session.delete.assert_called_once_with(stored_sandbox)
        # Nothing to stop when the runtime is already gone.
        remote_sandbox_service.httpx_client.request.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_sandbox_runtime_transient_error_keeps_record(
        self, remote_sandbox_service
    ):
        """A transient (5xx) _get_runtime error leaves the row for a retry."""
        stored_sandbox = create_stored_sandbox()

        remote_sandbox_service._get_stored_sandbox = AsyncMock(
            return_value=stored_sandbox
        )
        server_error = httpx.HTTPStatusError(
            'Server Error',
            request=httpx.Request('GET', 'https://api.example.com/sessions/x'),
            response=httpx.Response(503),
        )
        remote_sandbox_service._get_runtime = AsyncMock(side_effect=server_error)
        remote_sandbox_service.db_session.delete = AsyncMock()

        # Sandbox still exists -> raise (router 503), keep the row for retry;
        # never report 404.
        with pytest.raises(SandboxDeleteRetryError):
            await remote_sandbox_service.delete_sandbox('test-sandbox-123')
        remote_sandbox_service.db_session.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_sandbox_transient_error_invalidates_session_key(
        self, remote_sandbox_service
    ):
        """A transient runtime error keeps the row + running runtime for retry, but
        the session key is invalidated UP FRONT and committed — a delete is often a
        revoke of a leaked key, so the key must die promptly and the caller's
        rollback on the raised error must not be able to resurrect it."""
        stored_sandbox = create_stored_sandbox(session_api_key_hash='live-hash')

        remote_sandbox_service._get_stored_sandbox = AsyncMock(
            return_value=stored_sandbox
        )
        remote_sandbox_service._get_runtime = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                'Server Error',
                request=httpx.Request('GET', 'https://api.example.com/sessions/x'),
                response=httpx.Response(503),
            )
        )
        remote_sandbox_service.db_session.delete = AsyncMock()
        remote_sandbox_service.db_session.commit = AsyncMock()

        with pytest.raises(SandboxDeleteRetryError):
            await remote_sandbox_service.delete_sandbox('test-sandbox-123')

        # Row kept for retry, but the key is dead and that has been persisted so a
        # rollback cannot bring it back.
        remote_sandbox_service.db_session.delete.assert_not_called()
        assert stored_sandbox.session_api_key_hash is None
        remote_sandbox_service.db_session.commit.assert_awaited_once()


class TestSandboxSearch:
    """Test cases for sandbox search and retrieval."""

    @pytest.mark.asyncio
    async def test_search_sandboxes_basic(self, remote_sandbox_service):
        """Test basic sandbox search functionality."""
        # Setup
        stored_sandboxes = [
            create_stored_sandbox('sb1'),
            create_stored_sandbox('sb2'),
        ]

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = stored_sandboxes
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        remote_sandbox_service.db_session.execute = AsyncMock(return_value=mock_result)

        # Mock the batch endpoint response
        mock_batch_response = MagicMock()
        mock_batch_response.raise_for_status.return_value = None
        mock_batch_response.json.return_value = {
            'runtimes': [
                create_runtime_data('sb1'),
                create_runtime_data('sb2'),
            ]
        }
        remote_sandbox_service.httpx_client.request = AsyncMock(
            return_value=mock_batch_response
        )

        # Execute
        result = await remote_sandbox_service.search_sandboxes()

        # Verify
        assert len(result.items) == 2
        assert result.next_page_id is None
        assert result.items[0].id == 'sb1'
        assert result.items[1].id == 'sb2'

        # Verify that the batch endpoint was called
        remote_sandbox_service.httpx_client.request.assert_called_once_with(
            'GET',
            'https://api.example.com/sessions/batch',
            headers={'X-API-Key': 'test-api-key'},
            params=[('ids', 'sb1'), ('ids', 'sb2')],
        )

    @pytest.mark.asyncio
    async def test_search_sandboxes_with_pagination(self, remote_sandbox_service):
        """Test sandbox search with pagination."""
        # Setup - return limit + 1 items to trigger pagination
        stored_sandboxes = [
            create_stored_sandbox(f'sb{i}') for i in range(6)
        ]  # limit=5, so 6 items

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = stored_sandboxes
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        remote_sandbox_service.db_session.execute = AsyncMock(return_value=mock_result)

        # Mock the batch endpoint response
        mock_batch_response = MagicMock()
        mock_batch_response.raise_for_status.return_value = None
        mock_batch_response.json.return_value = {
            'runtimes': [create_runtime_data(f'sb{i}') for i in range(6)]
        }
        remote_sandbox_service.httpx_client.request = AsyncMock(
            return_value=mock_batch_response
        )

        # Execute
        result = await remote_sandbox_service.search_sandboxes(limit=5)

        # Verify
        assert len(result.items) == 5  # Should be limited to 5
        assert result.next_page_id == '5'  # Next page offset

    @pytest.mark.asyncio
    async def test_search_sandboxes_with_page_id(self, remote_sandbox_service):
        """Test sandbox search with page_id offset."""
        # Setup
        stored_sandboxes = [create_stored_sandbox('sb1')]

        mock_scalars = MagicMock()
        mock_scalars.all.return_value = stored_sandboxes
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        remote_sandbox_service.db_session.execute = AsyncMock(return_value=mock_result)

        # Mock the batch endpoint response
        mock_batch_response = MagicMock()
        mock_batch_response.raise_for_status.return_value = None
        mock_batch_response.json.return_value = {
            'runtimes': [create_runtime_data('sb1')]
        }
        remote_sandbox_service.httpx_client.request = AsyncMock(
            return_value=mock_batch_response
        )

        # Execute
        await remote_sandbox_service.search_sandboxes(page_id='10', limit=5)

        # Verify that offset was applied to the query
        # Note: We can't easily verify the exact SQL query, but we can verify the method was called
        remote_sandbox_service.db_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_runtimes_batch_success(self, remote_sandbox_service):
        """Test successful batch runtime retrieval."""
        # Setup
        sandbox_ids = ['sb1', 'sb2', 'sb3']
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = [
            create_runtime_data('sb1'),
            create_runtime_data('sb2'),
            create_runtime_data('sb3'),
        ]
        remote_sandbox_service.httpx_client.request = AsyncMock(
            return_value=mock_response
        )

        # Execute
        result = await remote_sandbox_service._get_runtimes_batch(sandbox_ids)

        # Verify
        assert len(result) == 3
        assert 'sb1' in result
        assert 'sb2' in result
        assert 'sb3' in result
        assert result['sb1']['session_id'] == 'sb1'

        # Verify the correct API call was made
        remote_sandbox_service.httpx_client.request.assert_called_once_with(
            'GET',
            'https://api.example.com/sessions/batch',
            headers={'X-API-Key': 'test-api-key'},
            params=[('ids', 'sb1'), ('ids', 'sb2'), ('ids', 'sb3')],
        )

    @pytest.mark.asyncio
    async def test_get_runtimes_batch_empty_list(self, remote_sandbox_service):
        """Test batch runtime retrieval with empty sandbox list."""
        # Execute
        result = await remote_sandbox_service._get_runtimes_batch([])

        # Verify
        assert result == {}
        # Verify no API call was made
        remote_sandbox_service.httpx_client.request.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_runtimes_batch_partial_results(self, remote_sandbox_service):
        """Test batch runtime retrieval with partial results (some sandboxes not found)."""
        # Setup
        sandbox_ids = ['sb1', 'sb2', 'sb3']
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = [
            create_runtime_data('sb1'),
            create_runtime_data('sb3'),
            # sb2 is missing from the response
        ]
        remote_sandbox_service.httpx_client.request = AsyncMock(
            return_value=mock_response
        )

        # Execute
        result = await remote_sandbox_service._get_runtimes_batch(sandbox_ids)

        # Verify
        assert len(result) == 2
        assert 'sb1' in result
        assert 'sb2' not in result  # Missing from response
        assert 'sb3' in result

    @pytest.mark.asyncio
    async def test_get_sandbox_exists(self, remote_sandbox_service):
        """Test getting an existing sandbox."""
        # Setup
        stored_sandbox = create_stored_sandbox()
        remote_sandbox_service._get_stored_sandbox = AsyncMock(
            return_value=stored_sandbox
        )
        remote_sandbox_service._to_sandbox_info = MagicMock(
            return_value=SandboxInfo(
                id='test-sandbox-123',
                created_by_user_id='test-user-123',
                sandbox_spec_id='test-image:latest',
                status=SandboxStatus.RUNNING,
                session_api_key='test-key',
                created_at=stored_sandbox.created_at,
            )
        )

        # Execute
        result = await remote_sandbox_service.get_sandbox('test-sandbox-123')

        # Verify
        assert result is not None
        assert result.id == 'test-sandbox-123'
        remote_sandbox_service._get_stored_sandbox.assert_called_once_with(
            'test-sandbox-123'
        )

    @pytest.mark.asyncio
    async def test_get_sandbox_not_exists(self, remote_sandbox_service):
        """Test getting a non-existent sandbox."""
        # Setup
        remote_sandbox_service._get_stored_sandbox = AsyncMock(return_value=None)

        # Execute
        result = await remote_sandbox_service.get_sandbox('non-existent')

        # Verify
        assert result is None


class TestUserSecurity:
    """Test cases for user-scoped operations and security."""

    @pytest.mark.asyncio
    async def test_secure_select_with_user_id(self, remote_sandbox_service):
        """Test that _secure_select filters by user ID."""
        # Setup
        remote_sandbox_service.user_context.get_user_id.return_value = 'test-user-123'

        # Execute
        await remote_sandbox_service._secure_select()

        # Verify
        # Note: We can't easily test the exact SQL query structure, but we can verify
        # that get_user_id was called, which means user filtering should be applied
        remote_sandbox_service.user_context.get_user_id.assert_called_once()

    @pytest.mark.asyncio
    async def test_secure_select_without_user_id(self, remote_sandbox_service):
        """Test that _secure_select works when user ID is None."""
        # Setup
        remote_sandbox_service.user_context.get_user_id.return_value = None

        # Execute
        await remote_sandbox_service._secure_select()

        # Verify
        remote_sandbox_service.user_context.get_user_id.assert_called_once()


class TestErrorHandling:
    """Test cases for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_resume_sandbox_http_error(self, remote_sandbox_service):
        """Test resume sandbox with HTTP error."""
        # Setup
        stored_sandbox = create_stored_sandbox()
        runtime_data = create_runtime_data()

        remote_sandbox_service._get_stored_sandbox = AsyncMock(
            return_value=stored_sandbox
        )
        remote_sandbox_service._get_runtime = AsyncMock(return_value=runtime_data)
        remote_sandbox_service.pause_old_sandboxes = AsyncMock(return_value=[])
        remote_sandbox_service.httpx_client.request.side_effect = httpx.HTTPError(
            'API Error'
        )

        # Execute
        result = await remote_sandbox_service.resume_sandbox('test-sandbox-123')

        # Verify
        assert result is False

    @pytest.mark.asyncio
    async def test_pause_sandbox_http_error(self, remote_sandbox_service):
        """Test pause sandbox with HTTP error."""
        # Setup
        stored_sandbox = create_stored_sandbox()
        runtime_data = create_runtime_data()

        remote_sandbox_service._get_stored_sandbox = AsyncMock(
            return_value=stored_sandbox
        )
        remote_sandbox_service._get_runtime = AsyncMock(return_value=runtime_data)
        remote_sandbox_service.httpx_client.request.side_effect = httpx.HTTPError(
            'API Error'
        )

        # Execute
        result = await remote_sandbox_service.pause_sandbox('test-sandbox-123')

        # Verify
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_sandbox_http_error(self, remote_sandbox_service):
        """Test delete sandbox with HTTP error."""
        # Setup
        stored_sandbox = create_stored_sandbox()
        runtime_data = create_runtime_data()

        remote_sandbox_service._get_stored_sandbox = AsyncMock(
            return_value=stored_sandbox
        )
        remote_sandbox_service._get_runtime = AsyncMock(return_value=runtime_data)
        remote_sandbox_service.db_session.delete = AsyncMock()
        remote_sandbox_service.db_session.commit = AsyncMock()
        remote_sandbox_service.httpx_client.request.side_effect = httpx.HTTPError(
            'API Error'
        )

        # A transient runtime error on a live sandbox raises (router 503), not 404.
        with pytest.raises(SandboxDeleteRetryError):
            await remote_sandbox_service.delete_sandbox('test-sandbox-123')


class TestGetSandboxBySessionApiKey:
    """Test cases for get_sandbox_by_session_api_key functionality."""

    @pytest.mark.asyncio
    async def test_get_sandbox_by_session_api_key_with_hash(
        self, remote_sandbox_service
    ):
        """Test finding sandbox by session API key using stored hash."""
        from openhands.app_server.sandbox.remote_sandbox_service import (
            _hash_session_api_key,
        )

        # Setup
        session_api_key = 'test-session-key'
        expected_hash = _hash_session_api_key(session_api_key)
        stored_sandbox = create_stored_sandbox(session_api_key_hash=expected_hash)
        runtime_data = create_runtime_data(session_api_key=session_api_key)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = stored_sandbox
        remote_sandbox_service.db_session.execute = AsyncMock(return_value=mock_result)
        remote_sandbox_service._get_runtime = AsyncMock(return_value=runtime_data)
        remote_sandbox_service.user_context.get_user_id.return_value = 'test-user-123'

        # Execute
        result = await remote_sandbox_service.get_sandbox_by_session_api_key(
            session_api_key
        )

        # Verify
        assert result is not None
        assert result.id == 'test-sandbox-123'
        assert result.session_api_key == session_api_key

    @pytest.mark.asyncio
    async def test_get_sandbox_by_session_api_key_not_found(
        self, remote_sandbox_service
    ):
        """Test that None is returned when no sandbox matches the session API key hash."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        remote_sandbox_service.db_session.execute = AsyncMock(return_value=mock_result)
        remote_sandbox_service.user_context.get_user_id.return_value = 'test-user-123'

        result = await remote_sandbox_service.get_sandbox_by_session_api_key(
            'unknown-key'
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_get_sandbox_by_session_api_key_runtime_error(
        self, remote_sandbox_service
    ):
        """Test handling runtime error when getting sandbox."""
        from openhands.app_server.sandbox.remote_sandbox_service import (
            _hash_session_api_key,
        )

        # Setup
        session_api_key = 'test-session-key'
        expected_hash = _hash_session_api_key(session_api_key)
        stored_sandbox = create_stored_sandbox(session_api_key_hash=expected_hash)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = stored_sandbox
        remote_sandbox_service.db_session.execute = AsyncMock(return_value=mock_result)
        remote_sandbox_service._get_runtime = AsyncMock(
            side_effect=Exception('Runtime error')
        )
        remote_sandbox_service.user_context.get_user_id.return_value = 'test-user-123'

        # Execute
        result = await remote_sandbox_service.get_sandbox_by_session_api_key(
            session_api_key
        )

        # Verify - should still return sandbox info, just with None runtime
        assert result is not None
        assert result.id == 'test-sandbox-123'
        assert result.status == SandboxStatus.MISSING  # No runtime means MISSING


class TestUtilityFunctions:
    """Test cases for utility functions."""

    def test_build_service_url_subdomain_mode(self):
        """Test _build_service_url function with subdomain-based routing."""
        from openhands.app_server.sandbox.remote_sandbox_service import (
            _build_service_url,
        )

        # Test HTTPS URL with path (subdomain mode)
        result = _build_service_url(
            'https://sandbox.example.com/path', 'vscode', 'runtime-123'
        )
        assert result == 'https://vscode-sandbox.example.com/path'

        # Test HTTP URL without path (subdomain mode)
        result = _build_service_url(
            'http://localhost:8000', 'work-1', 'different-runtime'
        )
        assert result == 'http://work-1-localhost:8000/'

        # Test URL with empty path (subdomain mode)
        result = _build_service_url('https://sandbox.example.com', 'work-2', 'some-id')
        assert result == 'https://work-2-sandbox.example.com/'

    def test_build_service_url_path_mode(self):
        """Test _build_service_url function with path-based routing."""
        from openhands.app_server.sandbox.remote_sandbox_service import (
            _build_service_url,
        )

        # Test path-based routing where URL path starts with /{runtime_id}
        result = _build_service_url(
            'https://sandbox.example.com/runtime-123', 'vscode', 'runtime-123'
        )
        assert result == 'https://sandbox.example.com/runtime-123/vscode'

        # Test path-based routing with work-1
        result = _build_service_url(
            'https://sandbox.example.com/my-runtime-id', 'work-1', 'my-runtime-id'
        )
        assert result == 'https://sandbox.example.com/my-runtime-id/work-1'

        # Test path-based routing with work-2
        result = _build_service_url(
            'http://localhost:8080/abc-xyz-123', 'work-2', 'abc-xyz-123'
        )
        assert result == 'http://localhost:8080/abc-xyz-123/work-2'

    def test_hash_session_api_key(self):
        """Test _hash_session_api_key function."""
        from openhands.app_server.sandbox.remote_sandbox_service import (
            _hash_session_api_key,
        )

        # Test that same input always produces same hash
        key = 'test-session-api-key'
        hash1 = _hash_session_api_key(key)
        hash2 = _hash_session_api_key(key)
        assert hash1 == hash2

        # Test that different inputs produce different hashes
        key2 = 'another-session-api-key'
        hash3 = _hash_session_api_key(key2)
        assert hash1 != hash3

        # Test that hash is a 64-character hex string (SHA-256)
        assert len(hash1) == 64
        assert all(c in '0123456789abcdef' for c in hash1)


class TestConstants:
    """Test cases for constants and mappings."""

    def test_status_mapping_completeness(self):
        """Test that STATUS_MAPPING covers expected statuses."""
        expected_statuses = ['running', 'paused', 'stopped', 'starting', 'error']
        for status in expected_statuses:
            assert status in STATUS_MAPPING, f'Missing status: {status}'

    def test_environment_variable_constants(self):
        """Test that environment variable constants are defined."""
        assert WEBHOOK_CALLBACK_VARIABLE == 'OH_WEBHOOKS_0_BASE_URL'
        assert ALLOW_CORS_ORIGINS_VARIABLE == 'OH_ALLOW_CORS_ORIGINS_0'


class TestRemoteSandboxRunningCleanup:
    """Test runtime-backed cleanup helpers for remote sandboxes."""

    def _mock_list_response(self, service, session_ids: list[str]):
        """Configure the httpx mock to return the given session_ids from /list."""
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'runtimes': [{'session_id': sid} for sid in session_ids]
        }
        service.httpx_client.request = AsyncMock(return_value=mock_response)

    def _mock_db_sandboxes(self, service, sandboxes: list):
        """Configure the DB mock to return the given StoredRemoteSandbox rows."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = sandboxes
        service.db_session.execute = AsyncMock(return_value=mock_result)

    @pytest.mark.asyncio
    async def test_get_user_running_sandboxes_cross_references_list_with_db(
        self, remote_sandbox_service
    ):
        """_get_user_running_sandboxes uses /list to filter DB records to running ones."""
        sb1 = create_stored_sandbox(sandbox_id='running-1')
        sb2 = create_stored_sandbox(sandbox_id='running-2')
        self._mock_list_response(remote_sandbox_service, ['running-1', 'running-2'])
        self._mock_db_sandboxes(remote_sandbox_service, [sb1, sb2])

        result = await remote_sandbox_service._get_user_running_sandboxes()

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_user_running_sandboxes_empty_when_none_running(
        self, remote_sandbox_service
    ):
        """Returns empty list when /list reports no running sessions for this user."""
        self._mock_list_response(remote_sandbox_service, [])
        self._mock_db_sandboxes(remote_sandbox_service, [])

        result = await remote_sandbox_service._get_user_running_sandboxes()

        assert result == []

    @pytest.mark.asyncio
    async def test_pause_old_sandboxes_pauses_oldest_when_over_limit(
        self, remote_sandbox_service
    ):
        """pause_old_sandboxes pauses the oldest sandboxes to reach max_num_sandboxes."""
        from datetime import timezone

        old = create_stored_sandbox(
            'old-sb', created_at=datetime(2024, 1, 1, tzinfo=timezone.utc)
        )
        new = create_stored_sandbox(
            'new-sb', created_at=datetime(2024, 6, 1, tzinfo=timezone.utc)
        )
        remote_sandbox_service._get_user_running_sandboxes = AsyncMock(
            return_value=[old, new]
        )
        remote_sandbox_service.pause_sandbox = AsyncMock(return_value=True)

        paused = await remote_sandbox_service.pause_old_sandboxes(max_num_sandboxes=1)

        assert paused == ['old-sb']
        remote_sandbox_service.pause_sandbox.assert_called_once_with('old-sb')

    @pytest.mark.asyncio
    async def test_pause_old_sandboxes_noop_when_within_limit(
        self, remote_sandbox_service
    ):
        """pause_old_sandboxes does nothing when running count is within the limit."""
        remote_sandbox_service._get_user_running_sandboxes = AsyncMock(
            return_value=[create_stored_sandbox('sb-1')]
        )
        remote_sandbox_service.pause_sandbox = AsyncMock(return_value=True)

        paused = await remote_sandbox_service.pause_old_sandboxes(max_num_sandboxes=3)

        assert paused == []
        remote_sandbox_service.pause_sandbox.assert_not_called()


def _async_cm_factory(value):
    """Return a callable that yields ``value`` as an async context manager.

    Mirrors ``get_*_service(state)``, which is used as
    ``async with get_x(state) as y``.
    """

    @asynccontextmanager
    async def _cm(*args, **kwargs):
        yield value

    return _cm


def _make_page(items, next_page_id=None):
    """Build a minimal object satisfying the page_iterator protocol."""
    page = MagicMock()
    page.items = items
    page.next_page_id = next_page_id
    return page


class _SessionTracker:
    """Counts how many mocked DB sessions are open at any instant.

    Used as the ``side_effect`` for a patched ``get_db_session``: every
    ``async with get_db_session(state)`` increments ``open`` on enter and
    decrements it on exit. Network-call probes record ``open`` at the moment
    they fire, so the regression guard ``open == 0 during network I/O`` can be
    asserted directly.
    """

    def __init__(self):
        self.open = 0
        self.enter_count = 0
        self.max_open = 0

    def __call__(self, *args, **kwargs):
        tracker = self

        class _Session:
            async def __aenter__(self):
                tracker.open += 1
                tracker.enter_count += 1
                tracker.max_open = max(tracker.max_open, tracker.open)
                return MagicMock()

            async def __aexit__(self, *exc):
                tracker.open -= 1
                return False

        return _Session()


class TestPollAgentServersSessionScoping:
    """Test cases for DB session scoping in poll_agent_servers and refresh_conversation.

    These tests verify that DB sessions are released before network I/O to
    prevent 'idle in transaction' issues. The key invariant:

    - No DB session/transaction is held open across an await'ed agent-server
      network call.

    A shared :class:`_SessionTracker` counts how many DB sessions are open at
    any instant, and every mocked agent-server network call records the count
    it observes. The regression guard is that this count is always ``0`` during
    network I/O. The mocks deliberately drive the *real* poll/refresh code
    paths (including the DB-write phases) so the assertions are not vacuous.
    """

    def _patches(
        self,
        tracker,
        httpx_client,
        conv_service,
        event_service,
        callback_service,
        validated_conv,
        event_pages,
    ):
        """Common patches shared by the tests in this class.

        Returns a list of ``patch`` context managers. ``ConversationInfo`` and
        ``EventPage`` validation is stubbed so the refresh code reaches its
        DB-write phases without needing fully-formed agent-server payloads.
        """
        mock_conv_info = MagicMock()
        mock_conv_info.model_validate.return_value = validated_conv
        mock_event_page = MagicMock()
        mock_event_page.model_validate.side_effect = list(event_pages)
        return [
            patch('openhands.app_server.config.get_db_session', side_effect=tracker),
            patch(
                'openhands.app_server.config.get_app_conversation_info_service',
                side_effect=_async_cm_factory(conv_service),
            ),
            patch(
                'openhands.app_server.config.get_event_service',
                side_effect=_async_cm_factory(event_service),
            ),
            patch(
                'openhands.app_server.config.get_event_callback_service',
                side_effect=_async_cm_factory(callback_service),
            ),
            patch(
                'openhands.app_server.config.get_httpx_client',
                side_effect=_async_cm_factory(httpx_client),
            ),
            patch(
                'openhands.app_server.sandbox.remote_sandbox_service.ConversationInfo',
                mock_conv_info,
            ),
            patch(
                'openhands.app_server.sandbox.remote_sandbox_service.EventPage',
                mock_event_page,
            ),
            patch('openhands.app_server.sandbox.remote_sandbox_service.InjectorState'),
            patch('openhands.app_server.sandbox.remote_sandbox_service.ADMIN'),
            patch(
                'openhands.app_server.sandbox.remote_sandbox_service.USER_CONTEXT_ATTR',
                'user_context',
            ),
        ]

    @staticmethod
    def _validated_conv():
        validated_conv = MagicMock()
        validated_conv.updated_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
        validated_conv.stats.get_combined_metrics.return_value = MagicMock()
        return validated_conv

    @staticmethod
    def _app_conv(sandbox_id='sandbox-1'):
        app_conv = MagicMock()
        app_conv.id = MagicMock(hex='c0ffee')
        app_conv.sandbox_id = sandbox_id
        app_conv.metrics = None
        return app_conv

    @pytest.mark.asyncio
    async def test_poll_agent_servers_releases_db_before_network_io(self):
        """poll_agent_servers must release the read session before network I/O."""
        from openhands.app_server.sandbox.remote_sandbox_service import (
            poll_agent_servers,
        )

        tracker = _SessionTracker()
        network_open_counts: list[int] = []

        app_conv = self._app_conv()
        list_payload = {
            'runtimes': [
                {
                    'session_id': 'sandbox-1',
                    'status': 'running',
                    'url': 'https://sandbox.example.com',
                    'session_api_key': 'key1',
                }
            ]
        }

        async def probe_get(url, *args, **kwargs):
            # Record open DB sessions at the moment of every agent-server call.
            network_open_counts.append(tracker.open)
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if url.endswith('/list'):
                resp.json.return_value = list_payload
            else:
                resp.json.return_value = {}
            return resp

        httpx_client = AsyncMock()
        httpx_client.get.side_effect = probe_get

        conv_service = AsyncMock()
        conv_service.search_app_conversation_info = AsyncMock(
            return_value=_make_page([app_conv])
        )
        conv_service.save_app_conversation_info = AsyncMock()
        event_service = AsyncMock()
        event_service.get_event = AsyncMock(return_value=None)
        callback_service = AsyncMock()

        patches = self._patches(
            tracker,
            httpx_client,
            conv_service,
            event_service,
            callback_service,
            self._validated_conv(),
            event_pages=[_make_page([], None)],
        )
        with ExitStack() as stack:
            for patch_cm in patches:
                stack.enter_context(patch_cm)
            task = asyncio.create_task(
                poll_agent_servers(
                    api_url='https://api.example.com',
                    api_key='test-key',
                    sleep_interval=3600,  # long, so cancellation ends the loop
                )
            )
            await asyncio.sleep(0.1)
            task.cancel()
            await task  # poll swallows CancelledError and returns

        # The runtime-list call and the per-conversation refresh both ran.
        assert network_open_counts, 'expected agent-server network calls to fire'
        assert len(network_open_counts) >= 2, (
            'expected at least the /list call plus a conversation refresh, '
            f'got {len(network_open_counts)} calls'
        )
        # Core regression guard: no DB session may be open during network I/O.
        assert all(count == 0 for count in network_open_counts), (
            'DB session held during network I/O (idle-in-transaction risk); '
            f'observed open-session counts at network calls: {network_open_counts}'
        )
        # Non-vacuous: the read session (Phase 1) and a write session were used.
        assert tracker.enter_count >= 2, (
            'expected the read session plus at least one write session to open'
        )
        assert tracker.max_open >= 1, 'expected at least one DB session to open'
        assert tracker.open == 0, 'all DB sessions must be released after polling'
        conv_service.save_app_conversation_info.assert_awaited()

    @pytest.mark.asyncio
    async def test_refresh_conversation_acquires_own_db_session(self):
        """refresh_conversation must open its own short-lived write sessions."""
        from openhands.app_server.sandbox.remote_sandbox_service import (
            refresh_conversation,
        )

        tracker = _SessionTracker()
        network_open_counts: list[int] = []

        async def probe_get(url, *args, **kwargs):
            network_open_counts.append(tracker.open)
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {}
            return resp

        httpx_client = AsyncMock()
        httpx_client.get.side_effect = probe_get

        conv_service = AsyncMock()
        conv_service.save_app_conversation_info = AsyncMock()
        event_service = AsyncMock()
        event_service.get_event = AsyncMock(return_value=None)
        event_service.save_event = AsyncMock()
        callback_service = AsyncMock()
        callback_service.execute_callbacks = AsyncMock()

        # One event on the first page, then an empty page to end pagination.
        event = MagicMock()
        event.id = str(uuid4())
        event_pages = [_make_page([event], 'page-2'), _make_page([], None)]

        patches = self._patches(
            tracker,
            httpx_client,
            conv_service,
            event_service,
            callback_service,
            self._validated_conv(),
            event_pages=event_pages,
        )
        runtime = {
            'url': 'https://sandbox.example.com',
            'session_api_key': 'test-key',
        }
        with ExitStack() as stack:
            for patch_cm in patches:
                stack.enter_context(patch_cm)
            await refresh_conversation(
                app_conversation_info=self._app_conv(),
                runtime=runtime,
                httpx_client=httpx_client,
            )

        # The write paths actually ran (otherwise the assertions are vacuous).
        conv_service.save_app_conversation_info.assert_awaited_once()
        event_service.save_event.assert_awaited_once()
        callback_service.execute_callbacks.assert_awaited_once()
        # refresh_conversation opened its own sessions: one for the conversation
        # save, one for the single new event.
        assert tracker.enter_count >= 2, (
            'refresh_conversation should acquire its own DB sessions for writes'
        )
        # Those sessions were short-lived and never overlapped network I/O.
        assert all(count == 0 for count in network_open_counts), (
            'DB session held during network I/O; observed open counts: '
            f'{network_open_counts}'
        )
        assert tracker.open == 0, 'all DB sessions must be released afterwards'

    @pytest.mark.asyncio
    async def test_db_session_not_held_across_network_call(self):
        """The key regression test: no DB session is open during a network call.

        Uses an artificial network delay so a held session would visibly span
        the await; the open-session count is sampled inside that window.
        """
        from openhands.app_server.sandbox.remote_sandbox_service import (
            refresh_conversation,
        )

        tracker = _SessionTracker()
        open_counts_during_network: list[int] = []

        async def slow_get(url, *args, **kwargs):
            # Sample the open-session count while the "network call" is mid-flight.
            await asyncio.sleep(0.01)
            open_counts_during_network.append(tracker.open)
            await asyncio.sleep(0.01)
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {}
            return resp

        httpx_client = AsyncMock()
        httpx_client.get.side_effect = slow_get

        conv_service = AsyncMock()
        conv_service.save_app_conversation_info = AsyncMock()
        event_service = AsyncMock()
        event_service.get_event = AsyncMock(return_value=None)
        event_service.save_event = AsyncMock()
        callback_service = AsyncMock()
        callback_service.execute_callbacks = AsyncMock()

        event = MagicMock()
        event.id = str(uuid4())
        event_pages = [_make_page([event], 'page-2'), _make_page([], None)]

        patches = self._patches(
            tracker,
            httpx_client,
            conv_service,
            event_service,
            callback_service,
            self._validated_conv(),
            event_pages=event_pages,
        )
        runtime = {
            'url': 'https://sandbox.example.com',
            'session_api_key': 'test-key',
        }
        with ExitStack() as stack:
            for patch_cm in patches:
                stack.enter_context(patch_cm)
            await refresh_conversation(
                app_conversation_info=self._app_conv(),
                runtime=runtime,
                httpx_client=httpx_client,
            )

        # The conversation fetch and at least one events fetch happened...
        assert len(open_counts_during_network) >= 2, (
            'expected the conversation fetch and an events fetch'
        )
        # ...and no DB session was open during any of them.
        assert all(count == 0 for count in open_counts_during_network), (
            'DB session must NOT be active during network I/O to prevent '
            f"'idle in transaction' issues; observed: {open_counts_during_network}"
        )
        # Non-vacuous: a write session really did open at some point.
        assert tracker.max_open >= 1, (
            'expected refresh_conversation to open a write session'
        )
        assert tracker.open == 0


class TestBatchGetSandboxes:
    """Test cases for batch_get_sandboxes method."""

    @pytest.mark.asyncio
    async def test_batch_get_sandboxes_success(self, remote_sandbox_service):
        """Test successful batch retrieval of sandboxes."""
        # Setup
        sandbox_ids = ['sandbox-1', 'sandbox-2']
        stored_sandbox_1 = create_stored_sandbox(sandbox_id='sandbox-1')
        stored_sandbox_2 = create_stored_sandbox(sandbox_id='sandbox-2')
        runtime_1 = create_runtime_data(session_id='sandbox-1', status='running')

        # Mock DB query result
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(
            return_value=iter([(stored_sandbox_1,), (stored_sandbox_2,)])
        )
        remote_sandbox_service.db_session.execute = AsyncMock(return_value=mock_result)

        # Mock successful runtime batch response
        remote_sandbox_service._get_runtimes_batch = AsyncMock(
            return_value={'sandbox-1': runtime_1}
        )

        # Execute
        results = await remote_sandbox_service.batch_get_sandboxes(sandbox_ids)

        # Verify
        assert len(results) == 2
        assert results[0] is not None
        assert results[0].id == 'sandbox-1'
        assert results[0].status == SandboxStatus.RUNNING
        assert results[1] is not None
        assert results[1].id == 'sandbox-2'
        # sandbox-2 has no runtime, so it's marked as MISSING
        assert results[1].status == SandboxStatus.MISSING

    @pytest.mark.asyncio
    async def test_batch_get_sandboxes_empty_ids(self, remote_sandbox_service):
        """Test batch retrieval with empty sandbox IDs list."""
        # Execute
        results = await remote_sandbox_service.batch_get_sandboxes([])

        # Verify
        assert results == []

    @pytest.mark.asyncio
    async def test_batch_get_sandboxes_graceful_fallback_on_timeout(
        self, remote_sandbox_service
    ):
        """Test that batch_get_sandboxes gracefully handles runtime API timeout.

        This is the key regression test: when the runtime API times out,
        batch_get_sandboxes should not raise but should return sandboxes
        with MISSING status (matching the behavior of get_sandbox).
        """
        # Setup
        sandbox_ids = ['sandbox-1', 'sandbox-2']
        stored_sandbox_1 = create_stored_sandbox(sandbox_id='sandbox-1')
        stored_sandbox_2 = create_stored_sandbox(sandbox_id='sandbox-2')

        # Mock DB query result
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(
            return_value=iter([(stored_sandbox_1,), (stored_sandbox_2,)])
        )
        remote_sandbox_service.db_session.execute = AsyncMock(return_value=mock_result)

        # Mock runtime API timeout
        remote_sandbox_service._get_runtimes_batch = AsyncMock(
            side_effect=httpx.TimeoutException('Request timeout')
        )

        # Execute - should NOT raise, should gracefully fall back
        results = await remote_sandbox_service.batch_get_sandboxes(sandbox_ids)

        # Verify - all sandboxes should be returned with MISSING status
        assert len(results) == 2
        assert results[0] is not None
        assert results[0].id == 'sandbox-1'
        assert results[0].status == SandboxStatus.MISSING
        assert results[1] is not None
        assert results[1].id == 'sandbox-2'
        assert results[1].status == SandboxStatus.MISSING

    @pytest.mark.asyncio
    async def test_batch_get_sandboxes_graceful_fallback_on_http_error(
        self, remote_sandbox_service
    ):
        """Test that batch_get_sandboxes gracefully handles runtime API HTTP error.

        When the runtime API returns an error (e.g., 500 Internal Server Error),
        batch_get_sandboxes should not raise but should return sandboxes
        with MISSING status.
        """
        # Setup
        sandbox_ids = ['sandbox-1']
        stored_sandbox_1 = create_stored_sandbox(sandbox_id='sandbox-1')

        # Mock DB query result
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([(stored_sandbox_1,)]))
        remote_sandbox_service.db_session.execute = AsyncMock(return_value=mock_result)

        # Mock runtime API HTTP error
        remote_sandbox_service._get_runtimes_batch = AsyncMock(
            side_effect=httpx.HTTPError('Internal server error')
        )

        # Execute - should NOT raise, should gracefully fall back
        results = await remote_sandbox_service.batch_get_sandboxes(sandbox_ids)

        # Verify - sandbox should be returned with MISSING status
        assert len(results) == 1
        assert results[0] is not None
        assert results[0].id == 'sandbox-1'
        assert results[0].status == SandboxStatus.MISSING

    @pytest.mark.asyncio
    async def test_batch_get_sandboxes_graceful_fallback_on_raise_for_status(
        self, remote_sandbox_service
    ):
        """Test graceful fallback when raise_for_status() fails in _get_runtimes_batch.

        The _get_runtimes_batch method calls response.raise_for_status().
        When this raises an httpx.HTTPStatusError, batch_get_sandboxes should
        catch it and fall back gracefully.
        """
        # Setup
        sandbox_ids = ['sandbox-1']
        stored_sandbox_1 = create_stored_sandbox(sandbox_id='sandbox-1')

        # Mock DB query result
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([(stored_sandbox_1,)]))
        remote_sandbox_service.db_session.execute = AsyncMock(return_value=mock_result)

        # Mock HTTP status error from raise_for_status()
        remote_sandbox_service._get_runtimes_batch = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                message='500 Internal Server Error',
                request=MagicMock(),
                response=MagicMock(status_code=500),
            )
        )

        # Execute - should NOT raise, should gracefully fall back
        results = await remote_sandbox_service.batch_get_sandboxes(sandbox_ids)

        # Verify - sandbox should be returned with MISSING status
        assert len(results) == 1
        assert results[0] is not None
        assert results[0].id == 'sandbox-1'
        assert results[0].status == SandboxStatus.MISSING


class TestArchiveConversationWorkspace:
    """archive_conversation_workspace contract (APP-2403): the sole app-server
    capture path, run by the conversation-delete finalizer BEFORE delete_sandbox
    (which is now purely sandbox-scoped). Its bool return gates the finalizer —
    False (REQUIRED + retryable failure) keeps the sandbox for the idle reap; True
    lets the delete proceed. It never stops the runtime or deletes the row."""

    @pytest.mark.asyncio
    async def test_required_failure_returns_false(self, remote_sandbox_service):
        """A REQUIRED archive failure returns False (and never stops/deletes), so
        the finalizer keeps the sandbox + runtime for the idle reap."""
        stored = create_stored_sandbox()
        remote_sandbox_service._get_stored_sandbox = AsyncMock(return_value=stored)
        remote_sandbox_service._get_runtime = AsyncMock(
            return_value=create_runtime_data()
        )
        remote_sandbox_service.db_session.delete = AsyncMock()
        remote_sandbox_service.httpx_client.request = AsyncMock()

        with (
            patch(
                'openhands.app_server.sandbox.workspace_archive.archive_enabled',
                return_value=True,
            ),
            patch(
                'openhands.app_server.sandbox.workspace_archive.archive_workspace',
                new=AsyncMock(return_value=False),
            ),
        ):
            ok = await remote_sandbox_service.archive_conversation_workspace(
                'test-sandbox-123', conversation_id='conv-1'
            )

        assert ok is False
        remote_sandbox_service.httpx_client.request.assert_not_called()
        remote_sandbox_service.db_session.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_nothing_to_archive_returns_true(self, remote_sandbox_service):
        """Nothing to capture (archive_workspace returns True for a 400/404) lets
        the delete proceed even under REQUIRED, so the sandbox is never orphaned."""
        stored = create_stored_sandbox()
        remote_sandbox_service._get_stored_sandbox = AsyncMock(return_value=stored)
        remote_sandbox_service._get_runtime = AsyncMock(
            return_value=create_runtime_data()
        )

        with (
            patch(
                'openhands.app_server.sandbox.workspace_archive.archive_enabled',
                return_value=True,
            ),
            patch(
                'openhands.app_server.sandbox.workspace_archive.archive_workspace',
                new=AsyncMock(return_value=True),
            ),
        ):
            ok = await remote_sandbox_service.archive_conversation_workspace(
                'test-sandbox-123', conversation_id='conv-1'
            )

        assert ok is True

    @pytest.mark.asyncio
    async def test_runtime_already_gone_returns_true(self, remote_sandbox_service):
        """Runtime already gone (404): nothing to capture, so the delete proceeds."""
        stored = create_stored_sandbox()
        remote_sandbox_service._get_stored_sandbox = AsyncMock(return_value=stored)
        remote_sandbox_service._get_runtime = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                'gone',
                request=httpx.Request('GET', 'https://api.example.com/x'),
                response=httpx.Response(404),
            )
        )
        with patch(
            'openhands.app_server.sandbox.workspace_archive.archive_enabled',
            return_value=True,
        ):
            ok = await remote_sandbox_service.archive_conversation_workspace(
                'test-sandbox-123', conversation_id='conv-1'
            )
        assert ok is True

    @pytest.mark.asyncio
    async def test_pinned_path_forwarded_to_archive(self, remote_sandbox_service):
        """A pinned workspace_path is forwarded to archive_workspace verbatim and
        no grouping/spec re-derivation happens."""
        stored = create_stored_sandbox()
        remote_sandbox_service._get_stored_sandbox = AsyncMock(return_value=stored)
        remote_sandbox_service._get_runtime = AsyncMock(
            return_value=create_runtime_data()
        )
        with (
            patch(
                'openhands.app_server.sandbox.workspace_archive.archive_enabled',
                return_value=True,
            ),
            patch(
                'openhands.app_server.sandbox.workspace_archive.archive_workspace',
                new=AsyncMock(return_value=True),
            ) as mock_archive,
        ):
            ok = await remote_sandbox_service.archive_conversation_workspace(
                'test-sandbox-123',
                conversation_id='conv-1',
                workspace_path='/home/openhands/workspace/conv-1',
            )
        assert ok is True
        _, kwargs = mock_archive.call_args
        assert kwargs['archive_path'] == '/home/openhands/workspace/conv-1'
        remote_sandbox_service.sandbox_spec_service.get_sandbox_spec.assert_not_called()

    @pytest.mark.asyncio
    async def test_resolve_archive_path_uses_pinned(self, remote_sandbox_service):
        """A pinned path is returned verbatim — no spec/settings lookup, so a
        grouping toggle since creation can never misroute the capture."""
        stored = create_stored_sandbox()
        path = await remote_sandbox_service._resolve_archive_path(
            stored, 'conv-1', '/home/openhands/workspace/conv-1'
        )
        assert path == '/home/openhands/workspace/conv-1'
        remote_sandbox_service.sandbox_spec_service.get_sandbox_spec.assert_not_called()

    @pytest.mark.asyncio
    async def test_resolve_archive_path_legacy_grouped(self, remote_sandbox_service):
        """No pinned path (pre-pinning conversation): rebuilt from the sandbox
        spec's working_dir nested under the conversation id."""
        stored = create_stored_sandbox()
        remote_sandbox_service.user_context.get_user_info = AsyncMock(
            return_value=MagicMock(
                sandbox_grouping_strategy=SandboxGroupingStrategy.GROUP_BY_NEWEST
            )
        )
        path = await remote_sandbox_service._resolve_archive_path(
            stored, 'deadbeef', None
        )
        # mock_sandbox_spec_service fixture pins working_dir='/workspace/project'.
        assert path == '/workspace/project/deadbeef'

    @pytest.mark.asyncio
    async def test_resolve_archive_path_legacy_no_grouping(
        self, remote_sandbox_service
    ):
        """NO_GROUPING fallback keeps the bare spec working_dir."""
        stored = create_stored_sandbox()
        remote_sandbox_service.user_context.get_user_info = AsyncMock(
            return_value=MagicMock(
                sandbox_grouping_strategy=SandboxGroupingStrategy.NO_GROUPING
            )
        )
        path = await remote_sandbox_service._resolve_archive_path(
            stored, 'deadbeef', None
        )
        assert path == '/workspace/project'


class TestArchiveWorkspaceHelper:
    """Unit tests for the workspace_archive helper."""

    @pytest.mark.asyncio
    async def test_archive_uploads_archive_and_manifest(self, monkeypatch):
        from openhands.app_server.sandbox import workspace_archive

        monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_BUCKET', 'archive-bkt')
        monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_FORMAT', 'git-delta')

        archive_response = _make_stream_response(
            200,
            b'patch-bytes',
            {
                'X-Archive-Base-Commit': 'abc123',
                'X-Archive-Repo-Remote': (
                    'https%3A%2F%2Fgithub.com%2Fexample%2Frepo.git'
                ),
                'X-Archive-Branch': 'feature-x',
                'X-Archive-Head-Commit': 'def456',
                'X-Archive-Repo-Root': '%2Fworkspace%2Fproject%2Frepo',
            },
        )
        client = _stream_client(archive_response)
        store = MagicMock()
        writes: dict[str, bytes] = {}
        store.write.side_effect = lambda path, data: writes.__setitem__(path, data)
        # Archive blobs are streamed from a tempfile via write_from_path (the OOM
        # fix); record them too so blob-path assertions still see the .patch/.tar.gz.
        store.write_from_path.side_effect = lambda path, src: writes.__setitem__(
            path, open(src, 'rb').read()
        )

        probe_workspace = AsyncMock()

        async def _probe(*args):
            assert archive_response.stream_closed
            return {'packages': {'npm': {'example': '1.0.0'}}}

        probe_workspace.side_effect = _probe
        with (
            patch.object(
                workspace_archive, '_get_archive_file_store', return_value=store
            ),
            patch.object(workspace_archive, '_probe_workspace', probe_workspace),
        ):
            ok = await workspace_archive.archive_workspace(
                client,
                create_runtime_data(),
                'sandbox-1',
                archive_path='/workspace/project',
                conversation_id='conv-1',
            )

        assert ok is True
        # session key forwarded; the given archive_path is used, format passed.
        _, kwargs = client.stream.call_args
        assert kwargs['headers']['X-Session-API-Key'] == 'test-session-key'
        assert kwargs['params'] == {
            'path': '/workspace/project',
            'format': 'git-delta',
        }
        # archive + manifest written.
        assert any(p.endswith('.patch') for p in writes)
        manifest_path = next(p for p in writes if p.endswith('.manifest.json'))
        manifest = json.loads(writes[manifest_path])
        assert manifest['base_commit'] == 'abc123'
        assert manifest['conversation_id'] == 'conv-1'
        assert manifest['source_path'] == '/workspace/project'
        # Repo identity from the response headers makes the blob self-describing.
        assert manifest['repo_remote'] == 'https://github.com/example/repo.git'
        assert manifest['branch'] == 'feature-x'
        assert manifest['head_commit'] == 'def456'
        assert manifest['packages'] == {'npm': {'example': '1.0.0'}}
        probe_workspace.assert_awaited_once_with(
            client,
            'https://sandbox.example.com',
            {'X-Session-API-Key': 'test-session-key'},
            '/workspace/project/repo',
        )

    @pytest.mark.asyncio
    async def test_archive_missing_base_commit_header_defaults_empty(self, monkeypatch):
        from openhands.app_server.sandbox import workspace_archive

        monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_BUCKET', 'archive-bkt')
        monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_FORMAT', 'git-delta')

        client = _stream_client(_make_stream_response(200, b'patch-bytes', {}))
        store = MagicMock()
        writes: dict[str, bytes] = {}
        store.write.side_effect = lambda path, data: writes.__setitem__(path, data)
        # Archive blobs are streamed from a tempfile via write_from_path (the OOM
        # fix); record them too so blob-path assertions still see the .patch/.tar.gz.
        store.write_from_path.side_effect = lambda path, src: writes.__setitem__(
            path, open(src, 'rb').read()
        )

        with patch.object(
            workspace_archive, '_get_archive_file_store', return_value=store
        ):
            ok = await workspace_archive.archive_workspace(
                client,
                create_runtime_data(),
                'sandbox-1',
                archive_path='/workspace/project',
            )

        assert ok is True
        manifest_path = next(p for p in writes if p.endswith('.manifest.json'))
        manifest = json.loads(writes[manifest_path])
        assert manifest['base_commit'] == ''
        # Repo identity also degrades gracefully when the headers are absent.
        assert manifest['repo_remote'] == ''
        assert manifest['branch'] == ''
        assert manifest['head_commit'] == ''
        # Prod case: delete_sandbox passes no conversation_id, so the manifest
        # falls back to the sandbox id (which is the conversation_id.hex).
        assert manifest['conversation_id'] == 'sandbox-1'

    @pytest.mark.asyncio
    async def test_archive_non_200_not_required_allows_delete(self, monkeypatch):
        from openhands.app_server.sandbox import workspace_archive

        monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_BUCKET', 'archive-bkt')
        monkeypatch.delenv('RUNTIME_FILE_ARCHIVE_REQUIRED', raising=False)

        client = _stream_client(_make_stream_response(500))

        ok = await workspace_archive.archive_workspace(
            client,
            create_runtime_data(),
            'sandbox-1',
            archive_path='/workspace/project',
        )
        # Not required -> best-effort -> deletion may proceed.
        assert ok is True

    @pytest.mark.asyncio
    async def test_archive_non_200_required_blocks_delete(self, monkeypatch):
        from openhands.app_server.sandbox import workspace_archive

        monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_BUCKET', 'archive-bkt')
        monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_REQUIRED', 'true')

        client = _stream_client(_make_stream_response(500))

        ok = await workspace_archive.archive_workspace(
            client,
            create_runtime_data(),
            'sandbox-1',
            archive_path='/workspace/project',
        )
        # Required -> a failure blocks deletion.
        assert ok is False

    @pytest.mark.asyncio
    async def test_archive_no_url_skips(self, monkeypatch):
        from openhands.app_server.sandbox import workspace_archive

        monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_BUCKET', 'archive-bkt')
        monkeypatch.delenv('RUNTIME_FILE_ARCHIVE_REQUIRED', raising=False)
        client = AsyncMock()

        ok = await workspace_archive.archive_workspace(
            client,
            create_runtime_data(url=''),
            'sandbox-1',
            archive_path='/workspace/project',
        )
        assert ok is True
        client.stream.assert_not_called()

    @pytest.mark.asyncio
    async def test_archive_400_no_repo_required_allows_delete(self, monkeypatch):
        """No git repo to capture is a permanent 400 from the agent-server, not a
        failure: even when archiving is required, the delete may proceed (BUG 2)."""
        from openhands.app_server.sandbox import workspace_archive

        monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_BUCKET', 'archive-bkt')
        monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_REQUIRED', 'true')

        # "Not a git repository" from GET /api/file/archive
        client = _stream_client(_make_stream_response(400))

        ok = await workspace_archive.archive_workspace(
            client,
            create_runtime_data(),
            'sandbox-1',
            archive_path='/workspace/project',
        )
        # Permanent 4xx -> nothing to lose -> deletion proceeds despite REQUIRED.
        assert ok is True

    @pytest.mark.asyncio
    async def test_archive_404_required_blocks_delete(self, monkeypatch):
        """A 404 (path missing) is the misrouted-path symptom, NOT a confirmed-empty
        workspace: under REQUIRED it must block the delete (keep for the idle reap)
        rather than silently tearing the sandbox down uncaptured."""
        from openhands.app_server.sandbox import workspace_archive

        monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_BUCKET', 'archive-bkt')
        monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_REQUIRED', 'true')

        client = _stream_client(_make_stream_response(404))

        ok = await workspace_archive.archive_workspace(
            client,
            create_runtime_data(),
            'sandbox-1',
            archive_path='/workspace/project',
        )
        assert ok is False

    @pytest.mark.asyncio
    async def test_archive_500_required_blocks_delete(self, monkeypatch):
        """A 5xx is a genuine transient failure: required archiving blocks the
        delete so it can be retried (BUG 2 — only 5xx/network block)."""
        from openhands.app_server.sandbox import workspace_archive

        monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_BUCKET', 'archive-bkt')
        monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_REQUIRED', 'true')

        client = _stream_client(_make_stream_response(500))

        ok = await workspace_archive.archive_workspace(
            client,
            create_runtime_data(),
            'sandbox-1',
            archive_path='/workspace/project',
        )
        assert ok is False

    @pytest.mark.asyncio
    async def test_archive_422_required_blocks_delete(self, monkeypatch):
        """A 422 (malformed request from the producer) is NOT "nothing to archive"
        — under REQUIRED it is treated as retryable and must block the delete
        instead of being misread as permanent (infra#1444 M1)."""
        from openhands.app_server.sandbox import workspace_archive

        monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_BUCKET', 'archive-bkt')
        monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_REQUIRED', 'true')

        client = _stream_client(_make_stream_response(422))

        ok = await workspace_archive.archive_workspace(
            client,
            create_runtime_data(),
            'sandbox-1',
            archive_path='/workspace/project',
        )
        assert ok is False

    @pytest.mark.asyncio
    async def test_archive_401_required_blocks_delete(self, monkeypatch):
        """A 401 (session key rejected) means the capture did NOT happen and is not
        a confirmed-empty workspace, so under REQUIRED it must block the delete
        (keep for the idle reap) rather than tear the sandbox down uncaptured."""
        from openhands.app_server.sandbox import workspace_archive

        monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_BUCKET', 'archive-bkt')
        monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_REQUIRED', 'true')

        client = _stream_client(_make_stream_response(401))

        ok = await workspace_archive.archive_workspace(
            client,
            create_runtime_data(),
            'sandbox-1',
            archive_path='/workspace/project',
        )
        assert ok is False

    @pytest.mark.asyncio
    async def test_archive_unsupported_format_proceeds_and_skips_request(
        self, monkeypatch
    ):
        """An unsupported RUNTIME_FILE_ARCHIVE_FORMAT (e.g. the removed 'zip') is a
        pure config error caught up front: no request is issued, and the delete
        PROCEEDS even under REQUIRED. Like an unset bucket, no retry makes a valid
        format appear, so blocking would wedge every delete forever with no
        app-server backstop; the producer never sees the bad format."""
        from openhands.app_server.sandbox import workspace_archive

        monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_BUCKET', 'archive-bkt')
        monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_FORMAT', 'zip')
        monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_REQUIRED', 'true')

        client = AsyncMock()

        ok = await workspace_archive.archive_workspace(
            client,
            create_runtime_data(),
            'sandbox-1',
            archive_path='/workspace/project',
        )
        assert ok is True
        client.stream.assert_not_called()

    @pytest.mark.asyncio
    async def test_archive_unsupported_format_not_required_proceeds(self, monkeypatch):
        from openhands.app_server.sandbox import workspace_archive

        monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_BUCKET', 'archive-bkt')
        monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_FORMAT', 'zip')
        monkeypatch.delenv('RUNTIME_FILE_ARCHIVE_REQUIRED', raising=False)

        client = AsyncMock()

        ok = await workspace_archive.archive_workspace(
            client,
            create_runtime_data(),
            'sandbox-1',
            archive_path='/workspace/project',
        )
        assert ok is True
        client.stream.assert_not_called()

    @pytest.mark.asyncio
    async def test_archive_no_bucket_required_proceeds(self, monkeypatch):
        """RUNTIME_FILE_ARCHIVE_BUCKET unset is a config error no retry can fix;
        it must NOT wedge every delete under REQUIRED — proceed loudly instead
        (infra#1444 L8)."""
        from openhands.app_server.sandbox import workspace_archive

        monkeypatch.delenv('RUNTIME_FILE_ARCHIVE_BUCKET', raising=False)
        monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_REQUIRED', 'true')

        client = AsyncMock()

        ok = await workspace_archive.archive_workspace(
            client,
            create_runtime_data(),
            'sandbox-1',
            archive_path='/workspace/project',
        )
        assert ok is True
        client.stream.assert_not_called()

    @pytest.mark.asyncio
    async def test_archive_uses_given_path_verbatim(self, monkeypatch):
        """archive_workspace targets the caller-resolved archive_path exactly (the
        path was pinned at creation; the grouping nesting is resolved upstream in
        RemoteSandboxService._resolve_archive_path)."""
        from openhands.app_server.sandbox import workspace_archive

        monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_BUCKET', 'archive-bkt')
        monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_FORMAT', 'git-delta')

        client = _stream_client(_make_stream_response(200, b'patch-bytes', {}))
        store = MagicMock()
        writes: dict[str, bytes] = {}
        store.write.side_effect = lambda path, data: writes.__setitem__(path, data)
        store.write_from_path.side_effect = lambda path, src: writes.__setitem__(
            path, open(src, 'rb').read()
        )

        with patch.object(
            workspace_archive, '_get_archive_file_store', return_value=store
        ):
            ok = await workspace_archive.archive_workspace(
                client,
                create_runtime_data(),
                'sandbox-1',
                archive_path='/workspace/project/deadbeef',
                conversation_id='deadbeef',
            )

        assert ok is True
        _, kwargs = client.stream.call_args
        assert kwargs['params']['path'] == '/workspace/project/deadbeef'
        manifest_path = next(p for p in writes if p.endswith('.manifest.json'))
        manifest = json.loads(writes[manifest_path])
        assert manifest['source_path'] == '/workspace/project/deadbeef'
        assert manifest['conversation_id'] == 'deadbeef'

    @pytest.mark.asyncio
    async def test_archive_both_uploads_delta_and_targz(self, monkeypatch):
        """FORMAT=both (the default) captures git-delta AND a full tar.gz, each
        with its own artifact + manifest; base_commit rides the git-delta header
        and is reused for the tar.gz manifest."""
        from openhands.app_server.sandbox import workspace_archive

        monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_BUCKET', 'archive-bkt')
        monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_FORMAT', 'both')

        responses = {
            'git-delta': _make_stream_response(
                200, b'patch-bytes', {'X-Archive-Base-Commit': 'abc123'}
            ),
            'tar.gz': _make_stream_response(200, b'tar-bytes', {}),
        }
        client = _stream_client(responses)
        store = MagicMock()
        writes: dict[str, bytes] = {}
        store.write.side_effect = lambda path, data: writes.__setitem__(path, data)
        # Archive blobs are streamed from a tempfile via write_from_path (the OOM
        # fix); record them too so blob-path assertions still see the .patch/.tar.gz.
        store.write_from_path.side_effect = lambda path, src: writes.__setitem__(
            path, open(src, 'rb').read()
        )

        with patch.object(
            workspace_archive, '_get_archive_file_store', return_value=store
        ):
            ok = await workspace_archive.archive_workspace(
                client,
                create_runtime_data(),
                'sandbox-1',
                archive_path='/workspace/project',
                conversation_id='conv-1',
            )

        assert ok is True
        # Two artifacts under the same {ts} base, one per format.
        assert any(p.endswith('.patch') for p in writes)
        assert any(p.endswith('.tar.gz') for p in writes)
        patch_manifest = json.loads(
            writes[next(p for p in writes if p.endswith('.patch.manifest.json'))]
        )
        targz_manifest = json.loads(
            writes[next(p for p in writes if p.endswith('.tar.gz.manifest.json'))]
        )
        assert patch_manifest['format'] == 'git-delta'
        assert targz_manifest['format'] == 'tar.gz'
        # base_commit from the git-delta header is carried onto the tar.gz too.
        assert patch_manifest['base_commit'] == 'abc123'
        assert targz_manifest['base_commit'] == 'abc123'
        assert patch_manifest['phase'] == targz_manifest['phase'] == 'final'

    @pytest.mark.asyncio
    async def test_archive_both_required_blocks_if_one_format_fails(self, monkeypatch):
        """In REQUIRED both-mode, a retryable failure on either format blocks the
        delete so a retry can capture the complete pair."""
        from openhands.app_server.sandbox import workspace_archive

        monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_BUCKET', 'archive-bkt')
        monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_FORMAT', 'both')
        monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_REQUIRED', 'true')

        responses = {
            'git-delta': _make_stream_response(
                200, b'patch-bytes', {'X-Archive-Base-Commit': 'abc123'}
            ),
            'tar.gz': _make_stream_response(500),
        }
        client = _stream_client(responses)
        store = MagicMock()
        store.write.side_effect = lambda path, data: None
        store.write_from_path.side_effect = lambda path, src: None

        with patch.object(
            workspace_archive, '_get_archive_file_store', return_value=store
        ):
            ok = await workspace_archive.archive_workspace(
                client,
                create_runtime_data(),
                'sandbox-1',
                archive_path='/workspace/project',
            )

        # tar.gz 500 is retryable + REQUIRED -> block the delete for a retry.
        assert ok is False

    @pytest.mark.asyncio
    async def test_archive_sibling_conversations_distinct_keys(self, monkeypatch):
        """Two sibling conversations on the SAME grouped sandbox, captured in the
        same second, must land at distinct blob keys: the conversation id is in the
        object path, not just the manifest body (regression — same-second overwrite
        silently dropped one conversation's archive)."""
        from datetime import datetime, timezone

        from openhands.app_server.sandbox import workspace_archive

        monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_BUCKET', 'archive-bkt')
        monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_FORMAT', 'git-delta')
        # Pin the timestamp so both captures compute the SAME {ts}; only the
        # conversation segment can keep them apart.
        fixed = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(workspace_archive, 'utc_now', lambda: fixed)

        writes: dict[str, bytes] = {}
        store = MagicMock()
        store.write.side_effect = lambda path, data: writes.__setitem__(path, data)
        store.write_from_path.side_effect = lambda path, src: writes.__setitem__(
            path, open(src, 'rb').read()
        )

        with patch.object(
            workspace_archive, '_get_archive_file_store', return_value=store
        ):
            for conv in ('conva', 'convb'):
                client = _stream_client(_make_stream_response(200, b'patch-bytes', {}))
                ok = await workspace_archive.archive_workspace(
                    client,
                    create_runtime_data(),
                    'shared-sandbox',
                    archive_path=f'/workspace/project/{conv}',
                    conversation_id=conv,
                )
                assert ok is True

        # Both artifacts AND both manifests survive — nothing overwritten.
        patch_blobs = [p for p in writes if p.endswith('.patch')]
        manifests = [p for p in writes if p.endswith('.manifest.json')]
        assert len(patch_blobs) == 2
        assert len(manifests) == 2
        assert any('/shared-sandbox/conva/' in p for p in patch_blobs)
        assert any('/shared-sandbox/convb/' in p for p in patch_blobs)


class TestDeleteSandboxKeyHandling:
    """The session_api_key_hash is invalidated UP FRONT on delete (a delete is
    often a revoke of a leaked key). When a transient error keeps the row for
    retry, the invalidation is committed first so the DELETE route's rollback
    cannot resurrect the key. Backed by a real SQLite session so persistence is
    provable across a rollback."""

    @pytest.fixture
    async def async_engine(self):
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy.pool import StaticPool

        from openhands.app_server.utils.sql_utils import Base

        engine = create_async_engine(
            'sqlite+aiosqlite:///:memory:',
            poolclass=StaticPool,
            connect_args={'check_same_thread': False},
            echo=False,
        )
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        yield engine
        await engine.dispose()

    @pytest.fixture
    async def real_session(self, async_engine):
        from sqlalchemy.ext.asyncio import async_sessionmaker

        maker = async_sessionmaker(
            async_engine, class_=AsyncSession, expire_on_commit=False
        )
        async with maker() as session:
            yield session

    @pytest.fixture
    def service_with_real_db(
        self, mock_sandbox_spec_service, mock_user_context, real_session
    ):
        return RemoteSandboxService(
            sandbox_spec_service=mock_sandbox_spec_service,
            api_url='https://api.example.com',
            api_key='test-api-key',
            web_url='https://web.example.com',
            resource_factor=1,
            runtime_class='gvisor',
            start_sandbox_timeout=120,
            max_num_sandboxes=10,
            user_context=mock_user_context,
            httpx_client=AsyncMock(spec=httpx.AsyncClient),
            db_session=real_session,
        )

    @pytest.mark.asyncio
    async def test_session_key_invalidated_and_survives_rollback(
        self, service_with_real_db, real_session
    ):
        # Seed a running sandbox with a live session key hash.
        row = create_stored_sandbox(session_api_key_hash='live-hash')
        real_session.add(row)
        await real_session.commit()

        # Transient (non-404) runtime error -> delete_sandbox raises (keeps the
        # row + running runtime for retry) without deleting the row.
        service_with_real_db._get_runtime = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                'Server Error',
                request=httpx.Request('GET', 'https://api.example.com/sessions/x'),
                response=httpx.Response(503),
            )
        )

        with pytest.raises(SandboxDeleteRetryError):
            await service_with_real_db.delete_sandbox('test-sandbox-123')

        # Simulate the DELETE route's session rolling back the failed delete.
        await real_session.rollback()
        real_session.expire_all()

        # Row kept for retry, but the key is DEAD and the rollback could not bring
        # it back — delete_sandbox committed the invalidation before raising. This
        # is the security fix: a leaked key is revoked even when the stop fails.
        refreshed = await service_with_real_db._get_stored_sandbox('test-sandbox-123')
        assert refreshed is not None
        assert refreshed.session_api_key_hash is None

    @pytest.mark.asyncio
    async def test_session_key_cleared_on_successful_delete(
        self, service_with_real_db, real_session, monkeypatch
    ):
        # Archiving off -> a clean delete goes straight to /stop + row delete.
        monkeypatch.delenv('RUNTIME_FILE_ARCHIVE_ENABLED', raising=False)
        row = create_stored_sandbox(session_api_key_hash='live-hash')
        real_session.add(row)
        await real_session.commit()

        service_with_real_db._get_runtime = AsyncMock(
            return_value={'runtime_id': 'rt-123'}
        )
        stop_response = MagicMock()
        stop_response.status_code = 200
        stop_response.raise_for_status.return_value = None
        service_with_real_db._send_runtime_api_request = AsyncMock(
            return_value=stop_response
        )

        assert await service_with_real_db.delete_sandbox('test-sandbox-123') is True
        await real_session.commit()
        real_session.expire_all()

        # The row — and with it the session key — is gone after a clean delete.
        refreshed = await service_with_real_db._get_stored_sandbox('test-sandbox-123')
        assert refreshed is None


class TestArchiveEnvToggles:
    """Per OpenHands/AGENTS.md, env enable-toggles must accept '1' as well as
    'true' (older Helm charts default these to '1'). A regression to == 'true'
    would silently disable the entire workspace-archive capture, so pin both."""

    @pytest.mark.parametrize('value', ['1', 'true', 'TRUE', 'True'])
    def test_archive_enabled_truthy(self, monkeypatch, value):
        from openhands.app_server.sandbox import workspace_archive

        monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_ENABLED', value)
        assert workspace_archive.archive_enabled() is True

    @pytest.mark.parametrize('value', ['0', 'false', '', 'no'])
    def test_archive_enabled_falsy(self, monkeypatch, value):
        from openhands.app_server.sandbox import workspace_archive

        monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_ENABLED', value)
        assert workspace_archive.archive_enabled() is False

    @pytest.mark.parametrize('value', ['1', 'true'])
    def test_archive_required_truthy(self, monkeypatch, value):
        from openhands.app_server.sandbox import workspace_archive

        monkeypatch.setenv('RUNTIME_FILE_ARCHIVE_REQUIRED', value)
        assert workspace_archive.archive_required() is True

    def test_toggles_default_off(self, monkeypatch):
        from openhands.app_server.sandbox import workspace_archive

        for var in (
            'RUNTIME_FILE_ARCHIVE_ENABLED',
            'RUNTIME_FILE_ARCHIVE_REQUIRED',
        ):
            monkeypatch.delenv(var, raising=False)
        assert workspace_archive.archive_enabled() is False
        assert workspace_archive.archive_required() is False


class TestArchiveRequestParams:
    """The tar.gz is the self-contained full capture, so it disables the
    endpoint's default excludes; git-delta stays compact (keeps them)."""

    def test_tar_gz_disables_default_excludes(self):
        from openhands.app_server.sandbox import workspace_archive

        params = workspace_archive._archive_request_params(
            '/workspace/project', 'tar.gz'
        )
        assert params == {
            'path': '/workspace/project',
            'format': 'tar.gz',
            'use_default_excludes': 'false',
        }

    def test_git_delta_keeps_default_excludes(self):
        from openhands.app_server.sandbox import workspace_archive

        params = workspace_archive._archive_request_params(
            '/workspace/project', 'git-delta'
        )
        assert params == {'path': '/workspace/project', 'format': 'git-delta'}
