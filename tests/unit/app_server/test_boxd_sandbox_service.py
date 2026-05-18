"""Tests for BoxdSandboxService.

Focuses on:
- boxd SDK exception handling
- Sandbox lifecycle management (start, pause, resume, delete)
- Status mapping from boxd VM status to internal sandbox statuses
- Environment variable injection for CORS and webhooks
- Data transformation from boxd Box + stored row into SandboxInfo
- User-scoped sandbox operations and security
- Pagination and search functionality via DB-backed index
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from openhands.app_server.errors import SandboxError
from openhands.app_server.sandbox.boxd_sandbox_service import (
    AGENT_PROXY_NAME,
    AGENT_SERVER_PORT,
    STATUS_MAPPING,
    BoxdSandboxService,
    StoredBoxdSandbox,
    _hash_session_api_key,
)
from openhands.app_server.sandbox.sandbox_models import SandboxStatus
from openhands.app_server.sandbox.sandbox_service import (
    ALLOW_CORS_ORIGINS_VARIABLE,
    SESSION_API_KEY_VARIABLE,
    WEBHOOK_CALLBACK_VARIABLE,
)
from openhands.app_server.sandbox.sandbox_spec_models import SandboxSpecInfo
from openhands.app_server.user.user_context import UserContext


# ─── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def mock_sandbox_spec_service():
    """SandboxSpecService that returns a deterministic spec."""
    mock_service = AsyncMock()
    mock_spec = SandboxSpecInfo(
        id='ghcr.io/openhands/agent-server:1.22.1-python',
        command=['/usr/local/bin/openhands-agent-server', '--port', '60000'],
        initial_env={'LOG_JSON': 'true'},
        working_dir='/workspace/project',
    )
    mock_service.get_default_sandbox_spec.return_value = mock_spec
    mock_service.get_sandbox_spec.return_value = mock_spec
    return mock_service


@pytest.fixture
def mock_user_context():
    mock_context = AsyncMock(spec=UserContext)
    mock_context.get_user_id.return_value = 'test-user-123'
    return mock_context


@pytest.fixture
def mock_compute():
    """Mock boxd Compute client. Tests configure box.create/list/get per case."""
    # Use MagicMock with async methods explicitly stubbed per test, since the
    # boxd async SDK exposes coroutine methods on box.create / box.get / etc.
    mock = MagicMock()
    mock.box.create = AsyncMock()
    mock.box.get = AsyncMock()
    mock.box.list = AsyncMock()
    return mock


@pytest.fixture
def mock_db_session():
    return AsyncMock(spec=AsyncSession)


@pytest.fixture
def boxd_sandbox_service(
    mock_sandbox_spec_service, mock_user_context, mock_compute, mock_db_session
):
    return BoxdSandboxService(
        sandbox_spec_service=mock_sandbox_spec_service,
        compute=mock_compute,
        web_url='https://web.example.com',
        max_num_sandboxes=10,
        auto_suspend_timeout=300,
        vcpu=2,
        memory='8G',
        disk='100G',
        user_context=mock_user_context,
        db_session=mock_db_session,
    )


def make_box_mock(
    box_id: str = 'box-abc',
    name: str = 'oh-test-sandbox',
    status: str = 'running',
    proxy_domain: str = 'agent-oh-test-sandbox.boxd.sh',
):
    """Build a MagicMock Box. Production Box objects don't carry env back."""
    box = MagicMock()
    box.id = box_id
    box.name = name
    box.status = status
    proxy = MagicMock()
    proxy.name = AGENT_PROXY_NAME
    proxy.domain = proxy_domain
    proxy.port = AGENT_SERVER_PORT
    proxy.is_default = False
    # Async SDK: lifecycle and proxy methods are coroutines
    box.proxies = AsyncMock(return_value=[proxy])
    box.create_proxy = AsyncMock(return_value=proxy)
    box.suspend = AsyncMock()
    box.resume = AsyncMock()
    box.destroy = AsyncMock()
    return box


def make_stored(
    sandbox_id: str = 'test-sandbox-123',
    user_id: str = 'test-user-123',
    spec_id: str = 'ghcr.io/openhands/agent-server:1.22.1-python',
    session_api_key_hash: str | None = None,
    created_at: datetime | None = None,
) -> StoredBoxdSandbox:
    return StoredBoxdSandbox(
        id=sandbox_id,
        created_by_user_id=user_id,
        sandbox_spec_id=spec_id,
        session_api_key_hash=session_api_key_hash,
        created_at=created_at or datetime.now(timezone.utc),
    )


def stub_search_returns(boxd_sandbox_service, stored_rows):
    """Make _secure_select() / db_session.execute return the given rows."""
    scalar_result = MagicMock()
    scalar_result.all.return_value = stored_rows
    exec_result = MagicMock()
    exec_result.scalars.return_value = scalar_result
    boxd_sandbox_service.db_session.execute = AsyncMock(return_value=exec_result)


def stub_get_stored_returns(boxd_sandbox_service, stored):
    """Make a single-row select return the given stored row (or None)."""
    exec_result = MagicMock()
    exec_result.scalar_one_or_none.return_value = stored
    boxd_sandbox_service.db_session.execute = AsyncMock(return_value=exec_result)


# ─── Tests ────────────────────────────────────────────────────────────────


class TestStatusMapping:
    def test_running_maps_to_running(self):
        assert STATUS_MAPPING['running'] == SandboxStatus.RUNNING

    def test_suspended_maps_to_paused(self):
        assert STATUS_MAPPING['suspended'] == SandboxStatus.PAUSED

    def test_starting_maps_to_starting(self):
        assert STATUS_MAPPING['starting'] == SandboxStatus.STARTING

    def test_error_maps_to_error(self):
        assert STATUS_MAPPING['error'] == SandboxStatus.ERROR

    def test_stopped_maps_to_missing(self):
        assert STATUS_MAPPING['stopped'] == SandboxStatus.MISSING


class TestStartSandbox:
    @pytest.mark.asyncio
    async def test_creates_box_with_default_spec(
        self, boxd_sandbox_service, mock_compute
    ):
        # pause_old_sandboxes will call search_sandboxes — stub it empty.
        stub_search_returns(boxd_sandbox_service, [])
        mock_compute.box.create.return_value = make_box_mock()
        info = await boxd_sandbox_service.start_sandbox()
        assert info.status == SandboxStatus.RUNNING
        assert info.created_by_user_id == 'test-user-123'
        mock_compute.box.create.assert_called_once()
        call_kwargs = mock_compute.box.create.call_args.kwargs
        assert 'config' in call_kwargs

    @pytest.mark.asyncio
    async def test_passes_image_from_spec(
        self, boxd_sandbox_service, mock_compute
    ):
        stub_search_returns(boxd_sandbox_service, [])
        mock_compute.box.create.return_value = make_box_mock()
        await boxd_sandbox_service.start_sandbox()
        call_kwargs = mock_compute.box.create.call_args.kwargs
        assert call_kwargs['image'] == 'ghcr.io/openhands/agent-server:1.22.1-python'

    @pytest.mark.asyncio
    async def test_sets_env_from_spec_and_webhook(
        self, boxd_sandbox_service, mock_compute
    ):
        stub_search_returns(boxd_sandbox_service, [])
        mock_compute.box.create.return_value = make_box_mock()
        await boxd_sandbox_service.start_sandbox()
        config = mock_compute.box.create.call_args.kwargs['config']
        assert config.env['LOG_JSON'] == 'true'
        assert WEBHOOK_CALLBACK_VARIABLE in config.env
        assert ALLOW_CORS_ORIGINS_VARIABLE in config.env

    @pytest.mark.asyncio
    async def test_inserts_stored_row(
        self, boxd_sandbox_service, mock_compute, mock_db_session
    ):
        stub_search_returns(boxd_sandbox_service, [])
        mock_compute.box.create.return_value = make_box_mock()
        await boxd_sandbox_service.start_sandbox()
        mock_db_session.add.assert_called_once()
        added = mock_db_session.add.call_args.args[0]
        assert isinstance(added, StoredBoxdSandbox)
        assert added.created_by_user_id == 'test-user-123'
        assert added.sandbox_spec_id == 'ghcr.io/openhands/agent-server:1.22.1-python'
        assert added.session_api_key_hash is not None

    @pytest.mark.asyncio
    async def test_session_api_key_in_env_matches_hash_in_row(
        self, boxd_sandbox_service, mock_compute, mock_db_session
    ):
        stub_search_returns(boxd_sandbox_service, [])
        mock_compute.box.create.return_value = make_box_mock()
        await boxd_sandbox_service.start_sandbox()
        config = mock_compute.box.create.call_args.kwargs['config']
        session_key = config.env[SESSION_API_KEY_VARIABLE]
        added = mock_db_session.add.call_args.args[0]
        assert added.session_api_key_hash == _hash_session_api_key(session_key)

    @pytest.mark.asyncio
    async def test_raises_sandbox_error_on_sdk_failure(
        self, boxd_sandbox_service, mock_compute
    ):
        from boxd.errors import BoxdError
        stub_search_returns(boxd_sandbox_service, [])
        mock_compute.box.create.side_effect = BoxdError('boxd is down')
        with pytest.raises(SandboxError):
            await boxd_sandbox_service.start_sandbox()

    @pytest.mark.asyncio
    async def test_creates_named_proxies_after_vm_create(
        self, boxd_sandbox_service, mock_compute
    ):
        """Named proxies aren't auto-created by BoxConfig — start_sandbox must
        explicitly call box.create_proxy for agent + vscode."""
        stub_search_returns(boxd_sandbox_service, [])
        box = make_box_mock()
        mock_compute.box.create.return_value = box
        await boxd_sandbox_service.start_sandbox()
        assert box.create_proxy.await_count == 2
        proxy_names = {call.args[0] for call in box.create_proxy.await_args_list}
        assert proxy_names == {'agent', 'vscode'}


class TestGetSandbox:
    @pytest.mark.asyncio
    async def test_returns_none_when_no_stored_row(self, boxd_sandbox_service):
        stub_get_stored_returns(boxd_sandbox_service, None)
        result = await boxd_sandbox_service.get_sandbox('missing')
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_info_when_row_and_vm_exist(
        self, boxd_sandbox_service, mock_compute
    ):
        stored = make_stored(sandbox_id='abc')
        stub_get_stored_returns(boxd_sandbox_service, stored)
        mock_compute.box.get.return_value = make_box_mock(name='oh-abc')
        info = await boxd_sandbox_service.get_sandbox('abc')
        assert info is not None
        assert info.id == 'abc'
        assert info.status == SandboxStatus.RUNNING
        assert info.created_by_user_id == 'test-user-123'

    @pytest.mark.asyncio
    async def test_status_missing_when_vm_gone_but_row_present(
        self, boxd_sandbox_service, mock_compute
    ):
        from boxd.errors import NotFoundError
        stored = make_stored(sandbox_id='abc')
        stub_get_stored_returns(boxd_sandbox_service, stored)
        mock_compute.box.get.side_effect = NotFoundError('gone')
        info = await boxd_sandbox_service.get_sandbox('abc')
        assert info is not None
        assert info.status == SandboxStatus.MISSING


class TestSearchSandboxes:
    @pytest.mark.asyncio
    async def test_returns_page_with_running_status(
        self, boxd_sandbox_service, mock_compute
    ):
        stored = make_stored(sandbox_id='abc')
        stub_search_returns(boxd_sandbox_service, [stored])
        mock_compute.box.get.return_value = make_box_mock(name='oh-abc')
        page = await boxd_sandbox_service.search_sandboxes()
        assert len(page.items) == 1
        assert page.items[0].id == 'abc'

    @pytest.mark.asyncio
    async def test_pagination_sets_next_page_id(
        self, boxd_sandbox_service, mock_compute
    ):
        # limit=3 but we return 4 rows — the +1 sentinel signals "more".
        rows = [make_stored(sandbox_id=f'sb-{i}') for i in range(4)]
        stub_search_returns(boxd_sandbox_service, rows)
        mock_compute.box.get.return_value = make_box_mock()
        page = await boxd_sandbox_service.search_sandboxes(limit=3)
        assert len(page.items) == 3
        assert page.next_page_id == '3'


class TestGetSandboxBySessionApiKey:
    @pytest.mark.asyncio
    async def test_finds_by_session_key(
        self, boxd_sandbox_service, mock_compute
    ):
        session_key = 'session-secret'
        stored = make_stored(
            sandbox_id='abc',
            session_api_key_hash=_hash_session_api_key(session_key),
        )
        stub_get_stored_returns(boxd_sandbox_service, stored)
        mock_compute.box.get.return_value = make_box_mock(name='oh-abc')
        info = await boxd_sandbox_service.get_sandbox_by_session_api_key(session_key)
        assert info is not None
        assert info.session_api_key == session_key

    @pytest.mark.asyncio
    async def test_returns_none_when_no_match(self, boxd_sandbox_service):
        stub_get_stored_returns(boxd_sandbox_service, None)
        assert (
            await boxd_sandbox_service.get_sandbox_by_session_api_key('nope') is None
        )


class TestLifecycleOperations:
    @pytest.mark.asyncio
    async def test_pause_calls_suspend(self, boxd_sandbox_service, mock_compute):
        stored = make_stored(sandbox_id='abc', session_api_key_hash='HASH')
        stub_get_stored_returns(boxd_sandbox_service, stored)
        box = make_box_mock(name='oh-abc')
        mock_compute.box.get.return_value = box
        ok = await boxd_sandbox_service.pause_sandbox('abc')
        assert ok is True
        box.suspend.assert_called_once()
        # Security: hash is cleared on pause
        assert stored.session_api_key_hash is None

    @pytest.mark.asyncio
    async def test_pause_returns_false_when_stored_row_missing(
        self, boxd_sandbox_service
    ):
        stub_get_stored_returns(boxd_sandbox_service, None)
        assert await boxd_sandbox_service.pause_sandbox('abc') is False

    @pytest.mark.asyncio
    async def test_resume_calls_resume(self, boxd_sandbox_service, mock_compute):
        # resume_sandbox calls pause_old_sandboxes first → needs search stub
        # and _get_stored stub. The mock_db_session is shared, so we install
        # a side_effect that returns different shapes per call.
        stored = make_stored(sandbox_id='abc')
        search_scalar = MagicMock()
        search_scalar.all.return_value = []
        search_result = MagicMock()
        search_result.scalars.return_value = search_scalar
        get_result = MagicMock()
        get_result.scalar_one_or_none.return_value = stored
        boxd_sandbox_service.db_session.execute = AsyncMock(
            side_effect=[search_result, get_result]
        )
        box = make_box_mock(name='oh-abc')
        mock_compute.box.get.return_value = box
        ok = await boxd_sandbox_service.resume_sandbox('abc')
        assert ok is True
        box.resume.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_calls_destroy_and_drops_row(
        self, boxd_sandbox_service, mock_compute, mock_db_session
    ):
        stored = make_stored(sandbox_id='abc')
        stub_get_stored_returns(boxd_sandbox_service, stored)
        box = make_box_mock(name='oh-abc')
        mock_compute.box.get.return_value = box
        ok = await boxd_sandbox_service.delete_sandbox('abc')
        assert ok is True
        box.destroy.assert_called_once()
        mock_db_session.delete.assert_called_once_with(stored)

    @pytest.mark.asyncio
    async def test_delete_returns_true_when_vm_already_gone(
        self, boxd_sandbox_service, mock_compute, mock_db_session
    ):
        from boxd.errors import NotFoundError
        stored = make_stored(sandbox_id='abc')
        stub_get_stored_returns(boxd_sandbox_service, stored)
        mock_compute.box.get.side_effect = NotFoundError('gone')
        ok = await boxd_sandbox_service.delete_sandbox('abc')
        # We still cleaned the index — return True so callers don't retry.
        assert ok is True
        mock_db_session.delete.assert_called_once_with(stored)


class TestInjector:
    @pytest.mark.asyncio
    async def test_injector_carries_config(self):
        from openhands.app_server.sandbox.boxd_sandbox_service import (
            BoxdSandboxServiceInjector,
        )

        injector = BoxdSandboxServiceInjector(
            api_key='bxk_test',
            max_num_sandboxes=5,
            auto_suspend_timeout=600,
            vcpu=2,
            memory='8G',
            disk='100G',
        )

        assert injector.api_key == 'bxk_test'
        assert injector.max_num_sandboxes == 5
        assert injector.auto_suspend_timeout == 600
        assert injector.vcpu == 2
        assert injector.memory == '8G'
        assert injector.disk == '100G'
