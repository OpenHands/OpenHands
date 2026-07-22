"""Tests for SandboxService base class.

This module tests the SandboxService base class implementation, focusing on:
- pause_old_sandboxes method functionality
- Proper handling of pagination when searching sandboxes
- Correct filtering of running vs non-running sandboxes
- Proper sorting by creation time (oldest first)
- Error handling and edge cases
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from openhands.app_server.errors import SandboxError
from openhands.app_server.sandbox.sandbox_models import (
    SandboxInfo,
    SandboxPage,
    SandboxRecord,
    SandboxStatus,
)
from openhands.app_server.sandbox.sandbox_service import (
    SandboxService,
    _classify_start_failure,
    _start_failure_error,
)


class MockSandboxService(SandboxService):
    """Mock implementation of SandboxService for testing."""

    def __init__(self):
        self.search_sandboxes_mock = AsyncMock()
        self.get_sandbox_mock = AsyncMock()
        self.get_sandbox_by_session_api_key_mock = AsyncMock()
        self.get_sandbox_record_by_session_api_key_mock = AsyncMock()
        self.start_sandbox_mock = AsyncMock()
        self.resume_sandbox_mock = AsyncMock()
        self.pause_sandbox_mock = AsyncMock()
        self.delete_sandbox_mock = AsyncMock()

    async def search_sandboxes(
        self, page_id: str | None = None, limit: int = 100
    ) -> SandboxPage:
        return await self.search_sandboxes_mock(page_id=page_id, limit=limit)

    async def get_sandbox(self, sandbox_id: str) -> SandboxInfo | None:
        return await self.get_sandbox_mock(sandbox_id)

    async def get_sandbox_by_session_api_key(
        self, session_api_key: str
    ) -> SandboxInfo | None:
        return await self.get_sandbox_by_session_api_key_mock(session_api_key)

    async def get_sandbox_record_by_session_api_key(
        self, session_api_key: str
    ) -> SandboxRecord | None:
        return await self.get_sandbox_record_by_session_api_key_mock(session_api_key)

    async def start_sandbox(
        self, sandbox_spec_id: str | None = None, sandbox_id: str | None = None
    ) -> SandboxInfo:
        return await self.start_sandbox_mock(sandbox_spec_id, sandbox_id)

    async def resume_sandbox(self, sandbox_id: str) -> bool:
        return await self.resume_sandbox_mock(sandbox_id)

    async def pause_sandbox(self, sandbox_id: str) -> bool:
        return await self.pause_sandbox_mock(sandbox_id)

    async def delete_sandbox(self, sandbox_id: str) -> bool:
        return await self.delete_sandbox_mock(sandbox_id)


def create_sandbox_info(
    sandbox_id: str,
    status: SandboxStatus,
    created_at: datetime,
    created_by_user_id: str | None = None,
    sandbox_spec_id: str = 'test-spec',
) -> SandboxInfo:
    """Helper function to create SandboxInfo objects for testing."""
    return SandboxInfo(
        id=sandbox_id,
        created_by_user_id=created_by_user_id,
        sandbox_spec_id=sandbox_spec_id,
        status=status,
        session_api_key='test-api-key' if status == SandboxStatus.RUNNING else None,
        created_at=created_at,
    )


@pytest.fixture
def mock_sandbox_service():
    """Fixture providing a mock sandbox service."""
    return MockSandboxService()


class TestWaitForSandboxRunning:
    """wait_for_sandbox_running maps runtime status_detail to user-safe errors.

    Raw status_detail is logged, not surfaced: only classified/generic text
    reaches the user, so internal infra details (registry hosts, secret names,
    node taints/labels, cluster size, ...) can't leak.
    """

    def _sandbox(self, status, detail):
        return SandboxInfo(
            id='sb1',
            created_by_user_id=None,
            sandbox_spec_id='test-spec',
            status=status,
            session_api_key=None,
            created_at=datetime.now(timezone.utc),
            status_detail=detail,
        )

    def _patch_timeout(self, monkeypatch):
        calls = {'n': 0}

        def fake_time():
            calls['n'] += 1
            return 1000.0 if calls['n'] <= 3 else 2000.0

        monkeypatch.setattr(
            'openhands.app_server.sandbox.sandbox_service.time.time', fake_time
        )
        monkeypatch.setattr(
            'openhands.app_server.sandbox.sandbox_service.asyncio.sleep', AsyncMock()
        )

    @pytest.mark.asyncio
    async def test_capacity_error_state_is_friendly_and_trimmed(
        self, mock_sandbox_service
    ):
        mock_sandbox_service.get_sandbox_mock.return_value = self._sandbox(
            SandboxStatus.ERROR,
            '0/1 nodes are available: 1 Insufficient ephemeral-storage. '
            'preemption: 0/1 nodes are available.',
        )
        with pytest.raises(SandboxError) as ei:
            await mock_sandbox_service.wait_for_sandbox_running('sb1', timeout=5)
        msg = str(ei.value)
        assert 'at capacity' in msg
        assert 'insufficient ephemeral-storage' in msg
        # reference id lets an owner find the raw reason in the logs
        assert 'reference: sb1' in msg
        # node counts / preemption noise are not leaked
        assert '0/1 nodes' not in msg
        assert 'preemption' not in msg
        assert 'entered error state' not in msg

    @pytest.mark.asyncio
    async def test_capacity_at_timeout_is_retryable(
        self, mock_sandbox_service, monkeypatch
    ):
        mock_sandbox_service.get_sandbox_mock.return_value = self._sandbox(
            SandboxStatus.STARTING, '0/1 nodes are available: 1 Insufficient cpu.'
        )
        self._patch_timeout(monkeypatch)
        with pytest.raises(SandboxError) as ei:
            await mock_sandbox_service.wait_for_sandbox_running('sb1', timeout=1)
        msg = str(ei.value)
        assert 'at capacity' in msg and 'try again' in msg
        assert 'insufficient cpu' in msg
        assert 'failed to start within' not in msg
        assert mock_sandbox_service.get_sandbox_mock.call_count > 1

    @pytest.mark.asyncio
    async def test_image_pull_detail_is_redacted(self, mock_sandbox_service):
        mock_sandbox_service.get_sandbox_mock.return_value = self._sandbox(
            SandboxStatus.ERROR,
            'ImagePullBackOff: Back-off pulling image '
            '"registry.internal.corp/team/secret-project:tag"',
        )
        with pytest.raises(SandboxError) as ei:
            await mock_sandbox_service.wait_for_sandbox_running('sb1', timeout=5)
        msg = str(ei.value)
        assert 'could not be pulled' in msg
        assert 'registry.internal.corp' not in msg
        assert 'secret-project' not in msg

    @pytest.mark.asyncio
    async def test_config_error_detail_is_redacted(self, mock_sandbox_service):
        mock_sandbox_service.get_sandbox_mock.return_value = self._sandbox(
            SandboxStatus.ERROR,
            "CreateContainerConfigError: couldn't find key db-password "
            'in secret "prod-db-creds"',
        )
        with pytest.raises(SandboxError) as ei:
            await mock_sandbox_service.wait_for_sandbox_running('sb1', timeout=5)
        msg = str(ei.value)
        assert 'configuration issue' in msg
        assert 'prod-db-creds' not in msg
        assert 'db-password' not in msg

    @pytest.mark.asyncio
    async def test_container_create_error_is_distinct_and_redacted(
        self, mock_sandbox_service
    ):
        # CreateContainerError is broader than a missing secret: don't mislabel
        # it as a config issue, and don't leak the path.
        mock_sandbox_service.get_sandbox_mock.return_value = self._sandbox(
            SandboxStatus.ERROR,
            'CreateContainerError: failed to mount /var/secrets/prod-token',
        )
        with pytest.raises(SandboxError) as ei:
            await mock_sandbox_service.wait_for_sandbox_running('sb1', timeout=5)
        msg = str(ei.value)
        assert 'could not be created' in msg
        assert 'configuration issue' not in msg
        assert 'prod-token' not in msg

    @pytest.mark.asyncio
    async def test_scheduling_constraint_is_redacted(self, mock_sandbox_service):
        mock_sandbox_service.get_sandbox_mock.return_value = self._sandbox(
            SandboxStatus.ERROR,
            'node(s) had untolerated taint {dedicated: internal-gpu-pool}',
        )
        with pytest.raises(SandboxError) as ei:
            await mock_sandbox_service.wait_for_sandbox_running('sb1', timeout=5)
        msg = str(ei.value)
        assert 'could not be scheduled' in msg
        assert 'internal-gpu-pool' not in msg
        assert 'dedicated' not in msg
        # deterministic placement failure -> no misleading 'try again'
        assert 'try again' not in msg

    @pytest.mark.asyncio
    async def test_device_resource_shortage_is_redacted(self, mock_sandbox_service):
        mock_sandbox_service.get_sandbox_mock.return_value = self._sandbox(
            SandboxStatus.ERROR,
            '0/3 nodes are available: 3 Insufficient smarter-devices/kvm.',
        )
        with pytest.raises(SandboxError) as ei:
            await mock_sandbox_service.wait_for_sandbox_running('sb1', timeout=5)
        msg = str(ei.value)
        assert 'device or resource' in msg
        # the device-plugin name must not leak
        assert 'smarter-devices' not in msg
        assert 'kvm' not in msg

    @pytest.mark.asyncio
    async def test_unrecognized_detail_is_generic(self, mock_sandbox_service):
        mock_sandbox_service.get_sandbox_mock.return_value = self._sandbox(
            SandboxStatus.ERROR, 'the sandbox imploded for reasons unknown'
        )
        with pytest.raises(SandboxError) as ei:
            await mock_sandbox_service.wait_for_sandbox_running('sb1', timeout=5)
        msg = str(ei.value)
        assert 'unexpected reason' in msg
        assert 'imploded' not in msg

    @pytest.mark.asyncio
    async def test_no_detail_is_generic(self, mock_sandbox_service):
        mock_sandbox_service.get_sandbox_mock.return_value = self._sandbox(
            SandboxStatus.ERROR, None
        )
        with pytest.raises(SandboxError) as ei:
            await mock_sandbox_service.wait_for_sandbox_running('sb1', timeout=5)
        assert 'failed to start' in str(ei.value)


def test_classify_start_failure_maps_each_class():
    assert 'at capacity' in _classify_start_failure('x Insufficient memory y')
    assert 'insufficient memory' in _classify_start_failure('x Insufficient memory y')
    assert 'could not be pulled' in _classify_start_failure('ImagePullBackOff: x')
    assert 'configuration issue' in _classify_start_failure(
        'CreateContainerConfigError: x'
    )
    assert 'could not be created' in _classify_start_failure('CreateContainerError: x')
    assert 'could not be scheduled' in _classify_start_failure(
        "0/2 nodes: 2 node(s) didn't match Pod's node affinity/selector."
    )
    assert 'device or resource' in _classify_start_failure(
        '0/1 nodes: 1 Insufficient nvidia.com/gpu.'
    )
    # unrecognized / empty -> None so the caller uses a generic message
    assert _classify_start_failure('some novel reason') is None
    assert _classify_start_failure(None) is None
    assert _classify_start_failure('') is None


def test_start_failure_error_appends_reference():
    err = _start_failure_error('sb-xyz', 'ImagePullBackOff: secret/path')
    assert 'could not be pulled' in str(err)
    assert '(reference: sb-xyz)' in str(err)
    # unknown detail still gets the generic message + the reference id
    generic = _start_failure_error('sb-xyz', 'mystery')
    assert 'unexpected reason' in str(generic)
    assert '(reference: sb-xyz)' in str(generic)


class TestCleanupOldSandboxes:
    """Test cases for the pause_old_sandboxes method."""

    @pytest.mark.asyncio
    async def test_cleanup_with_no_sandboxes(self, mock_sandbox_service):
        """Test cleanup when there are no sandboxes."""
        # Setup: No sandboxes
        mock_sandbox_service.search_sandboxes_mock.return_value = SandboxPage(
            items=[], next_page_id=None
        )

        # Execute
        result = await mock_sandbox_service.pause_old_sandboxes(max_num_sandboxes=5)

        # Verify
        assert result == []
        mock_sandbox_service.search_sandboxes_mock.assert_called_once_with(
            page_id=None, limit=100
        )
        mock_sandbox_service.pause_sandbox_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup_within_limit(self, mock_sandbox_service):
        """Test cleanup when sandbox count is within the limit."""
        # Setup: 3 running sandboxes, limit is 5
        now = datetime.now(timezone.utc)
        sandboxes = [
            create_sandbox_info('sb1', SandboxStatus.RUNNING, now - timedelta(hours=3)),
            create_sandbox_info('sb2', SandboxStatus.RUNNING, now - timedelta(hours=2)),
            create_sandbox_info('sb3', SandboxStatus.RUNNING, now - timedelta(hours=1)),
        ]

        mock_sandbox_service.search_sandboxes_mock.return_value = SandboxPage(
            items=sandboxes, next_page_id=None
        )

        # Execute
        result = await mock_sandbox_service.pause_old_sandboxes(max_num_sandboxes=5)

        # Verify
        assert result == []
        mock_sandbox_service.pause_sandbox_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup_exceeds_limit(self, mock_sandbox_service):
        """Test cleanup when sandbox count exceeds the limit."""
        # Setup: 5 running sandboxes, limit is 3
        now = datetime.now(timezone.utc)
        sandboxes = [
            create_sandbox_info(
                'sb1', SandboxStatus.RUNNING, now - timedelta(hours=5)
            ),  # oldest
            create_sandbox_info(
                'sb2', SandboxStatus.RUNNING, now - timedelta(hours=4)
            ),  # second oldest
            create_sandbox_info(
                'sb3', SandboxStatus.RUNNING, now - timedelta(hours=3)
            ),  # should be stopped
            create_sandbox_info(
                'sb4', SandboxStatus.RUNNING, now - timedelta(hours=2)
            ),  # should remain
            create_sandbox_info(
                'sb5', SandboxStatus.RUNNING, now - timedelta(hours=1)
            ),  # newest
        ]

        mock_sandbox_service.search_sandboxes_mock.return_value = SandboxPage(
            items=sandboxes, next_page_id=None
        )
        mock_sandbox_service.pause_sandbox_mock.return_value = True

        # Execute
        result = await mock_sandbox_service.pause_old_sandboxes(max_num_sandboxes=2)

        # Verify: Should pause the 2 oldest sandboxes
        assert result == ['sb1', 'sb2', 'sb3']
        assert mock_sandbox_service.pause_sandbox_mock.call_count == 3

    @pytest.mark.asyncio
    async def test_cleanup_filters_non_running_sandboxes(self, mock_sandbox_service):
        """Test that cleanup only considers running sandboxes."""
        # Setup: Mix of running and non-running sandboxes
        now = datetime.now(timezone.utc)
        sandboxes = [
            create_sandbox_info('sb1', SandboxStatus.RUNNING, now - timedelta(hours=5)),
            create_sandbox_info(
                'sb2', SandboxStatus.PAUSED, now - timedelta(hours=4)
            ),  # should be ignored
            create_sandbox_info('sb3', SandboxStatus.RUNNING, now - timedelta(hours=3)),
            create_sandbox_info(
                'sb4', SandboxStatus.ERROR, now - timedelta(hours=2)
            ),  # should be ignored
            create_sandbox_info('sb5', SandboxStatus.RUNNING, now - timedelta(hours=1)),
        ]

        mock_sandbox_service.search_sandboxes_mock.return_value = SandboxPage(
            items=sandboxes, next_page_id=None
        )
        mock_sandbox_service.pause_sandbox_mock.return_value = True

        # Execute: Limit is 2, but only 3 are running
        result = await mock_sandbox_service.pause_old_sandboxes(max_num_sandboxes=2)

        # Verify: Should stop only 1 sandbox (the oldest running one)
        assert len(result) == 1
        assert 'sb1' in result
        mock_sandbox_service.pause_sandbox_mock.assert_called_once_with('sb1')

    @pytest.mark.asyncio
    async def test_cleanup_with_pagination(self, mock_sandbox_service):
        """Test cleanup handles pagination correctly."""
        # Setup: Multiple pages of sandboxes
        now = datetime.now(timezone.utc)

        # First page
        page1_sandboxes = [
            create_sandbox_info('sb1', SandboxStatus.RUNNING, now - timedelta(hours=3)),
            create_sandbox_info('sb2', SandboxStatus.RUNNING, now - timedelta(hours=2)),
        ]

        # Second page
        page2_sandboxes = [
            create_sandbox_info('sb3', SandboxStatus.RUNNING, now - timedelta(hours=1)),
        ]

        def search_side_effect(page_id=None, limit=100):
            if page_id is None:
                return SandboxPage(items=page1_sandboxes, next_page_id='page2')
            elif page_id == 'page2':
                return SandboxPage(items=page2_sandboxes, next_page_id=None)

        mock_sandbox_service.search_sandboxes_mock.side_effect = search_side_effect
        mock_sandbox_service.pause_sandbox_mock.return_value = True

        # Execute: Limit is 2, total is 3
        result = await mock_sandbox_service.pause_old_sandboxes(max_num_sandboxes=2)

        # Verify: Should stop the oldest sandbox
        assert len(result) == 1
        assert 'sb1' in result
        assert mock_sandbox_service.search_sandboxes_mock.call_count == 2

    @pytest.mark.asyncio
    async def test_cleanup_handles_pause_failures(self, mock_sandbox_service):
        """Test cleanup continues when some pause operations fail."""
        # Setup: 4 running sandboxes, limit is 2
        now = datetime.now(timezone.utc)
        sandboxes = [
            create_sandbox_info('sb1', SandboxStatus.RUNNING, now - timedelta(hours=4)),
            create_sandbox_info('sb2', SandboxStatus.RUNNING, now - timedelta(hours=3)),
            create_sandbox_info('sb3', SandboxStatus.RUNNING, now - timedelta(hours=2)),
            create_sandbox_info('sb4', SandboxStatus.RUNNING, now - timedelta(hours=1)),
        ]

        mock_sandbox_service.search_sandboxes_mock.return_value = SandboxPage(
            items=sandboxes, next_page_id=None
        )

        # Setup: First pause fails, second succeeds
        def pause_side_effect(sandbox_id):
            if sandbox_id == 'sb1':
                return False  # Simulate failure
            return True

        mock_sandbox_service.pause_sandbox_mock.side_effect = pause_side_effect

        # Execute
        result = await mock_sandbox_service.pause_old_sandboxes(max_num_sandboxes=2)

        # Verify: Should only include successfully paused sandbox
        assert len(result) == 1
        assert 'sb2' in result
        assert mock_sandbox_service.pause_sandbox_mock.call_count == 2

    @pytest.mark.asyncio
    async def test_cleanup_handles_pause_exceptions(self, mock_sandbox_service):
        """Test cleanup continues when pause operations raise exceptions."""
        # Setup: 3 running sandboxes, limit is 1
        now = datetime.now(timezone.utc)
        sandboxes = [
            create_sandbox_info('sb1', SandboxStatus.RUNNING, now - timedelta(hours=3)),
            create_sandbox_info('sb2', SandboxStatus.RUNNING, now - timedelta(hours=2)),
            create_sandbox_info('sb3', SandboxStatus.RUNNING, now - timedelta(hours=1)),
        ]

        mock_sandbox_service.search_sandboxes_mock.return_value = SandboxPage(
            items=sandboxes, next_page_id=None
        )

        # Setup: First pause raises exception, second succeeds
        def pause_side_effect(sandbox_id):
            if sandbox_id == 'sb1':
                raise Exception('Delete failed')
            return True

        mock_sandbox_service.pause_sandbox_mock.side_effect = pause_side_effect

        # Execute
        result = await mock_sandbox_service.pause_old_sandboxes(max_num_sandboxes=1)

        # Verify: Should only include successfully paused sandbox
        assert len(result) == 1
        assert 'sb2' in result
        assert mock_sandbox_service.pause_sandbox_mock.call_count == 2

    @pytest.mark.asyncio
    async def test_cleanup_invalid_max_num_sandboxes(self, mock_sandbox_service):
        """Test cleanup raises ValueError for invalid max_num_sandboxes."""
        # Test zero
        with pytest.raises(
            ValueError, match='max_num_sandboxes must be greater than 0'
        ):
            await mock_sandbox_service.pause_old_sandboxes(max_num_sandboxes=0)

        # Test negative
        with pytest.raises(
            ValueError, match='max_num_sandboxes must be greater than 0'
        ):
            await mock_sandbox_service.pause_old_sandboxes(max_num_sandboxes=-1)

    @pytest.mark.asyncio
    async def test_cleanup_sorts_by_creation_time(self, mock_sandbox_service):
        """Test that cleanup properly sorts sandboxes by creation time."""
        # Setup: Sandboxes in random order by creation time
        now = datetime.now(timezone.utc)
        sandboxes = [
            create_sandbox_info('sb_newest', SandboxStatus.RUNNING, now),  # newest
            create_sandbox_info(
                'sb_oldest', SandboxStatus.RUNNING, now - timedelta(hours=5)
            ),  # oldest
            create_sandbox_info(
                'sb_middle', SandboxStatus.RUNNING, now - timedelta(hours=2)
            ),  # middle
        ]

        mock_sandbox_service.search_sandboxes_mock.return_value = SandboxPage(
            items=sandboxes, next_page_id=None
        )
        mock_sandbox_service.pause_sandbox_mock.return_value = True

        # Execute: Keep only 1 sandbox
        result = await mock_sandbox_service.pause_old_sandboxes(max_num_sandboxes=1)

        # Verify: Should stop the 2 oldest sandboxes in order
        assert len(result) == 2
        assert 'sb_oldest' in result
        assert 'sb_middle' in result

        # Verify pause was called in the correct order (oldest first)
        calls = mock_sandbox_service.pause_sandbox_mock.call_args_list
        assert calls[0][0][0] == 'sb_oldest'
        assert calls[1][0][0] == 'sb_middle'

    @pytest.mark.asyncio
    async def test_cleanup_exact_limit(self, mock_sandbox_service):
        """Test cleanup when sandbox count exactly equals the limit."""
        # Setup: Exactly 3 running sandboxes, limit is 3
        now = datetime.now(timezone.utc)
        sandboxes = [
            create_sandbox_info('sb1', SandboxStatus.RUNNING, now - timedelta(hours=3)),
            create_sandbox_info('sb2', SandboxStatus.RUNNING, now - timedelta(hours=2)),
            create_sandbox_info('sb3', SandboxStatus.RUNNING, now - timedelta(hours=1)),
        ]

        mock_sandbox_service.search_sandboxes_mock.return_value = SandboxPage(
            items=sandboxes, next_page_id=None
        )

        # Execute
        result = await mock_sandbox_service.pause_old_sandboxes(max_num_sandboxes=3)

        # Verify: No sandboxes should be stopped
        assert result == []
        mock_sandbox_service.pause_sandbox_mock.assert_not_called()
