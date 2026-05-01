"""Tests for OssAppLifespanService."""

import asyncio
from unittest.mock import patch

import pytest

import openhands.app_server.sandbox.remote_sandbox_service as rss_module
from openhands.app_server.app_lifespan.oss_app_lifespan_service import (
    OssAppLifespanService,
)


class TestOssAppLifespanShutdown:
    """Tests for OssAppLifespanService shutdown behaviour."""

    @pytest.mark.asyncio
    async def test_aexit_cancels_polling_task_when_running(self):
        """Test that __aexit__ cancels a running polling_task on shutdown.

        Fix A ensures that the global background polling task is cancelled when
        the application exits, preventing event-loop and DB connection leaks.
        """

        # Create a real asyncio Task that never completes on its own
        async def _never_ending():
            try:
                await asyncio.sleep(9999)
            except asyncio.CancelledError:
                return

        task = asyncio.create_task(_never_ending())

        service = OssAppLifespanService(run_alembic_on_startup=False)

        with patch.object(rss_module, 'polling_task', task):
            await service.__aexit__(None, None, None)

        # Task must have been cancelled
        assert task.cancelled() or task.done()

    @pytest.mark.asyncio
    async def test_aexit_skips_cancel_when_polling_task_none(self):
        """Test that __aexit__ does nothing when polling_task is None (no task started)."""
        service = OssAppLifespanService(run_alembic_on_startup=False)

        with patch.object(rss_module, 'polling_task', None):
            # Should complete without error
            await service.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_aexit_skips_cancel_when_polling_task_already_done(self):
        """Test that __aexit__ skips cancellation when the task is already finished."""

        async def _immediate():
            return

        task = asyncio.create_task(_immediate())
        await task  # let it finish

        service = OssAppLifespanService(run_alembic_on_startup=False)

        with patch.object(rss_module, 'polling_task', task):
            # Should complete without calling cancel on a done task
            await service.__aexit__(None, None, None)

        # Task is done and was NOT re-cancelled
        assert task.done()
        assert not task.cancelled()
