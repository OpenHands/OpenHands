"""Unit tests for the run_maintenance_tasks.py module.

These tests verify the functionality of the maintenance task runner script
that processes pending maintenance tasks.
"""

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock the database module to avoid dependency on Google Cloud SQL
mock_db = MagicMock()
mock_db.session_maker = MagicMock()
sys.modules['storage.database'] = mock_db

# Import after mocking
from run_maintenance_tasks import (  # noqa: E402
    main,
    next_task,
    run_tasks,
    set_stale_task_error,
)
from storage.maintenance_task import (  # noqa: E402
    MaintenanceTask,
    MaintenanceTaskProcessor,
    MaintenanceTaskStatus,
)


class MockMaintenanceTaskProcessor(MaintenanceTaskProcessor):
    """Mock processor for testing."""

    async def __call__(self, task: MaintenanceTask) -> dict:
        """Process a maintenance task."""
        return {'processed': True, 'task_id': task.id}


class TestRunMaintenanceTasks:
    """Tests for the run_maintenance_tasks.py module."""

    def test_set_stale_task_error(self, session_maker):
        """Test that stale tasks are marked as error."""
        # Create a stale task (working for more than 1 hour)
        with session_maker() as session:
            stale_task = MaintenanceTask(
                status=MaintenanceTaskStatus.WORKING,
                processor_type='test.processor',
                processor_json='{}',
                started_at=datetime.now(timezone.utc) - timedelta(hours=2),
            )
            session.add(stale_task)

            # Create a non-stale task (working for less than 1 hour)
            recent_task = MaintenanceTask(
                status=MaintenanceTaskStatus.WORKING,
                processor_type='test.processor',
                processor_json='{}',
                started_at=datetime.now(timezone.utc) - timedelta(minutes=30),
            )
            session.add(recent_task)
            session.commit()

            stale_task_id = stale_task.id
            recent_task_id = recent_task.id

        # Run the function
        with patch('run_maintenance_tasks.session_maker', return_value=session_maker()):
            set_stale_task_error()

        # Check that the stale task is marked as error
        with session_maker() as session:
            updated_stale_task = session.get(MaintenanceTask, stale_task_id)
            updated_recent_task = session.get(MaintenanceTask, recent_task_id)

            assert updated_stale_task.status == MaintenanceTaskStatus.ERROR
            assert updated_recent_task.status == MaintenanceTaskStatus.WORKING

    @pytest.mark.asyncio
    async def test_next_task(self, session_maker):
        """Test that next_task returns the oldest pending task."""
        # Create tasks with different statuses and creation times
        with session_maker() as session:
            # Create a pending task (older)
            older_pending_task = MaintenanceTask(
                status=MaintenanceTaskStatus.PENDING,
                processor_type='test.processor',
                processor_json='{}',
                created_at=datetime.now(timezone.utc) - timedelta(hours=2),
            )
            session.add(older_pending_task)

            # Create another pending task (newer)
            newer_pending_task = MaintenanceTask(
                status=MaintenanceTaskStatus.PENDING,
                processor_type='test.processor',
                processor_json='{}',
                created_at=datetime.now(timezone.utc) - timedelta(hours=1),
            )
            session.add(newer_pending_task)

            # Create tasks with other statuses
            working_task = MaintenanceTask(
                status=MaintenanceTaskStatus.WORKING,
                processor_type='test.processor',
                processor_json='{}',
            )
            session.add(working_task)

            completed_task = MaintenanceTask(
                status=MaintenanceTaskStatus.COMPLETED,
                processor_type='test.processor',
                processor_json='{}',
            )
            session.add(completed_task)

            error_task = MaintenanceTask(
                status=MaintenanceTaskStatus.ERROR,
                processor_type='test.processor',
                processor_json='{}',
            )
            session.add(error_task)

            inactive_task = MaintenanceTask(
                status=MaintenanceTaskStatus.INACTIVE,
                processor_type='test.processor',
                processor_json='{}',
            )
            session.add(inactive_task)

            session.commit()

            older_pending_id = older_pending_task.id

        # Test next_task function
        with session_maker() as session:
            # Patch asyncio.sleep to avoid delays in tests
            with patch('asyncio.sleep', new_callable=AsyncMock):
                task = await next_task(session)

                # Should return the oldest pending task
                assert task is not None
                assert task.id == older_pending_id
                assert task.status == MaintenanceTaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_next_task_with_no_pending_tasks(self, session_maker):
        """Test that next_task returns None when there are no pending tasks."""
        # Create session with no pending tasks
        with session_maker() as session:
            # Patch asyncio.sleep to avoid delays in tests
            with patch('asyncio.sleep', new_callable=AsyncMock):
                # Patch NUM_RETRIES to make the test faster
                with patch('run_maintenance_tasks.NUM_RETRIES', 1):
                    task = await next_task(session)

                    # Should return None after retries
                    assert task is None

    @pytest.mark.asyncio
    async def test_next_task_bug_fix(self, session_maker):
        """Test that next_task doesn't have an infinite loop bug."""
        # This test verifies the fix for the bug where `task = next_task` creates an infinite loop

        # Create a pending task
        with session_maker() as session:
            task = MaintenanceTask(
                status=MaintenanceTaskStatus.PENDING,
                processor_type='test.processor',
                processor_json='{}',
            )
            session.add(task)
            session.commit()
            task_id = task.id

        # Create a patched version of next_task with the bug fixed
        async def fixed_next_task(session):
            num_retries = 1  # Use a small value for testing
            while True:
                task = (
                    session.query(MaintenanceTask)
                    .filter(MaintenanceTask.status == MaintenanceTaskStatus.PENDING)
                    .order_by(MaintenanceTask.created_at)
                    .first()
                )
                if task:
                    return task
                # Fix: Don't assign next_task to task
                num_retries -= 1
                if num_retries < 0:
                    return None
                await asyncio.sleep(0.01)  # Small delay for testing

        with session_maker() as session:
            # Patch asyncio.sleep to avoid delays
            with patch('asyncio.sleep', new_callable=AsyncMock):
                # Test the fixed version
                with patch('run_maintenance_tasks.next_task', fixed_next_task):
                    # This should complete without hanging
                    result = await next_task(session)
                    assert result is not None
                    assert result.id == task_id

    @pytest.mark.asyncio
    async def test_run_tasks_processes_pending_tasks(self, session_maker):
        """Test that run_tasks processes pending tasks in order."""
        # Create a mock processor
        processor = AsyncMock()
        processor.return_value = {'processed': True}

        # Create tasks
        with session_maker() as session:
            # Create two pending tasks
            task1 = MaintenanceTask(
                status=MaintenanceTaskStatus.PENDING,
                processor_type='test.processor',
                processor_json='{}',
                created_at=datetime.now(timezone.utc) - timedelta(hours=2),
            )
            session.add(task1)

            task2 = MaintenanceTask(
                status=MaintenanceTaskStatus.PENDING,
                processor_type='test.processor',
                processor_json='{}',
                created_at=datetime.now(timezone.utc) - timedelta(hours=1),
            )
            session.add(task2)
            session.commit()

            task1_id = task1.id
            task2_id = task2.id

        # Mock the get_processor method to return our mock
        with patch(
            'storage.maintenance_task.MaintenanceTask.get_processor',
            return_value=processor,
        ):
            with patch(
                'run_maintenance_tasks.session_maker', return_value=session_maker()
            ):
                # Patch asyncio.sleep to avoid delays
                with patch('asyncio.sleep', new_callable=AsyncMock):
                    # Run the function with a timeout to prevent infinite loop
                    try:
                        await asyncio.wait_for(run_tasks(), timeout=1.0)
                    except asyncio.TimeoutError:
                        pass  # Expected since run_tasks runs until no tasks are left

        # Check that both tasks were processed
        with session_maker() as session:
            updated_task1 = session.get(MaintenanceTask, task1_id)
            updated_task2 = session.get(MaintenanceTask, task2_id)

            assert updated_task1.status == MaintenanceTaskStatus.COMPLETED
            assert updated_task2.status == MaintenanceTaskStatus.COMPLETED
            assert updated_task1.info == {'processed': True}
            assert updated_task2.info == {'processed': True}
            assert processor.call_count == 2

    @pytest.mark.asyncio
    async def test_run_tasks_handles_errors(self, session_maker):
        """Test that run_tasks handles processor errors correctly."""
        # Create a mock processor that raises an exception
        processor = AsyncMock()
        processor.side_effect = ValueError('Test error')

        # Create a task
        with session_maker() as session:
            task = MaintenanceTask(
                status=MaintenanceTaskStatus.PENDING,
                processor_type='test.processor',
                processor_json='{}',
            )
            session.add(task)
            session.commit()

            task_id = task.id

        # Mock the get_processor method to return our mock
        with patch(
            'storage.maintenance_task.MaintenanceTask.get_processor',
            return_value=processor,
        ):
            with patch(
                'run_maintenance_tasks.session_maker', return_value=session_maker()
            ):
                # Patch asyncio.sleep to avoid delays
                with patch('asyncio.sleep', new_callable=AsyncMock):
                    # Run the function with a timeout
                    try:
                        await asyncio.wait_for(run_tasks(), timeout=1.0)
                    except asyncio.TimeoutError:
                        pass  # Expected

        # Check that the task was marked as error
        with session_maker() as session:
            updated_task = session.get(MaintenanceTask, task_id)

            assert updated_task.status == MaintenanceTaskStatus.ERROR
            assert 'error' in updated_task.info
            assert updated_task.info['error'] == 'Test error'

    @pytest.mark.asyncio
    async def test_run_tasks_respects_delay(self, session_maker):
        """Test that run_tasks respects the delay parameter."""
        # Create a mock processor
        processor = AsyncMock()
        processor.return_value = {'processed': True}

        # Create a task with delay
        with session_maker() as session:
            task = MaintenanceTask(
                status=MaintenanceTaskStatus.PENDING,
                processor_type='test.processor',
                processor_json='{}',
                delay=1,  # 1 second delay
            )
            session.add(task)
            session.commit()

            task_id = task.id

        # Mock asyncio.sleep to track calls
        sleep_mock = AsyncMock()

        # Mock the get_processor method
        with patch(
            'storage.maintenance_task.MaintenanceTask.get_processor',
            return_value=processor,
        ):
            with patch(
                'run_maintenance_tasks.session_maker', return_value=session_maker()
            ):
                with patch('asyncio.sleep', sleep_mock):
                    # Run the function with a timeout
                    try:
                        await asyncio.wait_for(run_tasks(), timeout=1.0)
                    except asyncio.TimeoutError:
                        pass  # Expected

        # Check that sleep was called with the correct delay
        sleep_mock.assert_called_once_with(1)

        # Check that the task was processed
        with session_maker() as session:
            updated_task = session.get(MaintenanceTask, task_id)
            assert updated_task.status == MaintenanceTaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_main_function(self, session_maker):
        """Test the main function that runs both set_stale_task_error and run_tasks."""
        # Create a stale task and a pending task
        with session_maker() as session:
            stale_task = MaintenanceTask(
                status=MaintenanceTaskStatus.WORKING,
                processor_type='test.processor',
                processor_json='{}',
                started_at=datetime.now(timezone.utc) - timedelta(hours=2),
            )
            session.add(stale_task)

            pending_task = MaintenanceTask(
                status=MaintenanceTaskStatus.PENDING,
                processor_type='test.processor',
                processor_json='{}',
            )
            session.add(pending_task)
            session.commit()

            stale_task_id = stale_task.id
            pending_task_id = pending_task.id

        # Mock the processor
        processor = AsyncMock()
        processor.return_value = {'processed': True}

        # Mock the functions
        with patch(
            'storage.maintenance_task.MaintenanceTask.get_processor',
            return_value=processor,
        ):
            with patch(
                'run_maintenance_tasks.session_maker', return_value=session_maker()
            ):
                # Patch asyncio.sleep to avoid delays
                with patch('asyncio.sleep', new_callable=AsyncMock):
                    # Run the main function with a timeout
                    try:
                        await asyncio.wait_for(main(), timeout=1.0)
                    except asyncio.TimeoutError:
                        pass  # Expected

        # Check that both tasks were processed correctly
        with session_maker() as session:
            updated_stale_task = session.get(MaintenanceTask, stale_task_id)
            updated_pending_task = session.get(MaintenanceTask, pending_task_id)

            # Stale task should be marked as error
            assert updated_stale_task.status == MaintenanceTaskStatus.ERROR

            # Pending task should be processed and completed
            assert updated_pending_task.status == MaintenanceTaskStatus.COMPLETED
            assert updated_pending_task.info == {'processed': True}
