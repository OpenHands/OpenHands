"""Tests for SQLAppConversationStartTaskService.

This module tests the SQL implementation of AppConversationStartTaskService,
focusing on basic CRUD operations and batch operations using SQLite as a mock database.
"""

from typing import AsyncGenerator
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationStartRequest,
    AppConversationStartTask,
    AppConversationStartTaskSortOrder,
    AppConversationStartTaskStatus,
)
from openhands.app_server.app_conversation.sql_app_conversation_start_task_service import (
    SQLAppConversationStartTaskService,
)
from openhands.app_server.utils.sql_utils import Base


@pytest.fixture
async def async_engine():
    """Create an async SQLite engine for testing."""
    engine = create_async_engine(
        'sqlite+aiosqlite:///:memory:',
        poolclass=StaticPool,
        connect_args={'check_same_thread': False},
        echo=False,
    )

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()


@pytest.fixture
async def async_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create an async session for testing."""
    async_session_maker = async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session_maker() as session:
        yield session


@pytest.fixture
def service(async_session: AsyncSession) -> SQLAppConversationStartTaskService:
    """Create a SQLAppConversationStartTaskService instance for testing."""
    return SQLAppConversationStartTaskService(session=async_session)


@pytest.fixture
def sample_request() -> AppConversationStartRequest:
    """Create a sample AppConversationStartRequest for testing."""
    return AppConversationStartRequest(
        sandbox_id=None,
        initial_message=None,
        processors=[],
        llm_model='gpt-4',
        selected_repository=None,
        selected_branch=None,
        git_provider=None,
        title='Test Conversation',
        trigger=None,
        pr_number=[],
    )


@pytest.fixture
def sample_task(
    sample_request: AppConversationStartRequest,
) -> AppConversationStartTask:
    """Create a sample AppConversationStartTask for testing."""
    return AppConversationStartTask(
        id=uuid4(),
        created_by_user_id='test_user',
        status=AppConversationStartTaskStatus.WORKING,
        detail=None,
        app_conversation_id=None,
        sandbox_id=None,
        agent_server_url=None,
        request=sample_request,
    )


class TestSQLAppConversationStartTaskService:
    """Test cases for SQLAppConversationStartTaskService."""

    async def test_save_and_get_task(
        self,
        service: SQLAppConversationStartTaskService,
        sample_task: AppConversationStartTask,
    ):
        """Test saving and retrieving a single task."""
        # Save the task
        saved_task = await service.save_app_conversation_start_task(sample_task)

        # Verify the task was saved correctly
        assert saved_task.id == sample_task.id
        assert saved_task.created_by_user_id == sample_task.created_by_user_id
        assert saved_task.status == sample_task.status
        assert saved_task.request == sample_task.request

        # Retrieve the task
        retrieved_task = await service.get_app_conversation_start_task(sample_task.id)

        # Verify the retrieved task matches
        assert retrieved_task is not None
        assert retrieved_task.id == sample_task.id
        assert retrieved_task.created_by_user_id == sample_task.created_by_user_id
        assert retrieved_task.status == sample_task.status
        assert retrieved_task.request == sample_task.request

    async def test_get_nonexistent_task(
        self, service: SQLAppConversationStartTaskService
    ):
        """Test retrieving a task that doesn't exist."""
        nonexistent_id = uuid4()
        result = await service.get_app_conversation_start_task(nonexistent_id)
        assert result is None

    async def test_batch_get_tasks(
        self,
        service: SQLAppConversationStartTaskService,
        sample_request: AppConversationStartRequest,
    ):
        """Test batch retrieval of tasks."""
        # Create multiple tasks
        task1 = AppConversationStartTask(
            id=uuid4(),
            created_by_user_id='user1',
            status=AppConversationStartTaskStatus.WORKING,
            request=sample_request,
        )
        task2 = AppConversationStartTask(
            id=uuid4(),
            created_by_user_id='user2',
            status=AppConversationStartTaskStatus.READY,
            request=sample_request,
        )
        task3 = AppConversationStartTask(
            id=uuid4(),
            created_by_user_id='user3',
            status=AppConversationStartTaskStatus.ERROR,
            request=sample_request,
        )

        # Save all tasks
        await service.save_app_conversation_start_task(task1)
        await service.save_app_conversation_start_task(task2)
        await service.save_app_conversation_start_task(task3)

        # Test batch retrieval with all existing IDs
        task_ids = [task1.id, task2.id, task3.id]
        retrieved_tasks = await service.batch_get_app_conversation_start_tasks(task_ids)

        assert len(retrieved_tasks) == 3
        assert all(task is not None for task in retrieved_tasks)

        # Verify order is preserved
        assert retrieved_tasks[0].id == task1.id
        assert retrieved_tasks[1].id == task2.id
        assert retrieved_tasks[2].id == task3.id

    async def test_batch_get_tasks_with_missing(
        self,
        service: SQLAppConversationStartTaskService,
        sample_task: AppConversationStartTask,
    ):
        """Test batch retrieval with some missing tasks."""
        # Save one task
        await service.save_app_conversation_start_task(sample_task)

        # Request batch with existing and non-existing IDs
        nonexistent_id = uuid4()
        task_ids = [sample_task.id, nonexistent_id]
        retrieved_tasks = await service.batch_get_app_conversation_start_tasks(task_ids)

        assert len(retrieved_tasks) == 2
        assert retrieved_tasks[0] is not None
        assert retrieved_tasks[0].id == sample_task.id
        assert retrieved_tasks[1] is None

    async def test_batch_get_empty_list(
        self, service: SQLAppConversationStartTaskService
    ):
        """Test batch retrieval with empty list."""
        result = await service.batch_get_app_conversation_start_tasks([])
        assert result == []

    async def test_update_task_status(
        self,
        service: SQLAppConversationStartTaskService,
        sample_task: AppConversationStartTask,
    ):
        """Test updating a task's status."""
        # Save initial task
        await service.save_app_conversation_start_task(sample_task)

        # Update the task status
        sample_task.status = AppConversationStartTaskStatus.READY
        sample_task.app_conversation_id = uuid4()
        sample_task.sandbox_id = 'test_sandbox'
        sample_task.agent_server_url = 'http://localhost:8000'

        # Save the updated task
        updated_task = await service.save_app_conversation_start_task(sample_task)

        # Verify the update
        assert updated_task.status == AppConversationStartTaskStatus.READY
        assert updated_task.app_conversation_id == sample_task.app_conversation_id
        assert updated_task.sandbox_id == 'test_sandbox'
        assert updated_task.agent_server_url == 'http://localhost:8000'

        # Retrieve and verify persistence
        retrieved_task = await service.get_app_conversation_start_task(sample_task.id)
        assert retrieved_task is not None
        assert retrieved_task.status == AppConversationStartTaskStatus.READY
        assert retrieved_task.app_conversation_id == sample_task.app_conversation_id

    async def test_user_isolation(
        self,
        async_session: AsyncSession,
        sample_request: AppConversationStartRequest,
    ):
        """Test that users can only access their own tasks."""
        # Create services for different users
        user1_service = SQLAppConversationStartTaskService(
            session=async_session, user_id='user1'
        )
        user2_service = SQLAppConversationStartTaskService(
            session=async_session, user_id='user2'
        )

        # Create tasks for different users
        user1_task = AppConversationStartTask(
            id=uuid4(),
            created_by_user_id='user1',
            status=AppConversationStartTaskStatus.WORKING,
            request=sample_request,
        )
        user2_task = AppConversationStartTask(
            id=uuid4(),
            created_by_user_id='user2',
            status=AppConversationStartTaskStatus.WORKING,
            request=sample_request,
        )

        # Save tasks using respective services
        await user1_service.save_app_conversation_start_task(user1_task)
        await user2_service.save_app_conversation_start_task(user2_task)

        # Test that user1 can only access their task
        user1_retrieved = await user1_service.get_app_conversation_start_task(
            user1_task.id
        )
        user1_cannot_access = await user1_service.get_app_conversation_start_task(
            user2_task.id
        )

        assert user1_retrieved is not None
        assert user1_retrieved.id == user1_task.id
        assert user1_cannot_access is None

        # Test that user2 can only access their task
        user2_retrieved = await user2_service.get_app_conversation_start_task(
            user2_task.id
        )
        user2_cannot_access = await user2_service.get_app_conversation_start_task(
            user1_task.id
        )

        assert user2_retrieved is not None
        assert user2_retrieved.id == user2_task.id
        assert user2_cannot_access is None

    async def test_batch_get_with_user_isolation(
        self,
        async_session: AsyncSession,
        sample_request: AppConversationStartRequest,
    ):
        """Test batch retrieval with user isolation."""
        # Create services for different users
        user1_service = SQLAppConversationStartTaskService(
            session=async_session, user_id='user1'
        )
        user2_service = SQLAppConversationStartTaskService(
            session=async_session, user_id='user2'
        )

        # Create tasks for different users
        user1_task = AppConversationStartTask(
            id=uuid4(),
            created_by_user_id='user1',
            status=AppConversationStartTaskStatus.WORKING,
            request=sample_request,
        )
        user2_task = AppConversationStartTask(
            id=uuid4(),
            created_by_user_id='user2',
            status=AppConversationStartTaskStatus.WORKING,
            request=sample_request,
        )

        # Save tasks
        await user1_service.save_app_conversation_start_task(user1_task)
        await user2_service.save_app_conversation_start_task(user2_task)

        # Test batch retrieval with user isolation
        task_ids = [user1_task.id, user2_task.id]
        user1_results = await user1_service.batch_get_app_conversation_start_tasks(
            task_ids
        )

        # User1 should only see their task, user2's task should be None
        assert len(user1_results) == 2
        assert user1_results[0] is not None
        assert user1_results[0].id == user1_task.id
        assert user1_results[1] is None

    async def test_task_timestamps(
        self,
        service: SQLAppConversationStartTaskService,
        sample_task: AppConversationStartTask,
    ):
        """Test that timestamps are properly set and updated."""
        # Save initial task
        saved_task = await service.save_app_conversation_start_task(sample_task)

        # Verify timestamps are set
        assert saved_task.created_at is not None
        assert saved_task.updated_at is not None

        original_created_at = saved_task.created_at
        original_updated_at = saved_task.updated_at

        # Update the task
        saved_task.status = AppConversationStartTaskStatus.READY
        updated_task = await service.save_app_conversation_start_task(saved_task)

        # Verify created_at stays the same but updated_at changes
        assert updated_task.created_at == original_created_at
        assert updated_task.updated_at > original_updated_at

    async def test_search_app_conversation_start_tasks_basic(
        self,
        service: SQLAppConversationStartTaskService,
        sample_request: AppConversationStartRequest,
    ):
        """Test basic search functionality for start tasks."""
        # Create multiple tasks
        task1 = AppConversationStartTask(
            id=uuid4(),
            created_by_user_id='user1',
            status=AppConversationStartTaskStatus.WORKING,
            request=sample_request,
        )
        task2 = AppConversationStartTask(
            id=uuid4(),
            created_by_user_id='user1',
            status=AppConversationStartTaskStatus.READY,
            request=sample_request,
        )

        # Save tasks
        await service.save_app_conversation_start_task(task1)
        await service.save_app_conversation_start_task(task2)

        # Search for all tasks
        result = await service.search_app_conversation_start_tasks()

        assert len(result.items) == 2
        assert result.next_page_id is None

        # Verify tasks are returned in descending order by created_at (default)
        task_ids = [task.id for task in result.items]
        assert task2.id in task_ids
        assert task1.id in task_ids

    async def test_search_app_conversation_start_tasks_with_conversation_filter(
        self,
        service: SQLAppConversationStartTaskService,
        sample_request: AppConversationStartRequest,
    ):
        """Test search with conversation_id filter."""
        conversation_id1 = uuid4()
        conversation_id2 = uuid4()

        # Create tasks with different conversation IDs
        task1 = AppConversationStartTask(
            id=uuid4(),
            created_by_user_id='user1',
            status=AppConversationStartTaskStatus.READY,
            app_conversation_id=conversation_id1,
            request=sample_request,
        )
        task2 = AppConversationStartTask(
            id=uuid4(),
            created_by_user_id='user1',
            status=AppConversationStartTaskStatus.READY,
            app_conversation_id=conversation_id2,
            request=sample_request,
        )
        task3 = AppConversationStartTask(
            id=uuid4(),
            created_by_user_id='user1',
            status=AppConversationStartTaskStatus.WORKING,
            app_conversation_id=None,
            request=sample_request,
        )

        # Save tasks
        await service.save_app_conversation_start_task(task1)
        await service.save_app_conversation_start_task(task2)
        await service.save_app_conversation_start_task(task3)

        # Search for tasks with specific conversation ID
        result = await service.search_app_conversation_start_tasks(
            conversation_id__eq=conversation_id1
        )

        assert len(result.items) == 1
        assert result.items[0].id == task1.id
        assert result.items[0].app_conversation_id == conversation_id1

    async def test_search_app_conversation_start_tasks_sorting(
        self,
        service: SQLAppConversationStartTaskService,
        sample_request: AppConversationStartRequest,
    ):
        """Test search with different sort orders."""
        # Create tasks with slight time differences
        task1 = AppConversationStartTask(
            id=uuid4(),
            created_by_user_id='user1',
            status=AppConversationStartTaskStatus.WORKING,
            request=sample_request,
        )
        await service.save_app_conversation_start_task(task1)

        task2 = AppConversationStartTask(
            id=uuid4(),
            created_by_user_id='user1',
            status=AppConversationStartTaskStatus.READY,
            request=sample_request,
        )
        await service.save_app_conversation_start_task(task2)

        # Test ascending order
        result_asc = await service.search_app_conversation_start_tasks(
            sort_order=AppConversationStartTaskSortOrder.CREATED_AT
        )
        assert len(result_asc.items) == 2
        assert result_asc.items[0].id == task1.id  # First created
        assert result_asc.items[1].id == task2.id  # Second created

        # Test descending order (default)
        result_desc = await service.search_app_conversation_start_tasks(
            sort_order=AppConversationStartTaskSortOrder.CREATED_AT_DESC
        )
        assert len(result_desc.items) == 2
        assert result_desc.items[0].id == task2.id  # Most recent first
        assert result_desc.items[1].id == task1.id  # Older second

    async def test_search_app_conversation_start_tasks_pagination(
        self,
        service: SQLAppConversationStartTaskService,
        sample_request: AppConversationStartRequest,
    ):
        """Test search with pagination."""
        # Create multiple tasks
        tasks = []
        for i in range(5):
            task = AppConversationStartTask(
                id=uuid4(),
                created_by_user_id='user1',
                status=AppConversationStartTaskStatus.WORKING,
                request=sample_request,
            )
            tasks.append(task)
            await service.save_app_conversation_start_task(task)

        # Test first page with limit 2
        result_page1 = await service.search_app_conversation_start_tasks(limit=2)
        assert len(result_page1.items) == 2
        assert result_page1.next_page_id == '2'

        # Test second page
        result_page2 = await service.search_app_conversation_start_tasks(
            page_id='2', limit=2
        )
        assert len(result_page2.items) == 2
        assert result_page2.next_page_id == '4'

        # Test last page
        result_page3 = await service.search_app_conversation_start_tasks(
            page_id='4', limit=2
        )
        assert len(result_page3.items) == 1
        assert result_page3.next_page_id is None

    async def test_count_app_conversation_start_tasks_basic(
        self,
        service: SQLAppConversationStartTaskService,
        sample_request: AppConversationStartRequest,
    ):
        """Test basic count functionality for start tasks."""
        # Initially no tasks
        count = await service.count_app_conversation_start_tasks()
        assert count == 0

        # Create and save tasks
        task1 = AppConversationStartTask(
            id=uuid4(),
            created_by_user_id='user1',
            status=AppConversationStartTaskStatus.WORKING,
            request=sample_request,
        )
        task2 = AppConversationStartTask(
            id=uuid4(),
            created_by_user_id='user1',
            status=AppConversationStartTaskStatus.READY,
            request=sample_request,
        )

        await service.save_app_conversation_start_task(task1)
        count = await service.count_app_conversation_start_tasks()
        assert count == 1

        await service.save_app_conversation_start_task(task2)
        count = await service.count_app_conversation_start_tasks()
        assert count == 2

    async def test_count_app_conversation_start_tasks_with_filter(
        self,
        service: SQLAppConversationStartTaskService,
        sample_request: AppConversationStartRequest,
    ):
        """Test count with conversation_id filter."""
        conversation_id1 = uuid4()
        conversation_id2 = uuid4()

        # Create tasks with different conversation IDs
        task1 = AppConversationStartTask(
            id=uuid4(),
            created_by_user_id='user1',
            status=AppConversationStartTaskStatus.READY,
            app_conversation_id=conversation_id1,
            request=sample_request,
        )
        task2 = AppConversationStartTask(
            id=uuid4(),
            created_by_user_id='user1',
            status=AppConversationStartTaskStatus.READY,
            app_conversation_id=conversation_id2,
            request=sample_request,
        )
        task3 = AppConversationStartTask(
            id=uuid4(),
            created_by_user_id='user1',
            status=AppConversationStartTaskStatus.WORKING,
            app_conversation_id=conversation_id1,
            request=sample_request,
        )

        # Save tasks
        await service.save_app_conversation_start_task(task1)
        await service.save_app_conversation_start_task(task2)
        await service.save_app_conversation_start_task(task3)

        # Count all tasks
        total_count = await service.count_app_conversation_start_tasks()
        assert total_count == 3

        # Count tasks for specific conversation
        conv1_count = await service.count_app_conversation_start_tasks(
            conversation_id__eq=conversation_id1
        )
        assert conv1_count == 2

        conv2_count = await service.count_app_conversation_start_tasks(
            conversation_id__eq=conversation_id2
        )
        assert conv2_count == 1

    async def test_search_and_count_with_user_isolation(
        self,
        async_session: AsyncSession,
        sample_request: AppConversationStartRequest,
    ):
        """Test search and count with user isolation."""
        # Create services for different users
        user1_service = SQLAppConversationStartTaskService(
            session=async_session, user_id='user1'
        )
        user2_service = SQLAppConversationStartTaskService(
            session=async_session, user_id='user2'
        )

        # Create tasks for different users
        user1_task = AppConversationStartTask(
            id=uuid4(),
            created_by_user_id='user1',
            status=AppConversationStartTaskStatus.WORKING,
            request=sample_request,
        )
        user2_task = AppConversationStartTask(
            id=uuid4(),
            created_by_user_id='user2',
            status=AppConversationStartTaskStatus.WORKING,
            request=sample_request,
        )

        # Save tasks using respective services
        await user1_service.save_app_conversation_start_task(user1_task)
        await user2_service.save_app_conversation_start_task(user2_task)

        # Test search isolation
        user1_search = await user1_service.search_app_conversation_start_tasks()
        assert len(user1_search.items) == 1
        assert user1_search.items[0].id == user1_task.id

        user2_search = await user2_service.search_app_conversation_start_tasks()
        assert len(user2_search.items) == 1
        assert user2_search.items[0].id == user2_task.id

        # Test count isolation
        user1_count = await user1_service.count_app_conversation_start_tasks()
        assert user1_count == 1

        user2_count = await user2_service.count_app_conversation_start_tasks()
        assert user2_count == 1

    async def test_search_app_conversation_start_tasks_with_created_at_gte_filter(
        self,
        service: SQLAppConversationStartTaskService,
        sample_request: AppConversationStartRequest,
    ):
        """Test search with created_at__gte filter."""
        from datetime import timedelta

        from openhands.agent_server.models import utc_now

        # Create tasks with different creation times
        base_time = utc_now()

        # Task 1: created 2 hours ago
        task1 = AppConversationStartTask(
            id=uuid4(),
            created_by_user_id='user1',
            status=AppConversationStartTaskStatus.WORKING,
            request=sample_request,
        )
        task1.created_at = base_time - timedelta(hours=2)
        await service.save_app_conversation_start_task(task1)

        # Task 2: created 1 hour ago
        task2 = AppConversationStartTask(
            id=uuid4(),
            created_by_user_id='user1',
            status=AppConversationStartTaskStatus.READY,
            request=sample_request,
        )
        task2.created_at = base_time - timedelta(hours=1)
        await service.save_app_conversation_start_task(task2)

        # Task 3: created 30 minutes ago
        task3 = AppConversationStartTask(
            id=uuid4(),
            created_by_user_id='user1',
            status=AppConversationStartTaskStatus.WORKING,
            request=sample_request,
        )
        task3.created_at = base_time - timedelta(minutes=30)
        await service.save_app_conversation_start_task(task3)

        # Search for tasks created in the last 90 minutes
        filter_time = base_time - timedelta(minutes=90)
        result = await service.search_app_conversation_start_tasks(
            created_at__gte=filter_time
        )

        # Should return task2 and task3 (created within last 90 minutes)
        assert len(result.items) == 2
        task_ids = [task.id for task in result.items]
        assert task2.id in task_ids
        assert task3.id in task_ids
        assert task1.id not in task_ids

        # Test count with the same filter
        count = await service.count_app_conversation_start_tasks(
            created_at__gte=filter_time
        )
        assert count == 2

        # Search for tasks created in the last 45 minutes
        filter_time_recent = base_time - timedelta(minutes=45)
        result_recent = await service.search_app_conversation_start_tasks(
            created_at__gte=filter_time_recent
        )

        # Should return only task3
        assert len(result_recent.items) == 1
        assert result_recent.items[0].id == task3.id

        # Test count with recent filter
        count_recent = await service.count_app_conversation_start_tasks(
            created_at__gte=filter_time_recent
        )
        assert count_recent == 1

    async def test_search_app_conversation_start_tasks_combined_filters(
        self,
        service: SQLAppConversationStartTaskService,
        sample_request: AppConversationStartRequest,
    ):
        """Test search with both conversation_id and created_at__gte filters."""
        from datetime import timedelta

        from openhands.agent_server.models import utc_now

        conversation_id1 = uuid4()
        conversation_id2 = uuid4()
        base_time = utc_now()

        # Task 1: conversation_id1, created 2 hours ago
        task1 = AppConversationStartTask(
            id=uuid4(),
            created_by_user_id='user1',
            status=AppConversationStartTaskStatus.WORKING,
            app_conversation_id=conversation_id1,
            request=sample_request,
        )
        task1.created_at = base_time - timedelta(hours=2)
        await service.save_app_conversation_start_task(task1)

        # Task 2: conversation_id1, created 30 minutes ago
        task2 = AppConversationStartTask(
            id=uuid4(),
            created_by_user_id='user1',
            status=AppConversationStartTaskStatus.READY,
            app_conversation_id=conversation_id1,
            request=sample_request,
        )
        task2.created_at = base_time - timedelta(minutes=30)
        await service.save_app_conversation_start_task(task2)

        # Task 3: conversation_id2, created 30 minutes ago
        task3 = AppConversationStartTask(
            id=uuid4(),
            created_by_user_id='user1',
            status=AppConversationStartTaskStatus.WORKING,
            app_conversation_id=conversation_id2,
            request=sample_request,
        )
        task3.created_at = base_time - timedelta(minutes=30)
        await service.save_app_conversation_start_task(task3)

        # Search for tasks with conversation_id1 created in the last hour
        filter_time = base_time - timedelta(hours=1)
        result = await service.search_app_conversation_start_tasks(
            conversation_id__eq=conversation_id1, created_at__gte=filter_time
        )

        # Should return only task2 (conversation_id1 and created within last hour)
        assert len(result.items) == 1
        assert result.items[0].id == task2.id
        assert result.items[0].app_conversation_id == conversation_id1

        # Test count with combined filters
        count = await service.count_app_conversation_start_tasks(
            conversation_id__eq=conversation_id1, created_at__gte=filter_time
        )
        assert count == 1
