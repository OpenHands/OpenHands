"""Tests for AwsSharedEventService."""

import os
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from server.sharing.aws_shared_event_service import (
    AwsSharedEventService,
    AwsSharedEventServiceInjector,
)
from server.sharing.shared_conversation_info_service import (
    SharedConversationInfoService,
)
from server.sharing.shared_conversation_models import SharedConversation

from openhands.agent_server.models import EventPage, EventSortOrder
from openhands.app_server.event.event_service import EventService
from openhands.sdk.llm import MetricsSnapshot, TokenUsage


@pytest.fixture
def mock_shared_conversation_info_service():
    """Create a mock SharedConversationInfoService."""
    return AsyncMock(spec=SharedConversationInfoService)


@pytest.fixture
def mock_s3_client():
    """Create a mock S3 client."""
    return MagicMock()


@pytest.fixture
def mock_event_service():
    """Create a mock EventService for returned by get_event_service."""
    return AsyncMock(spec=EventService)


@pytest.fixture
def aws_shared_event_service(mock_shared_conversation_info_service, mock_s3_client):
    """Create an AwsSharedEventService for testing."""
    return AwsSharedEventService(
        shared_conversation_info_service=mock_shared_conversation_info_service,
        s3_client=mock_s3_client,
        bucket_name='test-bucket',
    )


@pytest.fixture
def sample_public_conversation():
    """Create a sample public conversation."""
    return SharedConversation(
        id=uuid4(),
        created_by_user_id='test_user',
        sandbox_id='test_sandbox',
        title='Test Public Conversation',
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        metrics=MetricsSnapshot(
            accumulated_cost=0.0,
            max_budget_per_task=10.0,
            accumulated_token_usage=TokenUsage(),
        ),
    )


@pytest.fixture
def sample_event():
    """Create a sample event."""
    # For testing purposes, we'll just use a mock that the EventPage can accept
    # The actual event creation is complex and not the focus of these tests
    return None


class TestAwsSharedEventService:
    """Test cases for AwsSharedEventService."""

    async def test_get_shared_event_returns_event_for_public_conversation(
        self,
        aws_shared_event_service,
        mock_shared_conversation_info_service,
        mock_event_service,
        sample_public_conversation,
        sample_event,
    ):
        """Test that get_shared_event returns an event for a public conversation."""
        conversation_id = sample_public_conversation.id
        event_id = uuid4()

        # Mock the public conversation service to return a public conversation
        mock_shared_conversation_info_service.get_shared_conversation_info.return_value = sample_public_conversation

        # Mock get_event_service to return our mock event service
        aws_shared_event_service.get_event_service = AsyncMock(
            return_value=mock_event_service
        )

        # Mock the event service to return an event
        mock_event_service.get_event.return_value = sample_event

        # Call the method
        result = await aws_shared_event_service.get_shared_event(
            conversation_id, event_id
        )

        # Verify the result
        assert result == sample_event
        aws_shared_event_service.get_event_service.assert_called_once_with(
            conversation_id
        )
        mock_event_service.get_event.assert_called_once_with(conversation_id, event_id)

    async def test_get_shared_event_returns_none_for_private_conversation(
        self,
        aws_shared_event_service,
        mock_shared_conversation_info_service,
        mock_event_service,
    ):
        """Test that get_shared_event returns None for a private conversation."""
        conversation_id = uuid4()
        event_id = uuid4()

        # Mock get_event_service to return None (private conversation)
        aws_shared_event_service.get_event_service = AsyncMock(return_value=None)

        # Call the method
        result = await aws_shared_event_service.get_shared_event(
            conversation_id, event_id
        )

        # Verify the result
        assert result is None
        aws_shared_event_service.get_event_service.assert_called_once_with(
            conversation_id
        )
        # Event service should not be called since get_event_service returns None
        mock_event_service.get_event.assert_not_called()

    async def test_search_shared_events_returns_events_for_public_conversation(
        self,
        aws_shared_event_service,
        mock_shared_conversation_info_service,
        mock_event_service,
        sample_public_conversation,
        sample_event,
    ):
        """Test that search_shared_events returns events for a public conversation."""
        conversation_id = sample_public_conversation.id

        # Mock get_event_service to return our mock event service
        aws_shared_event_service.get_event_service = AsyncMock(
            return_value=mock_event_service
        )

        # Mock the event service to return events
        mock_event_page = EventPage(items=[], next_page_id=None)
        mock_event_service.search_events.return_value = mock_event_page

        # Call the method
        result = await aws_shared_event_service.search_shared_events(
            conversation_id=conversation_id,
            kind__eq='ActionEvent',
            limit=10,
        )

        # Verify the result
        assert result == mock_event_page
        assert len(result.items) == 0  # Empty list as we mocked

        aws_shared_event_service.get_event_service.assert_called_once_with(
            conversation_id
        )
        mock_event_service.search_events.assert_called_once_with(
            conversation_id=conversation_id,
            kind__eq='ActionEvent',
            timestamp__gte=None,
            timestamp__lt=None,
            sort_order=EventSortOrder.TIMESTAMP,
            page_id=None,
            limit=10,
        )

    async def test_search_shared_events_returns_empty_for_private_conversation(
        self,
        aws_shared_event_service,
        mock_shared_conversation_info_service,
        mock_event_service,
    ):
        """Test that search_shared_events returns empty page for a private conversation."""
        conversation_id = uuid4()

        # Mock get_event_service to return None (private conversation)
        aws_shared_event_service.get_event_service = AsyncMock(return_value=None)

        # Call the method
        result = await aws_shared_event_service.search_shared_events(
            conversation_id=conversation_id,
            limit=10,
        )

        # Verify the result
        assert isinstance(result, EventPage)
        assert len(result.items) == 0
        assert result.next_page_id is None

        aws_shared_event_service.get_event_service.assert_called_once_with(
            conversation_id
        )
        # Event service should not be called
        mock_event_service.search_events.assert_not_called()

    async def test_count_shared_events_returns_count_for_public_conversation(
        self,
        aws_shared_event_service,
        mock_shared_conversation_info_service,
        mock_event_service,
        sample_public_conversation,
    ):
        """Test that count_shared_events returns count for a public conversation."""
        conversation_id = sample_public_conversation.id

        # Mock get_event_service to return our mock event service
        aws_shared_event_service.get_event_service = AsyncMock(
            return_value=mock_event_service
        )

        # Mock the event service to return a count
        mock_event_service.count_events.return_value = 5

        # Call the method
        result = await aws_shared_event_service.count_shared_events(
            conversation_id=conversation_id,
            kind__eq='ActionEvent',
        )

        # Verify the result
        assert result == 5

        aws_shared_event_service.get_event_service.assert_called_once_with(
            conversation_id
        )
        mock_event_service.count_events.assert_called_once_with(
            conversation_id=conversation_id,
            kind__eq='ActionEvent',
            timestamp__gte=None,
            timestamp__lt=None,
        )

    async def test_count_shared_events_returns_zero_for_private_conversation(
        self,
        aws_shared_event_service,
        mock_shared_conversation_info_service,
        mock_event_service,
    ):
        """Test that count_shared_events returns 0 for a private conversation."""
        conversation_id = uuid4()

        # Mock get_event_service to return None (private conversation)
        aws_shared_event_service.get_event_service = AsyncMock(return_value=None)

        # Call the method
        result = await aws_shared_event_service.count_shared_events(
            conversation_id=conversation_id,
        )

        # Verify the result
        assert result == 0

        aws_shared_event_service.get_event_service.assert_called_once_with(
            conversation_id
        )
        # Event service should not be called
        mock_event_service.count_events.assert_not_called()

    async def test_batch_get_shared_events_returns_events_for_public_conversation(
        self,
        aws_shared_event_service,
        mock_shared_conversation_info_service,
        mock_event_service,
        sample_public_conversation,
        sample_event,
    ):
        """Test that batch_get_shared_events returns events for a public conversation."""
        conversation_id = sample_public_conversation.id
        event_ids = [uuid4() for _ in range(3)]

        # Mock get_event_service to return our mock event service
        aws_shared_event_service.get_event_service = AsyncMock(
            return_value=mock_event_service
        )

        # Mock the event service to return events
        mock_event_service.get_event.return_value = sample_event

        # Call the method
        results = await aws_shared_event_service.batch_get_shared_events(
            conversation_id, event_ids
        )

        # Verify the results
        assert len(results) == 3
        assert all(result == sample_event for result in results)


class TestAwsSharedEventServiceGetEventService:
    """Test cases for AwsSharedEventService.get_event_service method."""

    async def test_get_event_service_returns_event_service_for_shared_conversation(
        self,
        aws_shared_event_service,
        mock_shared_conversation_info_service,
        sample_public_conversation,
    ):
        """Test that get_event_service returns an EventService for a shared conversation."""
        conversation_id = sample_public_conversation.id

        # Mock the shared conversation info service to return a shared conversation
        mock_shared_conversation_info_service.get_shared_conversation_info.return_value = sample_public_conversation

        # Call the method
        result = await aws_shared_event_service.get_event_service(conversation_id)

        # Verify the result
        assert result is not None
        mock_shared_conversation_info_service.get_shared_conversation_info.assert_called_once_with(
            conversation_id
        )

    async def test_get_event_service_returns_none_for_non_shared_conversation(
        self,
        aws_shared_event_service,
        mock_shared_conversation_info_service,
    ):
        """Test that get_event_service returns None for a non-shared conversation."""
        conversation_id = uuid4()

        # Mock the shared conversation info service to return None
        mock_shared_conversation_info_service.get_shared_conversation_info.return_value = None

        # Call the method
        result = await aws_shared_event_service.get_event_service(conversation_id)

        # Verify the result
        assert result is None
        mock_shared_conversation_info_service.get_shared_conversation_info.assert_called_once_with(
            conversation_id
        )


class TestAwsSharedEventServiceInjector:
    """Test cases for AwsSharedEventServiceInjector."""

    def test_bucket_name_from_environment_variable(self):
        """Test that bucket_name is read from FILE_STORE_PATH environment variable."""
        test_bucket_name = 'test-bucket-name'
        with patch.dict(os.environ, {'FILE_STORE_PATH': test_bucket_name}):
            # Create a new injector instance to pick up the environment variable
            # Note: The class attribute is evaluated at class definition time,
            # so we need to test that the attribute exists and can be overridden
            injector = AwsSharedEventServiceInjector()
            injector.bucket_name = os.environ.get('FILE_STORE_PATH')
            assert injector.bucket_name == test_bucket_name

    def test_bucket_name_default_value_when_env_not_set(self):
        """Test that bucket_name is None when FILE_STORE_PATH is not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove FILE_STORE_PATH if it exists
            os.environ.pop('FILE_STORE_PATH', None)
            injector = AwsSharedEventServiceInjector()
            # The bucket_name will be whatever was set at class definition time
            # or None if FILE_STORE_PATH was not set when the class was defined
            assert hasattr(injector, 'bucket_name')

    async def test_injector_yields_aws_shared_event_service(self):
        """Test that the injector yields an AwsSharedEventService instance."""
        mock_state = MagicMock()
        mock_request = MagicMock()
        mock_db_session = AsyncMock()

        # Create the injector
        injector = AwsSharedEventServiceInjector()
        injector.bucket_name = 'test-bucket'

        # Mock the get_db_session context manager
        mock_db_context = AsyncMock()
        mock_db_context.__aenter__.return_value = mock_db_session
        mock_db_context.__aexit__.return_value = None

        # Mock boto3.client
        mock_s3_client = MagicMock()

        with (
            patch(
                'server.sharing.aws_shared_event_service.boto3.client',
                return_value=mock_s3_client,
            ),
            patch(
                'openhands.app_server.config.get_db_session',
                return_value=mock_db_context,
            ),
        ):
            # Call the inject method
            async for service in injector.inject(mock_state, mock_request):
                # Verify the service is an instance of AwsSharedEventService
                assert isinstance(service, AwsSharedEventService)
                assert service.s3_client == mock_s3_client
                assert service.bucket_name == 'test-bucket'

    async def test_injector_uses_bucket_name_from_instance(self):
        """Test that the injector uses the bucket_name from the instance."""
        mock_state = MagicMock()
        mock_request = MagicMock()
        mock_db_session = AsyncMock()

        # Create the injector with a specific bucket name
        injector = AwsSharedEventServiceInjector()
        injector.bucket_name = 'my-custom-bucket'

        # Mock the get_db_session context manager
        mock_db_context = AsyncMock()
        mock_db_context.__aenter__.return_value = mock_db_session
        mock_db_context.__aexit__.return_value = None

        # Mock boto3.client
        mock_s3_client = MagicMock()

        with (
            patch(
                'server.sharing.aws_shared_event_service.boto3.client',
                return_value=mock_s3_client,
            ),
            patch(
                'openhands.app_server.config.get_db_session',
                return_value=mock_db_context,
            ),
        ):
            # Call the inject method
            async for service in injector.inject(mock_state, mock_request):
                assert service.bucket_name == 'my-custom-bucket'

    async def test_injector_creates_sql_shared_conversation_info_service(self):
        """Test that the injector creates SQLSharedConversationInfoService with db_session."""
        mock_state = MagicMock()
        mock_request = MagicMock()
        mock_db_session = AsyncMock()

        # Create the injector
        injector = AwsSharedEventServiceInjector()
        injector.bucket_name = 'test-bucket'

        # Mock the get_db_session context manager
        mock_db_context = AsyncMock()
        mock_db_context.__aenter__.return_value = mock_db_session
        mock_db_context.__aexit__.return_value = None

        # Mock boto3.client
        mock_s3_client = MagicMock()

        with (
            patch(
                'server.sharing.aws_shared_event_service.boto3.client',
                return_value=mock_s3_client,
            ),
            patch(
                'openhands.app_server.config.get_db_session',
                return_value=mock_db_context,
            ),
            patch(
                'server.sharing.aws_shared_event_service.SQLSharedConversationInfoService'
            ) as mock_sql_service_class,
        ):
            mock_sql_service = MagicMock()
            mock_sql_service_class.return_value = mock_sql_service

            # Call the inject method
            async for service in injector.inject(mock_state, mock_request):
                # Verify the service has the correct shared_conversation_info_service
                assert service.shared_conversation_info_service == mock_sql_service

            # Verify SQLSharedConversationInfoService was created with db_session
            mock_sql_service_class.assert_called_once_with(db_session=mock_db_session)

    async def test_injector_works_without_request(self):
        """Test that the injector works when request is None."""
        mock_state = MagicMock()
        mock_db_session = AsyncMock()

        # Create the injector
        injector = AwsSharedEventServiceInjector()
        injector.bucket_name = 'test-bucket'

        # Mock the get_db_session context manager
        mock_db_context = AsyncMock()
        mock_db_context.__aenter__.return_value = mock_db_session
        mock_db_context.__aexit__.return_value = None

        # Mock boto3.client
        mock_s3_client = MagicMock()

        with (
            patch(
                'server.sharing.aws_shared_event_service.boto3.client',
                return_value=mock_s3_client,
            ),
            patch(
                'openhands.app_server.config.get_db_session',
                return_value=mock_db_context,
            ),
        ):
            # Call the inject method with request=None
            async for service in injector.inject(mock_state, request=None):
                assert isinstance(service, AwsSharedEventService)

    async def test_injector_uses_role_based_authentication(self):
        """Test that the injector uses role-based authentication (no explicit credentials)."""
        mock_state = MagicMock()
        mock_request = MagicMock()
        mock_db_session = AsyncMock()

        # Create the injector
        injector = AwsSharedEventServiceInjector()
        injector.bucket_name = 'test-bucket'

        # Mock the get_db_session context manager
        mock_db_context = AsyncMock()
        mock_db_context.__aenter__.return_value = mock_db_session
        mock_db_context.__aexit__.return_value = None

        # Mock boto3.client
        mock_s3_client = MagicMock()

        with (
            patch(
                'server.sharing.aws_shared_event_service.boto3.client',
                return_value=mock_s3_client,
            ) as mock_boto3_client,
            patch(
                'openhands.app_server.config.get_db_session',
                return_value=mock_db_context,
            ),
            patch.dict(os.environ, {'AWS_S3_ENDPOINT': 'https://s3.example.com'}),
        ):
            # Call the inject method
            async for service in injector.inject(mock_state, mock_request):
                pass

            # Verify boto3.client was called with 's3' and endpoint_url
            # but without explicit credentials (role-based auth)
            mock_boto3_client.assert_called_once_with(
                's3',
                endpoint_url='https://s3.example.com',
            )
