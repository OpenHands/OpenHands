"""Tests for GoogleCloudEventService."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from openhands.agent_server.models import EventPage
from openhands.app_server.app_conversation.app_conversation_info_service import (
    AppConversationInfoService,
)
from openhands.app_server.app_conversation.app_conversation_models import (
    AppConversationInfo,
)
from openhands.app_server.errors import OpenHandsError
from openhands.app_server.event.google_cloud_event_service import (
    GoogleCloudEventService,
)


@pytest.fixture
def mock_app_conversation_info_service():
    """Create a mock AppConversationInfoService."""
    return AsyncMock(spec=AppConversationInfoService)


@pytest.fixture
def mock_bucket():
    """Create a mock GCS bucket."""
    return MagicMock()


@pytest.fixture
def google_cloud_event_service(mock_app_conversation_info_service, mock_bucket):
    """Create a GoogleCloudEventService for testing."""
    return GoogleCloudEventService(
        app_conversation_info_service=mock_app_conversation_info_service,
        bucket=mock_bucket,
    )


@pytest.fixture
def sample_conversation():
    """Create a sample conversation."""
    return AppConversationInfo(
        id=uuid4(),
        created_by_user_id='test_user_123',
        sandbox_id='test_sandbox',
        title='Test Conversation',
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


@pytest.fixture
def sample_event():
    """Create a sample event mock."""
    event = MagicMock()
    event.id = uuid4()
    event.timestamp = datetime.now(UTC)
    event.__class__.__name__ = 'MessageEvent'
    event.model_dump = MagicMock(
        return_value={'id': str(event.id), 'timestamp': event.timestamp.isoformat()}
    )
    return event


class TestGoogleCloudEventService:
    """Test cases for GoogleCloudEventService."""

    def test_get_event_path(
        self, google_cloud_event_service, sample_event, sample_conversation
    ):
        """Test that event paths are generated correctly."""
        user_id = 'test_user_123'
        conversation_id = sample_conversation.id

        path = google_cloud_event_service._get_event_path(
            user_id, conversation_id, sample_event
        )

        assert path.startswith(f'users/{user_id}/v1_conversations/{conversation_id}/')
        assert '_MessageEvent_' in path

    def test_get_event_filename(self, google_cloud_event_service, sample_event):
        """Test that event filenames are generated correctly."""
        filename = google_cloud_event_service._get_event_filename(sample_event)

        parts = filename.split('_')
        assert len(parts) >= 3
        assert parts[1] == 'MessageEvent'
        assert len(parts[0]) == 14  # YYYYMMDDHHMMSS

    def test_timestamp_to_str_with_datetime(self, google_cloud_event_service):
        """Test timestamp conversion with datetime object."""
        dt = datetime(2024, 1, 15, 10, 30, 45)
        result = google_cloud_event_service._timestamp_to_str(dt)
        assert result == '20240115103045'

    def test_timestamp_to_str_with_string(self, google_cloud_event_service):
        """Test timestamp conversion with ISO string."""
        ts_str = '2024-01-15T10:30:45+00:00'
        result = google_cloud_event_service._timestamp_to_str(ts_str)
        assert result == '20240115103045'

    def test_timestamp_to_str_with_z_suffix(self, google_cloud_event_service):
        """Test timestamp conversion with Z suffix."""
        ts_str = '2024-01-15T10:30:45Z'
        result = google_cloud_event_service._timestamp_to_str(ts_str)
        assert result == '20240115103045'

    def test_get_conversation_prefix(
        self, google_cloud_event_service, sample_conversation
    ):
        """Test conversation prefix generation."""
        user_id = 'test_user_123'
        conversation_id = sample_conversation.id

        prefix = google_cloud_event_service._get_conversation_prefix(
            user_id, conversation_id
        )

        assert prefix == f'users/{user_id}/v1_conversations/{conversation_id}/'

    def test_parse_filename_valid(self, google_cloud_event_service):
        """Test parsing a valid filename."""
        filename = '20240115103045_MessageEvent_abc123def456'
        result = google_cloud_event_service._parse_filename(filename)

        assert result is not None
        assert result['timestamp'] == '20240115103045'
        assert result['kind'] == 'MessageEvent'
        assert result['event_id'] == 'abc123def456'

    def test_parse_filename_with_underscore_in_kind(self, google_cloud_event_service):
        """Test parsing filename with underscores in kind."""
        filename = '20240115103045_Some_Event_Type_abc123'
        result = google_cloud_event_service._parse_filename(filename)

        assert result is not None
        assert result['kind'] == 'Some_Event_Type'
        assert result['event_id'] == 'abc123'

    def test_parse_filename_invalid(self, google_cloud_event_service):
        """Test parsing an invalid filename."""
        filename = 'invalid'
        result = google_cloud_event_service._parse_filename(filename)
        assert result is None

    def test_get_filename_from_blob(self, google_cloud_event_service):
        """Test extracting filename from blob."""
        blob = MagicMock()
        blob.name = (
            'users/user123/v1_conversations/conv456/20240115103045_MessageEvent_abc123'
        )

        filename = google_cloud_event_service._get_filename_from_blob(blob)

        assert filename == '20240115103045_MessageEvent_abc123'

    def test_get_conversation_id_from_blob(self, google_cloud_event_service):
        """Test extracting conversation_id from blob."""
        conversation_id = uuid4()
        blob = MagicMock()
        blob.name = (
            f'users/user123/v1_conversations/{conversation_id}/20240115_Event_abc123'
        )

        result = google_cloud_event_service._get_conversation_id_from_blob(blob)

        assert result == conversation_id

    def test_get_conversation_id_from_blob_invalid(self, google_cloud_event_service):
        """Test extracting conversation_id from invalid blob path."""
        blob = MagicMock()
        blob.name = 'invalid/path/structure'

        result = google_cloud_event_service._get_conversation_id_from_blob(blob)

        assert result is None

    def test_filter_blobs_by_kind(self, google_cloud_event_service):
        """Test filtering blobs by event kind."""
        blob1 = MagicMock()
        blob1.name = 'path/20240115103045_MessageEvent_abc123'
        blob2 = MagicMock()
        blob2.name = 'path/20240115103046_ActionEvent_def456'

        blobs = [blob1, blob2]
        filtered = google_cloud_event_service._filter_blobs_by_criteria(
            blobs, kind__eq='MessageEvent'
        )

        assert len(filtered) == 1
        assert filtered[0] == blob1

    def test_filter_blobs_by_timestamp_gte(self, google_cloud_event_service):
        """Test filtering blobs by timestamp >= threshold."""
        blob1 = MagicMock()
        blob1.name = 'path/20240115100000_Event_abc123'
        blob2 = MagicMock()
        blob2.name = 'path/20240115120000_Event_def456'

        blobs = [blob1, blob2]
        threshold = datetime(2024, 1, 15, 11, 0, 0)
        filtered = google_cloud_event_service._filter_blobs_by_criteria(
            blobs, timestamp__gte=threshold
        )

        assert len(filtered) == 1
        assert filtered[0] == blob2

    def test_filter_blobs_by_timestamp_lt(self, google_cloud_event_service):
        """Test filtering blobs by timestamp < threshold."""
        blob1 = MagicMock()
        blob1.name = 'path/20240115100000_Event_abc123'
        blob2 = MagicMock()
        blob2.name = 'path/20240115120000_Event_def456'

        blobs = [blob1, blob2]
        threshold = datetime(2024, 1, 15, 11, 0, 0)
        filtered = google_cloud_event_service._filter_blobs_by_criteria(
            blobs, timestamp__lt=threshold
        )

        assert len(filtered) == 1
        assert filtered[0] == blob1

    async def test_get_user_id_for_conversation(
        self,
        google_cloud_event_service,
        mock_app_conversation_info_service,
        sample_conversation,
    ):
        """Test getting user_id for a conversation."""
        mock_app_conversation_info_service.get_app_conversation_info.return_value = (
            sample_conversation
        )

        user_id = await google_cloud_event_service._get_user_id_for_conversation(
            sample_conversation.id
        )

        assert user_id == 'test_user_123'
        mock_app_conversation_info_service.get_app_conversation_info.assert_called_once_with(
            sample_conversation.id
        )

    async def test_get_user_id_for_conversation_cached(
        self,
        google_cloud_event_service,
        mock_app_conversation_info_service,
        sample_conversation,
    ):
        """Test that user_id is cached for subsequent calls."""
        mock_app_conversation_info_service.get_app_conversation_info.return_value = (
            sample_conversation
        )

        # First call
        await google_cloud_event_service._get_user_id_for_conversation(
            sample_conversation.id
        )
        # Second call should use cache
        user_id = await google_cloud_event_service._get_user_id_for_conversation(
            sample_conversation.id
        )

        assert user_id == 'test_user_123'
        assert (
            mock_app_conversation_info_service.get_app_conversation_info.call_count == 1
        )

    async def test_get_user_id_for_conversation_not_found(
        self, google_cloud_event_service, mock_app_conversation_info_service
    ):
        """Test getting user_id for non-existent conversation."""
        mock_app_conversation_info_service.get_app_conversation_info.return_value = None
        conversation_id = uuid4()

        user_id = await google_cloud_event_service._get_user_id_for_conversation(
            conversation_id
        )

        assert user_id is None

    async def test_save_event_success(
        self,
        google_cloud_event_service,
        mock_app_conversation_info_service,
        mock_bucket,
        sample_conversation,
        sample_event,
    ):
        """Test saving an event successfully."""
        mock_app_conversation_info_service.get_app_conversation_info.return_value = (
            sample_conversation
        )
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_file = MagicMock()
        mock_blob.open.return_value.__enter__ = MagicMock(return_value=mock_file)
        mock_blob.open.return_value.__exit__ = MagicMock(return_value=False)

        await google_cloud_event_service.save_event(
            sample_conversation.id, sample_event
        )

        mock_bucket.blob.assert_called_once()
        blob_path = mock_bucket.blob.call_args[0][0]
        assert f'users/{sample_conversation.created_by_user_id}/' in blob_path
        assert f'v1_conversations/{sample_conversation.id}/' in blob_path

    async def test_save_event_conversation_not_found(
        self,
        google_cloud_event_service,
        mock_app_conversation_info_service,
        sample_event,
    ):
        """Test saving an event when conversation doesn't exist."""
        mock_app_conversation_info_service.get_app_conversation_info.return_value = None
        conversation_id = uuid4()

        with pytest.raises(OpenHandsError, match='No such conversation'):
            await google_cloud_event_service.save_event(conversation_id, sample_event)

    async def test_save_event_no_user_id(
        self,
        google_cloud_event_service,
        mock_app_conversation_info_service,
        sample_event,
    ):
        """Test saving an event when conversation has no user_id."""
        conversation = AppConversationInfo(
            id=uuid4(),
            created_by_user_id=None,  # No user_id
            sandbox_id='test_sandbox',
        )
        mock_app_conversation_info_service.get_app_conversation_info.return_value = (
            conversation
        )

        with pytest.raises(OpenHandsError, match='has no associated user'):
            await google_cloud_event_service.save_event(conversation.id, sample_event)

    async def test_search_events_by_conversation(
        self,
        google_cloud_event_service,
        mock_app_conversation_info_service,
        mock_bucket,
        sample_conversation,
    ):
        """Test searching events for a specific conversation."""
        mock_app_conversation_info_service.get_app_conversation_info.return_value = (
            sample_conversation
        )

        # Mock list_blobs to return empty list
        mock_bucket.list_blobs.return_value = []

        result = await google_cloud_event_service.search_events(
            conversation_id__eq=sample_conversation.id
        )

        assert isinstance(result, EventPage)
        assert result.items == []
        assert result.next_page_id is None

    async def test_search_events_conversation_not_found(
        self, google_cloud_event_service, mock_app_conversation_info_service
    ):
        """Test searching events for non-existent conversation."""
        mock_app_conversation_info_service.get_app_conversation_info.return_value = None
        conversation_id = uuid4()

        result = await google_cloud_event_service.search_events(
            conversation_id__eq=conversation_id
        )

        assert isinstance(result, EventPage)
        assert result.items == []
        assert result.next_page_id is None

    async def test_count_events_by_conversation(
        self,
        google_cloud_event_service,
        mock_app_conversation_info_service,
        mock_bucket,
        sample_conversation,
    ):
        """Test counting events for a specific conversation."""
        mock_app_conversation_info_service.get_app_conversation_info.return_value = (
            sample_conversation
        )

        # Create mock blobs
        blob1 = MagicMock()
        blob1.name = f'users/{sample_conversation.created_by_user_id}/v1_conversations/{sample_conversation.id}/20240115103045_Event_abc123'
        blob2 = MagicMock()
        blob2.name = f'users/{sample_conversation.created_by_user_id}/v1_conversations/{sample_conversation.id}/20240115103046_Event_def456'

        mock_bucket.list_blobs.return_value = [blob1, blob2]

        result = await google_cloud_event_service.count_events(
            conversation_id__eq=sample_conversation.id
        )

        assert result == 2

    async def test_count_events_conversation_not_found(
        self, google_cloud_event_service, mock_app_conversation_info_service
    ):
        """Test counting events for non-existent conversation."""
        mock_app_conversation_info_service.get_app_conversation_info.return_value = None
        conversation_id = uuid4()

        result = await google_cloud_event_service.count_events(
            conversation_id__eq=conversation_id
        )

        assert result == 0

    async def test_get_event_not_found(self, google_cloud_event_service, mock_bucket):
        """Test getting a non-existent event."""
        mock_bucket.list_blobs.return_value = []

        result = await google_cloud_event_service.get_event('nonexistent_id')

        assert result is None
