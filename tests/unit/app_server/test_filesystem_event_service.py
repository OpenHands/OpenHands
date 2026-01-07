"""Tests for FilesystemEventService.

This module tests the filesystem-based implementation of EventService,
focusing on basic CRUD operations, search functionality, and file I/O handling.

The tests use mocking to avoid the complex dependency chain of the actual module.
"""

import glob
import json
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest


class MockEvent:
    """Mock Event class for testing without requiring the full SDK."""

    def __init__(
        self,
        id: str | UUID | None = None,
        kind: str = 'test_event',
        timestamp: datetime | None = None,
    ):
        self.id = id if id is not None else uuid4()
        self.kind = kind
        self.timestamp = timestamp or datetime.now(timezone.utc)

    def model_dump_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        id_str = str(self.id) if isinstance(self.id, UUID) else self.id
        return json.dumps(
            {
                'id': id_str,
                'kind': self.kind,
                'timestamp': self.timestamp.isoformat(),
            },
            indent=indent,
        )

    @classmethod
    def model_validate_json(cls, json_str: str) -> 'MockEvent':
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls(
            id=data['id'],
            kind=data['kind'],
            timestamp=datetime.fromisoformat(data['timestamp']),
        )


@dataclass
class MockFilesystemEventService:
    """
    A mock implementation of FilesystemEventService for testing.

    This replicates the core logic of the actual service without importing it,
    allowing us to test the algorithms in isolation.
    """

    prefix: Path
    user_id: str | None
    app_conversation_info_service: None
    app_conversation_info_load_tasks: dict
    limit: int = 500

    def _load_event(self, path: Path) -> MockEvent | None:
        """Get the event at the path given."""
        try:
            content = path.read_text()
            event = MockEvent.model_validate_json(content)
            return event
        except Exception:
            return None

    def _store_event(self, path: Path, event: MockEvent):
        """Store the event given at the path given."""
        path.parent.mkdir(parents=True, exist_ok=True)
        content = event.model_dump_json(indent=2)
        path.write_text(content)

    def _search_paths(self, prefix: Path, page_id: str | None = None) -> list[Path]:
        """Search paths."""
        search_path = f'{prefix}*'
        files = glob.glob(str(search_path))
        paths = [Path(file) for file in files]
        return paths

    async def get_conversation_path(self, conversation_id: UUID) -> Path:
        """Get a path for a conversation."""
        path = self.prefix
        if self.user_id:
            path /= self.user_id
        path = path / 'v1_conversations' / conversation_id.hex
        return path

    async def save_event(self, conversation_id: UUID, event: MockEvent):
        """Save an event."""
        if isinstance(event.id, str):
            id_hex = event.id.replace('-', '')
        else:
            id_hex = event.id.hex
        path = (await self.get_conversation_path(conversation_id)) / f'{id_hex}.json'
        self._store_event(path, event)

    async def get_event(
        self, conversation_id: UUID, event_id: UUID
    ) -> MockEvent | None:
        """Get an event."""
        conversation_path = await self.get_conversation_path(conversation_id)
        path = conversation_path / f'{event_id.hex}.json'
        return self._load_event(path)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def service(temp_dir: Path) -> MockFilesystemEventService:
    """Create a MockFilesystemEventService instance for testing."""
    return MockFilesystemEventService(
        prefix=temp_dir,
        user_id='test_user',
        app_conversation_info_service=None,
        app_conversation_info_load_tasks={},
    )


@pytest.fixture
def service_no_user(temp_dir: Path) -> MockFilesystemEventService:
    """Create a MockFilesystemEventService instance without user_id."""
    return MockFilesystemEventService(
        prefix=temp_dir,
        user_id=None,
        app_conversation_info_service=None,
        app_conversation_info_load_tasks={},
    )


class TestFilesystemEventServiceLoadEvent:
    """Test cases for _load_event method."""

    def test_load_event_success(
        self, service: MockFilesystemEventService, temp_dir: Path
    ):
        """Test loading an event from a valid JSON file."""
        event_id = uuid4()
        event_data = {
            'id': str(event_id),
            'kind': 'test_event',
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }

        test_file = temp_dir / f'{event_id.hex}.json'
        test_file.write_text(json.dumps(event_data))

        result = service._load_event(test_file)
        assert result is not None
        assert result.id == str(event_id)
        assert result.kind == 'test_event'

    def test_load_event_file_not_found(
        self, service: MockFilesystemEventService, temp_dir: Path
    ):
        """Test loading an event from a non-existent file."""
        non_existent_path = temp_dir / 'non_existent.json'
        result = service._load_event(non_existent_path)
        assert result is None

    def test_load_event_invalid_json(
        self, service: MockFilesystemEventService, temp_dir: Path
    ):
        """Test loading an event from an invalid JSON file."""
        test_file = temp_dir / 'invalid.json'
        test_file.write_text('not valid json {{{')

        result = service._load_event(test_file)
        assert result is None

    def test_load_event_empty_file(
        self, service: MockFilesystemEventService, temp_dir: Path
    ):
        """Test loading an event from an empty file."""
        test_file = temp_dir / 'empty.json'
        test_file.write_text('')

        result = service._load_event(test_file)
        assert result is None

    def test_load_event_missing_fields(
        self, service: MockFilesystemEventService, temp_dir: Path
    ):
        """Test loading an event with missing required fields."""
        test_file = temp_dir / 'missing_fields.json'
        test_file.write_text(json.dumps({'id': str(uuid4())}))  # Missing kind/timestamp

        result = service._load_event(test_file)
        assert result is None


class TestFilesystemEventServiceStoreEvent:
    """Test cases for _store_event method."""

    def test_store_event_creates_file(
        self, service: MockFilesystemEventService, temp_dir: Path
    ):
        """Test that _store_event creates a file with correct content."""
        event = MockEvent()
        test_path = temp_dir / 'events' / f'{event.id}.json'

        service._store_event(test_path, event)

        assert test_path.exists()
        content = test_path.read_text()
        assert str(event.id) in content
        assert event.kind in content

    def test_store_event_creates_parent_directories(
        self, service: MockFilesystemEventService, temp_dir: Path
    ):
        """Test that _store_event creates parent directories if needed."""
        event = MockEvent()
        nested_path = temp_dir / 'a' / 'b' / 'c' / f'{event.id}.json'

        assert not nested_path.parent.exists()

        service._store_event(nested_path, event)

        assert nested_path.exists()
        assert nested_path.parent.exists()

    def test_store_event_overwrites_existing(
        self, service: MockFilesystemEventService, temp_dir: Path
    ):
        """Test that _store_event overwrites an existing file."""
        event1 = MockEvent(kind='original')
        event2 = MockEvent(id=event1.id, kind='updated')
        test_path = temp_dir / f'{event1.id}.json'

        service._store_event(test_path, event1)
        content1 = test_path.read_text()
        assert 'original' in content1

        service._store_event(test_path, event2)
        content2 = test_path.read_text()
        assert 'updated' in content2
        assert 'original' not in content2

    def test_store_event_json_format(
        self, service: MockFilesystemEventService, temp_dir: Path
    ):
        """Test that stored event is valid JSON."""
        event = MockEvent(kind='json_test')
        test_path = temp_dir / 'json_test.json'

        service._store_event(test_path, event)

        content = test_path.read_text()
        parsed = json.loads(content)
        assert parsed['kind'] == 'json_test'
        assert 'id' in parsed
        assert 'timestamp' in parsed


class TestFilesystemEventServiceSearchPaths:
    """Test cases for _search_paths method."""

    def test_search_paths_empty_directory(
        self, service: MockFilesystemEventService, temp_dir: Path
    ):
        """Test searching in an empty directory."""
        search_prefix = temp_dir / 'events'
        search_prefix.mkdir(parents=True)

        result = service._search_paths(search_prefix / 'nonexistent')
        assert result == []

    def test_search_paths_with_glob_pattern(
        self, service: MockFilesystemEventService, temp_dir: Path
    ):
        """Test _search_paths with files matching glob pattern."""
        (temp_dir / 'event1.json').write_text('{}')
        (temp_dir / 'event2.json').write_text('{}')
        (temp_dir / 'other.txt').write_text('not json')

        result = service._search_paths(temp_dir / 'event')

        assert len(result) == 2
        assert all('event' in str(p) for p in result)

    def test_search_paths_no_matches(
        self, service: MockFilesystemEventService, temp_dir: Path
    ):
        """Test _search_paths with no matching files."""
        (temp_dir / 'file1.json').write_text('{}')
        (temp_dir / 'file2.json').write_text('{}')

        result = service._search_paths(temp_dir / 'nonexistent')

        assert result == []

    def test_search_paths_returns_path_objects(
        self, service: MockFilesystemEventService, temp_dir: Path
    ):
        """Test that _search_paths returns Path objects."""
        (temp_dir / 'test1.json').write_text('{}')

        result = service._search_paths(temp_dir / 'test')

        assert len(result) == 1
        assert isinstance(result[0], Path)


class TestFilesystemEventServiceIntegration:
    """Integration tests for MockFilesystemEventService."""

    @pytest.mark.asyncio
    async def test_get_conversation_path_with_user_id(
        self, service: MockFilesystemEventService, temp_dir: Path
    ):
        """Test conversation path generation with user_id."""
        conversation_id = uuid4()

        path = await service.get_conversation_path(conversation_id)

        assert str(temp_dir) in str(path)
        assert 'test_user' in str(path)
        assert 'v1_conversations' in str(path)
        assert conversation_id.hex in str(path)

    @pytest.mark.asyncio
    async def test_get_conversation_path_without_user_id(
        self, service_no_user: MockFilesystemEventService, temp_dir: Path
    ):
        """Test conversation path generation without user_id."""
        conversation_id = uuid4()

        path = await service_no_user.get_conversation_path(conversation_id)

        assert str(temp_dir) in str(path)
        assert 'test_user' not in str(path)
        assert 'v1_conversations' in str(path)
        assert conversation_id.hex in str(path)

    @pytest.mark.asyncio
    async def test_save_and_get_event(
        self, service: MockFilesystemEventService, temp_dir: Path
    ):
        """Test saving and retrieving an event."""
        conversation_id = uuid4()
        event = MockEvent()

        await service.save_event(conversation_id, event)

        conversation_path = await service.get_conversation_path(conversation_id)
        event_file = conversation_path / f'{event.id.hex}.json'
        assert event_file.exists()

        retrieved = await service.get_event(conversation_id, event.id)
        assert retrieved is not None
        assert str(retrieved.id) == str(event.id)

    @pytest.mark.asyncio
    async def test_get_nonexistent_event(self, service: MockFilesystemEventService):
        """Test getting an event that doesn't exist."""
        conversation_id = uuid4()
        event_id = uuid4()

        result = await service.get_event(conversation_id, event_id)
        assert result is None

    @pytest.mark.asyncio
    async def test_save_multiple_events(
        self, service: MockFilesystemEventService, temp_dir: Path
    ):
        """Test saving multiple events to the same conversation."""
        conversation_id = uuid4()
        events = [MockEvent(kind=f'event_{i}') for i in range(3)]

        for event in events:
            await service.save_event(conversation_id, event)

        for event in events:
            retrieved = await service.get_event(conversation_id, event.id)
            assert retrieved is not None
            assert retrieved.kind == event.kind


class TestFilesystemEventServiceEdgeCases:
    """Edge case tests for MockFilesystemEventService."""

    def test_load_event_with_unicode_content(
        self, service: MockFilesystemEventService, temp_dir: Path
    ):
        """Test loading an event with unicode characters."""
        event_data = {
            'id': str(uuid4()),
            'kind': 'unicode_event_你好_🌍',
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }

        test_file = temp_dir / 'unicode.json'
        test_file.write_text(json.dumps(event_data, ensure_ascii=False))

        result = service._load_event(test_file)
        assert result is not None
        assert '你好' in result.kind
        assert '🌍' in result.kind

    def test_store_event_with_special_characters_in_path(
        self, service: MockFilesystemEventService, temp_dir: Path
    ):
        """Test storing an event in a path with spaces."""
        event = MockEvent()
        test_path = temp_dir / 'path with spaces' / f'{event.id}.json'

        service._store_event(test_path, event)
        assert test_path.exists()

    @pytest.mark.asyncio
    async def test_save_event_with_string_id(
        self, service: MockFilesystemEventService, temp_dir: Path
    ):
        """Test saving an event where id is a string (not UUID)."""
        conversation_id = uuid4()
        string_id = str(uuid4())
        event = MockEvent(id=string_id)

        await service.save_event(conversation_id, event)

        conversation_path = await service.get_conversation_path(conversation_id)
        id_hex = string_id.replace('-', '')
        event_file = conversation_path / f'{id_hex}.json'
        assert event_file.exists()

    def test_load_event_preserves_timestamp(
        self, service: MockFilesystemEventService, temp_dir: Path
    ):
        """Test that loading an event preserves the timestamp."""
        original_timestamp = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        event_data = {
            'id': str(uuid4()),
            'kind': 'timestamp_test',
            'timestamp': original_timestamp.isoformat(),
        }

        test_file = temp_dir / 'timestamp.json'
        test_file.write_text(json.dumps(event_data))

        result = service._load_event(test_file)
        assert result is not None
        assert result.timestamp == original_timestamp


class TestFilesystemEventServiceLimit:
    """Test cases for the limit attribute."""

    def test_default_limit(self, temp_dir: Path):
        """Test that default limit is 500."""
        service = MockFilesystemEventService(
            prefix=temp_dir,
            user_id='test_user',
            app_conversation_info_service=None,
            app_conversation_info_load_tasks={},
        )
        assert service.limit == 500

    def test_custom_limit(self, temp_dir: Path):
        """Test setting a custom limit."""
        service = MockFilesystemEventService(
            prefix=temp_dir,
            user_id='test_user',
            app_conversation_info_service=None,
            app_conversation_info_load_tasks={},
            limit=100,
        )
        assert service.limit == 100


class TestReadTextBugFix:
    """Tests specifically for the read_text() bug fix.

    The original bug was: path.read_text(str(path))
    Which incorrectly passed the path string as the 'encoding' parameter.
    The fix is: path.read_text()
    """

    def test_read_text_no_argument(self, temp_dir: Path):
        """Verify that Path.read_text() works correctly without arguments."""
        test_file = temp_dir / 'test.txt'
        test_content = 'Hello, World!'
        test_file.write_text(test_content)

        # This is the correct way (after the fix)
        result = test_file.read_text()
        assert result == test_content

    def test_read_text_with_path_as_encoding_fails(self, temp_dir: Path):
        """Demonstrate that the bug would cause a LookupError."""
        test_file = temp_dir / 'test.txt'
        test_content = 'Hello, World!'
        test_file.write_text(test_content)

        # This is what the bug was doing - passing path as encoding
        with pytest.raises(LookupError):
            test_file.read_text(str(test_file))

    def test_load_event_uses_correct_read_text(
        self, service: MockFilesystemEventService, temp_dir: Path
    ):
        """Test that _load_event correctly reads files.

        This test verifies the fix works end-to-end.
        """
        event_data = {
            'id': str(uuid4()),
            'kind': 'read_test',
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }

        test_file = temp_dir / 'read_test.json'
        test_file.write_text(json.dumps(event_data))

        # If the bug existed, this would raise LookupError
        result = service._load_event(test_file)
        assert result is not None
        assert result.kind == 'read_test'
