from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest
from storage.api_key_store import ApiKeyStore


@pytest.fixture
def mock_session():
    session = MagicMock()
    return session


@pytest.fixture
def mock_session_maker(mock_session):
    session_maker = MagicMock()
    session_maker.return_value.__enter__.return_value = mock_session
    session_maker.return_value.__exit__.return_value = None
    return session_maker


@pytest.fixture
def api_key_store(mock_session_maker):
    return ApiKeyStore(mock_session_maker)


def test_generate_api_key(api_key_store):
    """Test that generate_api_key returns a string with sk-oh- prefix and expected length."""
    key = api_key_store.generate_api_key(length=32)
    assert isinstance(key, str)
    assert key.startswith('sk-oh-')
    # Total length should be prefix (6 chars) + random part (32 chars) = 38 chars
    assert len(key) == len('sk-oh-') + 32


def test_create_api_key(api_key_store, mock_session):
    """Test creating an API key."""
    # Setup
    user_id = 'test-user-123'
    name = 'Test Key'
    api_key_store.generate_api_key = MagicMock(return_value='test-api-key')

    # Execute
    result = api_key_store.create_api_key(user_id, name)

    # Verify
    assert result == 'test-api-key'
    mock_session.add.assert_called_once()
    mock_session.commit.assert_called_once()
    api_key_store.generate_api_key.assert_called_once()


def test_validate_api_key_valid(api_key_store, mock_session):
    """Test validating a valid API key."""
    # Setup
    api_key = 'test-api-key'
    user_id = 'test-user-123'
    mock_key_record = MagicMock()
    mock_key_record.user_id = user_id
    mock_key_record.expires_at = None
    mock_key_record.id = 1
    mock_session.query.return_value.filter.return_value.first.return_value = (
        mock_key_record
    )

    # Execute
    result = api_key_store.validate_api_key(api_key)

    # Verify
    assert result == user_id
    mock_session.execute.assert_called_once()
    mock_session.commit.assert_called_once()


def test_validate_api_key_expired(api_key_store, mock_session):
    """Test validating an expired API key."""
    # Setup
    api_key = 'test-api-key'
    mock_key_record = MagicMock()
    mock_key_record.expires_at = datetime.now(UTC) - timedelta(days=1)
    mock_key_record.id = 1
    mock_session.query.return_value.filter.return_value.first.return_value = (
        mock_key_record
    )

    # Execute
    result = api_key_store.validate_api_key(api_key)

    # Verify
    assert result is None
    mock_session.execute.assert_not_called()
    mock_session.commit.assert_not_called()


def test_validate_api_key_expired_timezone_naive(api_key_store, mock_session):
    """Test validating an expired API key with timezone-naive datetime from database."""
    # Setup
    api_key = 'test-api-key'
    mock_key_record = MagicMock()
    # Simulate timezone-naive datetime as returned from database
    mock_key_record.expires_at = datetime.now() - timedelta(days=1)  # No UTC timezone
    mock_key_record.id = 1
    mock_session.query.return_value.filter.return_value.first.return_value = (
        mock_key_record
    )

    # Execute
    result = api_key_store.validate_api_key(api_key)

    # Verify
    assert result is None
    mock_session.execute.assert_not_called()
    mock_session.commit.assert_not_called()


def test_validate_api_key_valid_timezone_naive(api_key_store, mock_session):
    """Test validating a valid API key with timezone-naive datetime from database."""
    # Setup
    api_key = 'test-api-key'
    user_id = 'test-user-123'
    mock_key_record = MagicMock()
    mock_key_record.user_id = user_id
    # Simulate timezone-naive datetime as returned from database (future date)
    mock_key_record.expires_at = datetime.now() + timedelta(days=1)  # No UTC timezone
    mock_key_record.id = 1
    mock_session.query.return_value.filter.return_value.first.return_value = (
        mock_key_record
    )

    # Execute
    result = api_key_store.validate_api_key(api_key)

    # Verify
    assert result == user_id
    mock_session.execute.assert_called_once()
    mock_session.commit.assert_called_once()


def test_validate_api_key_not_found(api_key_store, mock_session):
    """Test validating a non-existent API key."""
    # Setup
    api_key = 'test-api-key'
    query_result = mock_session.query.return_value.filter.return_value
    query_result.first.return_value = None

    # Execute
    result = api_key_store.validate_api_key(api_key)

    # Verify
    assert result is None
    mock_session.execute.assert_not_called()
    mock_session.commit.assert_not_called()


def test_delete_api_key(api_key_store, mock_session):
    """Test deleting an API key."""
    # Setup
    api_key = 'test-api-key'
    mock_key_record = MagicMock()
    mock_session.query.return_value.filter.return_value.first.return_value = (
        mock_key_record
    )

    # Execute
    result = api_key_store.delete_api_key(api_key)

    # Verify
    assert result is True
    mock_session.delete.assert_called_once_with(mock_key_record)
    mock_session.commit.assert_called_once()


def test_delete_api_key_not_found(api_key_store, mock_session):
    """Test deleting a non-existent API key."""
    # Setup
    api_key = 'test-api-key'
    query_result = mock_session.query.return_value.filter.return_value
    query_result.first.return_value = None

    # Execute
    result = api_key_store.delete_api_key(api_key)

    # Verify
    assert result is False
    mock_session.delete.assert_not_called()
    mock_session.commit.assert_not_called()


def test_delete_api_key_by_id(api_key_store, mock_session):
    """Test deleting an API key by ID."""
    # Setup
    key_id = 123
    mock_key_record = MagicMock()
    mock_session.query.return_value.filter.return_value.first.return_value = (
        mock_key_record
    )

    # Execute
    result = api_key_store.delete_api_key_by_id(key_id)

    # Verify
    assert result is True
    mock_session.delete.assert_called_once_with(mock_key_record)
    mock_session.commit.assert_called_once()


def test_list_api_keys(api_key_store, mock_session):
    """Test listing API keys for a user."""
    # Setup
    user_id = 'test-user-123'
    now = datetime.now(UTC)
    mock_key1 = MagicMock()
    mock_key1.id = 1
    mock_key1.name = 'Key 1'
    mock_key1.created_at = now
    mock_key1.last_used_at = now
    mock_key1.expires_at = now + timedelta(days=30)

    mock_key2 = MagicMock()
    mock_key2.id = 2
    mock_key2.name = 'Key 2'
    mock_key2.created_at = now
    mock_key2.last_used_at = None
    mock_key2.expires_at = None

    mock_session.query.return_value.filter.return_value.all.return_value = [
        mock_key1,
        mock_key2,
    ]

    # Execute
    result = api_key_store.list_api_keys(user_id)

    # Verify
    assert len(result) == 2
    assert result[0]['id'] == 1
    assert result[0]['name'] == 'Key 1'
    assert result[0]['created_at'] == now
    assert result[0]['last_used_at'] == now
    assert result[0]['expires_at'] == now + timedelta(days=30)

    assert result[1]['id'] == 2
    assert result[1]['name'] == 'Key 2'
    assert result[1]['created_at'] == now
    assert result[1]['last_used_at'] is None
    assert result[1]['expires_at'] is None
