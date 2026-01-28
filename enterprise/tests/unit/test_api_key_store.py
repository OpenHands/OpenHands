from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

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
def mock_user():
    """Mock user with org_id."""
    user = MagicMock()
    user.current_org_id = 'test-org-123'
    return user


@pytest.fixture
def api_key_store(mock_session_maker):
    return ApiKeyStore(mock_session_maker)


def run_sync(func, *args, **kwargs):
    """Helper to execute sync functions directly (mocks call_sync_from_async)."""
    return func(*args, **kwargs)


def test_generate_api_key(api_key_store):
    """Test that generate_api_key returns a string with sk-oh- prefix and expected length."""
    key = api_key_store.generate_api_key(length=32)
    assert isinstance(key, str)
    assert key.startswith('sk-oh-')
    # Total length should be prefix (6 chars) + random part (32 chars) = 38 chars
    assert len(key) == len('sk-oh-') + 32


@pytest.mark.asyncio
@patch('storage.api_key_store.call_sync_from_async', side_effect=run_sync)
@patch('storage.api_key_store.UserStore.get_user_by_id_async')
async def test_create_api_key(
    mock_get_user, mock_call_sync, api_key_store, mock_session, mock_user
):
    """Test creating an API key."""
    # Setup
    user_id = 'test-user-123'
    name = 'Test Key'
    mock_get_user.return_value = mock_user
    api_key_store.generate_api_key = MagicMock(return_value='test-api-key')

    # Execute
    result = await api_key_store.create_api_key(user_id, name)

    # Verify
    assert result == 'test-api-key'
    mock_get_user.assert_called_once_with(user_id)
    mock_session.add.assert_called_once()
    mock_session.commit.assert_called_once()
    api_key_store.generate_api_key.assert_called_once()

    # Verify the ApiKey was created with the correct org_id
    added_api_key = mock_session.add.call_args[0][0]
    assert added_api_key.org_id == mock_user.current_org_id


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


@pytest.mark.asyncio
@patch('storage.api_key_store.call_sync_from_async', side_effect=run_sync)
@patch('storage.api_key_store.UserStore.get_user_by_id_async')
async def test_list_api_keys(
    mock_get_user, mock_call_sync, api_key_store, mock_session, mock_user
):
    """Test listing API keys for a user."""
    # Setup
    user_id = 'test-user-123'
    mock_get_user.return_value = mock_user
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

    # Mock the chained query calls for filtering by user_id and org_id
    mock_query = mock_session.query.return_value
    mock_filter_user = mock_query.filter.return_value
    mock_filter_org = mock_filter_user.filter.return_value
    mock_filter_org.all.return_value = [mock_key1, mock_key2]

    # Execute
    result = await api_key_store.list_api_keys(user_id)

    # Verify
    mock_get_user.assert_called_once_with(user_id)
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


@pytest.mark.asyncio
@patch('storage.api_key_store.call_sync_from_async', side_effect=run_sync)
@patch('storage.api_key_store.UserStore.get_user_by_id_async')
async def test_retrieve_mcp_api_key(
    mock_get_user, mock_call_sync, api_key_store, mock_session, mock_user
):
    """Test retrieving MCP API key for a user."""
    # Setup
    user_id = 'test-user-123'
    mock_get_user.return_value = mock_user

    mock_mcp_key = MagicMock()
    mock_mcp_key.name = 'MCP_API_KEY'
    mock_mcp_key.key = 'mcp-test-key'

    mock_other_key = MagicMock()
    mock_other_key.name = 'Other Key'
    mock_other_key.key = 'other-test-key'

    # Mock the chained query calls for filtering by user_id and org_id
    mock_query = mock_session.query.return_value
    mock_filter_user = mock_query.filter.return_value
    mock_filter_org = mock_filter_user.filter.return_value
    mock_filter_org.all.return_value = [mock_other_key, mock_mcp_key]

    # Execute
    result = await api_key_store.retrieve_mcp_api_key(user_id)

    # Verify
    mock_get_user.assert_called_once_with(user_id)
    assert result == 'mcp-test-key'


@pytest.mark.asyncio
@patch('storage.api_key_store.call_sync_from_async', side_effect=run_sync)
@patch('storage.api_key_store.UserStore.get_user_by_id_async')
async def test_retrieve_mcp_api_key_not_found(
    mock_get_user, mock_call_sync, api_key_store, mock_session, mock_user
):
    """Test retrieving MCP API key when none exists."""
    # Setup
    user_id = 'test-user-123'
    mock_get_user.return_value = mock_user

    mock_other_key = MagicMock()
    mock_other_key.name = 'Other Key'
    mock_other_key.key = 'other-test-key'

    # Mock the chained query calls for filtering by user_id and org_id
    mock_query = mock_session.query.return_value
    mock_filter_user = mock_query.filter.return_value
    mock_filter_org = mock_filter_user.filter.return_value
    mock_filter_org.all.return_value = [mock_other_key]

    # Execute
    result = await api_key_store.retrieve_mcp_api_key(user_id)

    # Verify
    mock_get_user.assert_called_once_with(user_id)
    assert result is None
