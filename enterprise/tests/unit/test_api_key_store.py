import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select
from storage.api_key import ApiKey
from storage.api_key_store import ApiKeyStore


@pytest.fixture
def mock_user():
    """Mock user with org_id."""
    user = MagicMock()
    user.current_org_id = uuid.uuid4()
    return user


@pytest.fixture
def api_key_store():
    return ApiKeyStore()


@pytest.fixture
def mock_litellm_api():
    api_key_patch = patch('storage.lite_llm_manager.LITE_LLM_API_KEY', 'test_key')
    api_url_patch = patch(
        'storage.lite_llm_manager.LITE_LLM_API_URL', 'http://test.url'
    )
    team_id_patch = patch('storage.lite_llm_manager.LITE_LLM_TEAM_ID', 'test_team')
    client_patch = patch('httpx.AsyncClient')

    with api_key_patch, api_url_patch, team_id_patch, client_patch as mock_client:
        mock_response = AsyncMock()
        mock_response.is_success = True
        mock_response.json = MagicMock(return_value={'key': 'test_api_key'})
        mock_client.return_value.__aenter__.return_value.post.return_value = (
            mock_response
        )
        mock_client.return_value.__aenter__.return_value.get.return_value = (
            mock_response
        )
        mock_client.return_value.__aenter__.return_value.patch.return_value = (
            mock_response
        )
        yield mock_client


def test_generate_api_key(api_key_store):
    """Test that generate_api_key returns a string with sk-oh- prefix and expected length."""
    key = api_key_store.generate_api_key(length=32)
    assert isinstance(key, str)
    assert key.startswith('sk-oh-')
    # Total length should be prefix (6 chars) + random part (32 chars) = 38 chars
    assert len(key) == len('sk-oh-') + 32


@pytest.mark.asyncio
@patch('storage.api_key_store.UserStore.get_user_by_id_async')
async def test_create_api_key(
    mock_get_user, api_key_store, async_session_maker, mock_user
):
    """Test creating an API key."""
    # Setup
    user_id = str(uuid.uuid4())
    name = 'Test Key'
    mock_get_user.return_value = mock_user

    # Patch a_session_maker in the api_key_store module to use the test's async session maker
    with patch('storage.api_key_store.a_session_maker', async_session_maker):
        # Execute
        result = await api_key_store.create_api_key(user_id, name)

    # Verify
    assert result.startswith('sk-oh-')
    mock_get_user.assert_called_once_with(user_id)

    # Verify the ApiKey was created in the database using async session
    async with async_session_maker() as session:
        result_db = await session.execute(
            select(ApiKey).filter(ApiKey.user_id == user_id)
        )
        api_key = result_db.scalars().first()
        assert api_key is not None
        assert api_key.name == name
        assert api_key.org_id == mock_user.current_org_id


@pytest.mark.asyncio
async def test_validate_api_key_valid(api_key_store, async_session_maker):
    """Test validating a valid API key."""
    # Setup - create an API key in the database
    user_id = str(uuid.uuid4())
    org_id = uuid.uuid4()
    api_key_value = 'test-api-key'

    async with async_session_maker() as session:
        key_record = ApiKey(
            key=api_key_value,
            user_id=user_id,
            org_id=org_id,
            name='Test Key',
            expires_at=None,
        )
        session.add(key_record)
        await session.commit()

    # Execute - patch a_session_maker to use test's async session maker
    with patch('storage.api_key_store.a_session_maker', async_session_maker):
        result = await api_key_store.validate_api_key(api_key_value)

    # Verify
    assert result == user_id


@pytest.mark.asyncio
async def test_validate_api_key_expired(
    api_key_store, session_maker, async_session_maker
):
    """Test validating an expired API key."""
    # Setup - create an expired API key in the database
    user_id = str(uuid.uuid4())
    org_id = uuid.uuid4()
    api_key_value = 'test-expired-key'

    async with async_session_maker() as session:
        key_record = ApiKey(
            key=api_key_value,
            user_id=user_id,
            org_id=org_id,
            name='Test Key',
            expires_at=datetime.now(UTC) - timedelta(days=1),
        )
        session.add(key_record)
        await session.commit()

    # Execute - patch a_session_maker to use test's async session maker
    with patch('storage.api_key_store.a_session_maker', async_session_maker):
        result = await api_key_store.validate_api_key(api_key_value)

    # Verify
    assert result is None


@pytest.mark.asyncio
async def test_validate_api_key_expired_timezone_naive(
    api_key_store, session_maker, async_session_maker
):
    """Test validating an expired API key with timezone-naive datetime from database."""
    # Setup - create an expired API key with timezone-naive datetime
    user_id = str(uuid.uuid4())
    org_id = uuid.uuid4()
    api_key_value = 'test-expired-naive-key'

    async with async_session_maker() as session:
        key_record = ApiKey(
            key=api_key_value,
            user_id=user_id,
            org_id=org_id,
            name='Test Key',
            # Timezone-naive datetime (database stores this)
            expires_at=datetime.now() - timedelta(days=1),
        )
        session.add(key_record)
        await session.commit()

    # Execute - patch a_session_maker to use test's async session maker
    with patch('storage.api_key_store.a_session_maker', async_session_maker):
        result = await api_key_store.validate_api_key(api_key_value)

    # Verify
    assert result is None


@pytest.mark.asyncio
async def test_validate_api_key_valid_timezone_naive(
    api_key_store, session_maker, async_session_maker
):
    """Test validating a valid API key with timezone-naive datetime from database."""
    # Setup - create a valid API key with timezone-naive datetime (future date)
    user_id = str(uuid.uuid4())
    org_id = uuid.uuid4()
    api_key_value = 'test-valid-naive-key'

    async with async_session_maker() as session:
        key_record = ApiKey(
            key=api_key_value,
            user_id=user_id,
            org_id=org_id,
            name='Test Key',
            # Timezone-naive datetime in the future
            expires_at=datetime.now() + timedelta(days=1),
        )
        session.add(key_record)
        await session.commit()

    # Execute - patch a_session_maker to use test's async session maker
    with patch('storage.api_key_store.a_session_maker', async_session_maker):
        result = await api_key_store.validate_api_key(api_key_value)

    # Verify
    assert result == user_id


@pytest.mark.asyncio
async def test_validate_api_key_not_found(api_key_store, async_session_maker):
    """Test validating a non-existent API key."""
    # Execute
    with patch('storage.api_key_store.a_session_maker', async_session_maker):
        result = await api_key_store.validate_api_key('non-existent-key')

    # Verify
    assert result is None


@pytest.mark.asyncio
async def test_delete_api_key(api_key_store, async_session_maker):
    """Test deleting an API key."""
    # Setup - create an API key in the database
    user_id = str(uuid.uuid4())
    org_id = uuid.uuid4()
    api_key_value = 'test-delete-key'

    async with async_session_maker() as session:
        key_record = ApiKey(
            key=api_key_value,
            user_id=user_id,
            org_id=org_id,
            name='Test Key',
        )
        session.add(key_record)
        await session.commit()

    # Execute - patch a_session_maker to use test's async session maker
    with patch('storage.api_key_store.a_session_maker', async_session_maker):
        result = await api_key_store.delete_api_key(api_key_value)

    # Verify
    assert result is True

    # Verify it was deleted from the database
    async with async_session_maker() as session:
        result_db = await session.execute(
            select(ApiKey).filter(ApiKey.key == api_key_value)
        )
        api_key = result_db.scalars().first()
        assert api_key is None


@pytest.mark.asyncio
async def test_delete_api_key_not_found(api_key_store, async_session_maker):
    """Test deleting a non-existent API key."""
    # Execute
    with patch('storage.api_key_store.a_session_maker', async_session_maker):
        result = await api_key_store.delete_api_key('non-existent-key')

    # Verify
    assert result is False


@pytest.mark.asyncio
async def test_delete_api_key_by_id(api_key_store, async_session_maker):
    """Test deleting an API key by ID."""
    # Setup - create an API key in the database
    user_id = str(uuid.uuid4())
    org_id = uuid.uuid4()

    async with async_session_maker() as session:
        key_record = ApiKey(
            key='test-delete-by-id-key',
            user_id=user_id,
            org_id=org_id,
            name='Test Key',
        )
        session.add(key_record)
        await session.commit()
        key_id = key_record.id

    # Execute - patch a_session_maker to use test's async session maker
    with patch('storage.api_key_store.a_session_maker', async_session_maker):
        result = await api_key_store.delete_api_key_by_id(key_id)

    # Verify
    assert result is True

    # Verify it was deleted from the database
    async with async_session_maker() as session:
        result_db = await session.execute(select(ApiKey).filter(ApiKey.id == key_id))
        api_key = result_db.scalars().first()
        assert api_key is None


@pytest.mark.asyncio
@patch('storage.api_key_store.UserStore.get_user_by_id_async')
async def test_list_api_keys(
    mock_get_user, api_key_store, session_maker, async_session_maker, mock_user
):
    """Test listing API keys for a user."""
    # Setup
    user_id = str(uuid.uuid4())
    mock_get_user.return_value = mock_user
    now = datetime.now(UTC)

    # Create API keys in the database
    async with async_session_maker() as session:
        key1 = ApiKey(
            key='test-key-1',
            user_id=user_id,
            org_id=mock_user.current_org_id,
            name='Key 1',
            created_at=now,
            last_used_at=now,
            expires_at=now + timedelta(days=30),
        )
        key2 = ApiKey(
            key='test-key-2',
            user_id=user_id,
            org_id=mock_user.current_org_id,
            name='Key 2',
            created_at=now,
            last_used_at=None,
            expires_at=None,
        )
        # Add an MCP key that should be filtered out
        mcp_key = ApiKey(
            key='test-mcp-key',
            user_id=user_id,
            org_id=mock_user.current_org_id,
            name='MCP_API_KEY',
            created_at=now,
        )
        session.add_all([key1, key2, mcp_key])
        await session.commit()

    # Execute - patch a_session_maker to use test's async session maker
    with patch('storage.api_key_store.a_session_maker', async_session_maker):
        result = await api_key_store.list_api_keys(user_id)

    # Verify
    mock_get_user.assert_called_once_with(user_id)
    assert len(result) == 2
    assert result[0].name == 'Key 1'
    assert result[1].name == 'Key 2'


@pytest.mark.asyncio
@patch('storage.api_key_store.UserStore.get_user_by_id_async')
async def test_retrieve_mcp_api_key(
    mock_get_user, api_key_store, session_maker, async_session_maker, mock_user
):
    """Test retrieving MCP API key for a user."""
    # Setup
    user_id = str(uuid.uuid4())
    mock_get_user.return_value = mock_user
    now = datetime.now(UTC)

    # Create API keys in the database
    async with async_session_maker() as session:
        other_key = ApiKey(
            key='test-other-key',
            user_id=user_id,
            org_id=mock_user.current_org_id,
            name='Other Key',
            created_at=now,
        )
        mcp_key = ApiKey(
            key='test-mcp-key',
            user_id=user_id,
            org_id=mock_user.current_org_id,
            name='MCP_API_KEY',
            created_at=now,
        )
        session.add_all([other_key, mcp_key])
        await session.commit()

    # Execute - patch a_session_maker to use test's async session maker
    with patch('storage.api_key_store.a_session_maker', async_session_maker):
        result = await api_key_store.retrieve_mcp_api_key(user_id)

    # Verify
    mock_get_user.assert_called_once_with(user_id)
    assert result == 'test-mcp-key'


@pytest.mark.asyncio
@patch('storage.api_key_store.UserStore.get_user_by_id_async')
async def test_retrieve_mcp_api_key_not_found(
    mock_get_user, api_key_store, session_maker, async_session_maker, mock_user
):
    """Test retrieving MCP API key when none exists."""
    # Setup
    user_id = str(uuid.uuid4())
    mock_get_user.return_value = mock_user
    now = datetime.now(UTC)

    # Create only non-MCP keys in the database
    async with async_session_maker() as session:
        other_key = ApiKey(
            key='test-other-key',
            user_id=user_id,
            org_id=mock_user.current_org_id,
            name='Other Key',
            created_at=now,
        )
        session.add(other_key)
        await session.commit()

    # Execute - patch a_session_maker to use test's async session maker
    with patch('storage.api_key_store.a_session_maker', async_session_maker):
        result = await api_key_store.retrieve_mcp_api_key(user_id)

    # Verify
    mock_get_user.assert_called_once_with(user_id)
    assert result is None


@pytest.mark.asyncio
async def test_retrieve_api_key_by_name(
    api_key_store, session_maker, async_session_maker
):
    """Test retrieving an API key by name."""
    # Setup
    user_id = str(uuid.uuid4())
    org_id = uuid.uuid4()
    key_name = 'Test Key'
    key_value = 'test-key-by-name'

    async with async_session_maker() as session:
        key_record = ApiKey(
            key=key_value,
            user_id=user_id,
            org_id=org_id,
            name=key_name,
        )
        session.add(key_record)
        await session.commit()

    # Execute - patch a_session_maker to use test's async session maker
    with patch('storage.api_key_store.a_session_maker', async_session_maker):
        result = await api_key_store.retrieve_api_key_by_name(user_id, key_name)

    # Verify
    assert result == key_value


@pytest.mark.asyncio
async def test_retrieve_api_key_by_name_not_found(api_key_store, async_session_maker):
    """Test retrieving an API key by name that doesn't exist."""
    # Execute
    with patch('storage.api_key_store.a_session_maker', async_session_maker):
        result = await api_key_store.retrieve_api_key_by_name(
            'non-existent-user', 'Non Existent Key'
        )

    # Verify
    assert result is None


@pytest.mark.asyncio
async def test_delete_api_key_by_name(
    api_key_store, session_maker, async_session_maker
):
    """Test deleting an API key by name."""
    # Setup
    user_id = str(uuid.uuid4())
    org_id = uuid.uuid4()
    key_name = 'Test Key to Delete'
    key_value = 'test-delete-by-name'

    async with async_session_maker() as session:
        key_record = ApiKey(
            key=key_value,
            user_id=user_id,
            org_id=org_id,
            name=key_name,
        )
        session.add(key_record)
        await session.commit()

    # Execute - patch a_session_maker to use test's async session maker
    with patch('storage.api_key_store.a_session_maker', async_session_maker):
        result = await api_key_store.delete_api_key_by_name(user_id, key_name)

    # Verify
    assert result is True

    # Verify it was deleted from the database
    async with async_session_maker() as session:
        result_db = await session.execute(
            select(ApiKey).filter(ApiKey.key == key_value)
        )
        api_key = result_db.scalars().first()
        assert api_key is None


@pytest.mark.asyncio
async def test_delete_api_key_by_name_not_found(api_key_store, async_session_maker):
    """Test deleting an API key by name that doesn't exist."""
    # Execute
    with patch('storage.api_key_store.a_session_maker', async_session_maker):
        result = await api_key_store.delete_api_key_by_name(
            'non-existent-user', 'Non Existent Key'
        )

    # Verify
    assert result is False
