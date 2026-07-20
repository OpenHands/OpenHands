import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import or_, select, update
from storage.api_key import ApiKey
from storage.api_key_store import ApiKeyStore, ApiKeyValidationResult, _as_naive


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
@patch('storage.api_key_store.UserStore.get_user_by_id')
async def test_create_api_key_strips_timezone_from_expires_at(
    mock_get_user, api_key_store, async_session_maker, mock_user
):
    """Timezone-aware expires_at must be stored as naive UTC without shifting the value."""
    user_id = str(uuid.uuid4())
    aware_expiry = datetime.now(UTC) + timedelta(days=30)
    mock_get_user.return_value = mock_user

    with patch('storage.api_key_store.a_session_maker', async_session_maker):
        key = await api_key_store.create_api_key(user_id, expires_at=aware_expiry)

    async with async_session_maker() as session:
        result = await session.execute(select(ApiKey).filter(ApiKey.key == key))
        record = result.scalars().first()

    assert record.expires_at is not None
    assert record.expires_at.tzinfo is None
    assert record.expires_at == aware_expiry.replace(tzinfo=None)


@pytest.mark.asyncio
@patch('storage.api_key_store.UserStore.get_user_by_id')
async def test_create_api_key_strips_timezone_from_not_before(
    mock_get_user, api_key_store, async_session_maker, mock_user
):
    """Timezone-aware not_before must be stored as naive UTC without shifting the value."""
    user_id = str(uuid.uuid4())
    aware_not_before = datetime.now(UTC) + timedelta(days=7)
    mock_get_user.return_value = mock_user

    with patch('storage.api_key_store.a_session_maker', async_session_maker):
        key = await api_key_store.create_api_key(
            user_id, name='nb-key', not_before=aware_not_before
        )

    async with async_session_maker() as session:
        result = await session.execute(select(ApiKey).filter(ApiKey.key == key))
        record = result.scalars().first()

    assert record.not_before is not None
    assert record.not_before.tzinfo is None
    assert record.not_before == aware_not_before.replace(tzinfo=None)
    # expires_at should still be None when only not_before is provided
    assert record.expires_at is None


@pytest.mark.asyncio
@patch('storage.api_key_store.UserStore.get_user_by_id')
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
@patch('storage.api_key_store.UserStore.get_user_by_id')
async def test_create_api_key_unbound_with_fallback_disabled(
    mock_get_user, api_key_store, async_session_maker, mock_user
):
    """Explicit ``org_id=None`` is stored verbatim when fallback is disabled.

    Regression: previously, ``create_api_key(user, name, org_id=None)``
    silently rebound to ``user.current_org_id``, so the "All orgs" / unbound
    key flow in ``POST /api/keys`` returned 500 -- the route inserted
    with ``None`` matching nothing.
    """
    user_id = str(uuid.uuid4())
    name = 'Unbound Key'
    # If the fallback were still active, the store would look up the user
    # and rebind ``org_id`` to ``mock_user.current_org_id``.
    mock_get_user.return_value = mock_user

    with patch('storage.api_key_store.a_session_maker', async_session_maker):
        result = await api_key_store.create_api_key(
            user_id,
            name,
            org_id=None,
            use_current_org_fallback=False,
        )

    assert result.startswith('sk-oh-')
    # The user must NOT be looked up when the caller has already decided
    # the binding -- that's the whole point of opting out of the fallback.
    mock_get_user.assert_not_called()

    async with async_session_maker() as session:
        result_db = await session.execute(
            select(ApiKey).filter(ApiKey.user_id == user_id)
        )
        api_key = result_db.scalars().first()
        assert api_key is not None
        assert api_key.name == name
        assert api_key.org_id is None  # stored verbatim


@pytest.mark.asyncio
@patch('storage.api_key_store.UserStore.get_user_by_id')
async def test_create_api_key_default_fallback_still_works(
    mock_get_user, api_key_store, async_session_maker, mock_user
):
    """Default (fallback) behavior is preserved for existing internal callers."""
    user_id = str(uuid.uuid4())
    name = 'Legacy-style Key'
    mock_get_user.return_value = mock_user

    with patch('storage.api_key_store.a_session_maker', async_session_maker):
        # No ``use_current_org_fallback`` argument -- the default kicks in.
        await api_key_store.create_api_key(user_id, name, org_id=None)

    mock_get_user.assert_called_once_with(user_id)
    async with async_session_maker() as session:
        result_db = await session.execute(
            select(ApiKey).filter(ApiKey.user_id == user_id)
        )
        api_key = result_db.scalars().first()
        assert api_key is not None
        assert api_key.org_id == mock_user.current_org_id


@pytest.mark.asyncio
async def test_validate_api_key_valid(api_key_store, async_session_maker):
    """Test validating a valid API key returns user_id and org_id."""
    # Arrange
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
        key_id = key_record.id

    # Act
    with patch('storage.api_key_store.a_session_maker', async_session_maker):
        result = await api_key_store.validate_api_key(api_key_value)

    # Assert
    assert isinstance(result, ApiKeyValidationResult)
    assert result is not None
    assert result.user_id == user_id
    assert result.org_id == org_id
    assert result.key_id == key_id
    assert result.key_name == 'Test Key'


@pytest.mark.asyncio
async def test_validate_api_key_expired(api_key_store, async_session_maker):
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
    api_key_store, async_session_maker
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
    api_key_store, async_session_maker
):
    """Test validating a valid API key with timezone-naive datetime from database."""
    # Arrange
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

    # Act
    with patch('storage.api_key_store.a_session_maker', async_session_maker):
        result = await api_key_store.validate_api_key(api_key_value)

    # Assert
    assert isinstance(result, ApiKeyValidationResult)
    assert result.user_id == user_id
    assert result.org_id == org_id


@pytest.mark.asyncio
async def test_validate_api_key_unbound_returns_none_org_id(
    api_key_store, async_session_maker
):
    """An unbound API key validates successfully and reports ``org_id=None``.

    The effective org for an unbound key is resolved per-request by
    ``SaasUserAuth`` (from ``X-Org-Id`` or ``user.current_org_id``); the
    store layer only reflects the persisted binding.
    """
    # Arrange
    user_id = str(uuid.uuid4())
    api_key_value = 'test-unbound-key-no-org'

    async with async_session_maker() as session:
        key_record = ApiKey(
            key=api_key_value,
            user_id=user_id,
            org_id=None,  # Unbound key
            name='Multi-org Key',
        )
        session.add(key_record)
        await session.commit()

    # Act
    with patch('storage.api_key_store.a_session_maker', async_session_maker):
        result = await api_key_store.validate_api_key(api_key_value)

    # Assert
    assert isinstance(result, ApiKeyValidationResult)
    assert result is not None
    assert result.user_id == user_id
    assert result.org_id is None


@pytest.mark.asyncio
async def test_validate_api_key_not_yet_active(api_key_store, async_session_maker):
    """A key whose not_before is in the future must be rejected."""
    user_id = str(uuid.uuid4())
    org_id = uuid.uuid4()
    api_key_value = 'test-not-yet-active-key'

    async with async_session_maker() as session:
        key_record = ApiKey(
            key=api_key_value,
            user_id=user_id,
            org_id=org_id,
            name='Scheduled Key',
            not_before=datetime.now(UTC) + timedelta(days=1),
        )
        session.add(key_record)
        await session.commit()
        key_id = key_record.id

    with patch('storage.api_key_store.a_session_maker', async_session_maker):
        result = await api_key_store.validate_api_key(api_key_value)

    assert result is None

    # Rejected keys must NOT have last_used_at updated.
    async with async_session_maker() as session:
        stored = await session.execute(select(ApiKey).filter(ApiKey.id == key_id))
        assert stored.scalars().first().last_used_at is None


@pytest.mark.asyncio
async def test_validate_api_key_not_yet_active_timezone_naive(
    api_key_store, async_session_maker
):
    """A naive-UTC not_before in the future must also be rejected."""
    user_id = str(uuid.uuid4())
    org_id = uuid.uuid4()
    api_key_value = 'test-not-yet-active-naive'

    async with async_session_maker() as session:
        key_record = ApiKey(
            key=api_key_value,
            user_id=user_id,
            org_id=org_id,
            name='Scheduled Key Naive',
            not_before=datetime.now() + timedelta(days=1),
        )
        session.add(key_record)
        await session.commit()

    with patch('storage.api_key_store.a_session_maker', async_session_maker):
        result = await api_key_store.validate_api_key(api_key_value)

    assert result is None


@pytest.mark.asyncio
async def test_validate_api_key_active_window_inside(
    api_key_store, async_session_maker
):
    """A key with both bounds set is accepted when now is inside the window."""
    user_id = str(uuid.uuid4())
    org_id = uuid.uuid4()
    api_key_value = 'test-active-window-inside'

    async with async_session_maker() as session:
        key_record = ApiKey(
            key=api_key_value,
            user_id=user_id,
            org_id=org_id,
            name='Window Key',
            not_before=datetime.now(UTC) - timedelta(hours=1),
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )
        session.add(key_record)
        await session.commit()

    with patch('storage.api_key_store.a_session_maker', async_session_maker):
        result = await api_key_store.validate_api_key(api_key_value)

    assert isinstance(result, ApiKeyValidationResult)
    assert result.user_id == user_id
    assert result.org_id == org_id


@pytest.mark.asyncio
async def test_validate_api_key_only_not_before_past(
    api_key_store, async_session_maker
):
    """A key with only not_before in the past and no expires_at is accepted."""
    user_id = str(uuid.uuid4())
    org_id = uuid.uuid4()
    api_key_value = 'test-only-not-before'

    async with async_session_maker() as session:
        key_record = ApiKey(
            key=api_key_value,
            user_id=user_id,
            org_id=org_id,
            name='No Expiry Key',
            not_before=datetime.now(UTC) - timedelta(minutes=1),
            expires_at=None,
        )
        session.add(key_record)
        await session.commit()

    with patch('storage.api_key_store.a_session_maker', async_session_maker):
        result = await api_key_store.validate_api_key(api_key_value)

    assert isinstance(result, ApiKeyValidationResult)
    assert result.user_id == user_id


@pytest.mark.asyncio
async def test_validate_api_key_not_found(api_key_store, async_session_maker):
    """Test validating a non-existent API key."""
    # Execute
    with patch('storage.api_key_store.a_session_maker', async_session_maker):
        result = await api_key_store.validate_api_key('non-existent-key')

    # Verify
    assert result is None


@pytest.mark.asyncio
async def test_validate_api_key_stores_timezone_naive_last_used_at(
    api_key_store, async_session_maker
):
    """Test that validate_api_key stores a timezone-naive datetime for last_used_at."""
    # Arrange
    user_id = str(uuid.uuid4())
    org_id = uuid.uuid4()
    api_key_value = 'test-timezone-naive-key'

    async with async_session_maker() as session:
        key_record = ApiKey(
            key=api_key_value,
            user_id=user_id,
            org_id=org_id,
            name='Test Key',
            last_used_at=None,
        )
        session.add(key_record)
        await session.commit()

    # Act
    with patch('storage.api_key_store.a_session_maker', async_session_maker):
        await api_key_store.validate_api_key(api_key_value)

    # Assert
    async with async_session_maker() as session:
        result_db = await session.execute(
            select(ApiKey).filter(ApiKey.key == api_key_value)
        )
        api_key = result_db.scalars().first()
        assert api_key.last_used_at is not None
        assert api_key.last_used_at.tzinfo is None


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


# ---------------------------------------------------------------------------
# Tests for validate_api_key debounce + optimistic CAS on last_used_at
# ---------------------------------------------------------------------------


async def _fetch_last_used(async_session_maker, api_key_value: str):
    async with async_session_maker() as session:
        result = await session.execute(
            select(ApiKey).filter(ApiKey.key == api_key_value)
        )
        return result.scalars().first().last_used_at


@pytest.mark.asyncio
async def test_validate_api_key_writes_last_used_at_when_null(
    api_key_store, async_session_maker
):
    """First-ever use (last_used_at IS NULL) must populate the timestamp."""
    user_id = str(uuid.uuid4())
    org_id = uuid.uuid4()
    api_key_value = 'test-first-use-key'

    async with async_session_maker() as session:
        session.add(
            ApiKey(
                key=api_key_value,
                user_id=user_id,
                org_id=org_id,
                name='First Use',
                last_used_at=None,
            )
        )
        await session.commit()

    with patch('storage.api_key_store.a_session_maker', async_session_maker):
        result = await api_key_store.validate_api_key(api_key_value)

    assert isinstance(result, ApiKeyValidationResult)
    stored = await _fetch_last_used(async_session_maker, api_key_value)
    assert stored is not None
    # Column is TIMESTAMP WITHOUT TIME ZONE; the writer must strip tzinfo.
    assert stored.tzinfo is None


@pytest.mark.asyncio
async def test_validate_api_key_skips_update_within_debounce_window(
    api_key_store, async_session_maker
):
    """A key used within the last 5s must NOT have last_used_at rewritten."""
    user_id = str(uuid.uuid4())
    org_id = uuid.uuid4()
    api_key_value = 'test-debounce-skip-key'
    original = (datetime.now(UTC) - timedelta(seconds=2)).replace(tzinfo=None)

    async with async_session_maker() as session:
        session.add(
            ApiKey(
                key=api_key_value,
                user_id=user_id,
                org_id=org_id,
                name='Debounce Skip',
                last_used_at=original,
            )
        )
        await session.commit()

    with patch('storage.api_key_store.a_session_maker', async_session_maker):
        result = await api_key_store.validate_api_key(api_key_value)

    assert isinstance(result, ApiKeyValidationResult)
    stored = await _fetch_last_used(async_session_maker, api_key_value)
    assert stored == original, 'last_used_at should not move within the debounce window'


@pytest.mark.asyncio
async def test_validate_api_key_writes_last_used_at_outside_debounce_window(
    api_key_store, async_session_maker
):
    """A key last used > 5s ago must have last_used_at advanced."""
    user_id = str(uuid.uuid4())
    org_id = uuid.uuid4()
    api_key_value = 'test-debounce-write-key'
    original = (datetime.now(UTC) - timedelta(seconds=30)).replace(tzinfo=None)

    async with async_session_maker() as session:
        session.add(
            ApiKey(
                key=api_key_value,
                user_id=user_id,
                org_id=org_id,
                name='Debounce Write',
                last_used_at=original,
            )
        )
        await session.commit()

    with patch('storage.api_key_store.a_session_maker', async_session_maker):
        result = await api_key_store.validate_api_key(api_key_value)

    assert isinstance(result, ApiKeyValidationResult)
    stored = await _fetch_last_used(async_session_maker, api_key_value)
    assert stored is not None
    assert stored != original
    assert stored > original


@pytest.mark.asyncio
async def test_validate_api_key_zero_debounce_writes_every_time(
    api_key_store, async_session_maker
):
    """Disabling the debounce (LAST_USED_DEBOUNCE_SECONDS=0) must write every call."""
    user_id = str(uuid.uuid4())
    org_id = uuid.uuid4()
    api_key_value = 'test-debounce-disabled-key'
    # Even a "very recent" timestamp should be overwritten when debounce is off.
    original = datetime.now(UTC).replace(tzinfo=None)

    async with async_session_maker() as session:
        session.add(
            ApiKey(
                key=api_key_value,
                user_id=user_id,
                org_id=org_id,
                name='Debounce Off',
                last_used_at=original,
            )
        )
        await session.commit()

    with (
        patch.object(ApiKeyStore, 'LAST_USED_DEBOUNCE_SECONDS', 0),
        patch('storage.api_key_store.a_session_maker', async_session_maker),
    ):
        result = await api_key_store.validate_api_key(api_key_value)

    assert isinstance(result, ApiKeyValidationResult)
    stored = await _fetch_last_used(async_session_maker, api_key_value)
    assert stored is not None
    assert stored != original


@pytest.mark.asyncio
async def test_validate_api_key_optimistic_cas_does_not_overwrite_concurrent_update(
    api_key_store, async_session_maker
):
    """Optimistic CAS: if another writer already advanced last_used_at between
    our SELECT and our UPDATE, the conditional UPDATE must affect 0 rows.

    This mirrors the production behaviour under PostgreSQL READ COMMITTED:
    the WHERE clause is re-evaluated against the latest committed version of
    the row, and a stale ``last_used_at`` value no longer matches.

    The test is constructed so the *debounce* clause alone would still allow
    the UPDATE (both ``stale`` and ``fresh`` are well outside the debounce
    window). The only thing that prevents the write is the optimistic
    ``last_used_at == stale`` clause.
    """
    user_id = str(uuid.uuid4())
    org_id = uuid.uuid4()
    api_key_value = 'test-optimistic-cas-key'
    # Both values are well outside the debounce window (5s) so the debounce
    # clause cannot be what blocks the UPDATE.
    stale = (datetime.now(UTC) - timedelta(seconds=60)).replace(tzinfo=None)

    async with async_session_maker() as session:
        session.add(
            ApiKey(
                key=api_key_value,
                user_id=user_id,
                org_id=org_id,
                name='Optimistic CAS',
                last_used_at=stale,
            )
        )
        await session.commit()
        key_id = (
            (await session.execute(select(ApiKey).filter(ApiKey.key == api_key_value)))
            .scalars()
            .first()
            .id
        )

    # Simulate the race: another transaction advanced last_used_at after our
    # SELECT but before our UPDATE. The new value is still well outside the
    # debounce window, so only the optimistic CAS clause can stop the write.
    fresh = (datetime.now(UTC) - timedelta(seconds=45)).replace(tzinfo=None)
    assert fresh > stale, 'fresh must be later than stale for the CAS to matter'
    async with async_session_maker() as session:
        await session.execute(
            update(ApiKey).where(ApiKey.id == key_id).values(last_used_at=fresh)
        )
        await session.commit()

    # Replay the exact conditional UPDATE that validate_api_key would issue,
    # using the stale value our SELECT would have returned.
    debounce_cutoff = _as_naive(
        datetime.now(UTC) - timedelta(seconds=ApiKeyStore.LAST_USED_DEBOUNCE_SECONDS)
    )
    async with async_session_maker() as session:
        result = await session.execute(
            update(ApiKey)
            .where(
                ApiKey.id == key_id,
                ApiKey.last_used_at == stale,  # the value our SELECT returned
                or_(
                    ApiKey.last_used_at.is_(None),
                    ApiKey.last_used_at <= debounce_cutoff,
                ),
            )
            .values(last_used_at=_as_naive(datetime.now(UTC)))
        )
        await session.commit()

    assert result.rowcount == 0, (
        'Optimistic CAS must not overwrite a row whose last_used_at has '
        'already advanced; rowcount should be 0'
    )
    stored = await _fetch_last_used(async_session_maker, api_key_value)
    assert stored == fresh, 'concurrent writer value must be preserved'


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
@patch('storage.api_key_store.UserStore.get_user_by_id')
async def test_list_api_keys(
    mock_get_user, api_key_store, async_session_maker, mock_user
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
@patch('storage.api_key_store.UserStore.get_user_by_id')
async def test_retrieve_mcp_api_key(
    mock_get_user, api_key_store, async_session_maker, mock_user
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
@patch('storage.api_key_store.UserStore.get_user_by_id')
async def test_retrieve_mcp_api_key_not_found(
    mock_get_user, api_key_store, async_session_maker, mock_user
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
async def test_retrieve_api_key_by_name(api_key_store, async_session_maker):
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
async def test_delete_api_key_by_name(api_key_store, async_session_maker):
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
