from unittest.mock import MagicMock, patch

import pytest
from server.auth.token_manager import TokenManager
from storage.offline_token_store import OfflineTokenStore
from storage.stored_offline_token import StoredOfflineToken

from openhands.core.config.openhands_config import OpenHandsConfig


@pytest.fixture
def mock_config():
    return MagicMock(spec=OpenHandsConfig)


@pytest.fixture
def token_store(session_maker, mock_config):
    return OfflineTokenStore('test_user_id', session_maker, mock_config)


@pytest.fixture
def token_manager():
    with patch('server.config.get_config') as mock_get_config:
        mock_config = mock_get_config.return_value
        mock_config.jwt_secret.get_secret_value.return_value = 'test_secret'
        return TokenManager(external=False)


@pytest.mark.asyncio
async def test_store_token_new_record(token_store, session_maker):
    # Setup
    test_token = 'test_offline_token'

    # Execute
    await token_store.store_token(test_token)

    # Verify
    with session_maker() as session:
        query = session.query(StoredOfflineToken)
        assert query.count() == 1
        added_record = query.first()
        assert added_record.user_id == 'test_user_id'
    assert added_record.offline_token == test_token


@pytest.mark.asyncio
async def test_store_token_existing_record(token_store, session_maker):
    # Setup
    with session_maker() as session:
        session.add(
            StoredOfflineToken(user_id='test_user_id', offline_token='old_token')
        )
        session.commit()

    test_token = 'new_offline_token'

    # Execute
    await token_store.store_token(test_token)

    # Verify
    with session_maker() as session:
        query = session.query(StoredOfflineToken)
        assert query.count() == 1
        added_record = query.first()
        assert added_record.user_id == 'test_user_id'
        assert added_record.offline_token == test_token


@pytest.mark.asyncio
async def test_load_token_existing(token_store, session_maker):
    # Setup
    with session_maker() as session:
        session.add(
            StoredOfflineToken(
                user_id='test_user_id', offline_token='test_offline_token'
            )
        )
        session.commit()

    # Execute
    result = await token_store.load_token()

    # Verify
    assert result == 'test_offline_token'


@pytest.mark.asyncio
async def test_load_token_not_found(token_store):
    # Execute
    result = await token_store.load_token()

    # Verify
    assert result is None


@pytest.mark.asyncio
async def test_get_instance(mock_config):
    # Setup
    test_user_id = 'test_user_id'

    # Execute
    result = await OfflineTokenStore.get_instance(mock_config, test_user_id)

    # Verify
    assert isinstance(result, OfflineTokenStore)
    assert result.user_id == test_user_id
    assert result.config == mock_config


def test_load_store_org_token(token_manager, session_maker):
    with patch('server.auth.token_manager.session_maker', session_maker):
        token_manager.store_org_token('some-org-id', 'some-token')
        assert token_manager.load_org_token('some-org-id') == 'some-token'
