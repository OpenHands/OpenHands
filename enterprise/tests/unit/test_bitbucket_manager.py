"""
Tests for the Bitbucket manager.
"""

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from integrations.bitbucket.bitbucket_manager import BitbucketManager
from integrations.models import Message, SourceType
from server.auth.token_manager import ProviderType


@pytest.fixture
def bitbucket_manager():
    """Create a BitbucketManager instance for testing."""
    with patch('integrations.bitbucket.bitbucket_manager.TokenManager'):
        manager = BitbucketManager()
        manager.token_manager = AsyncMock()
        return manager


@pytest.mark.asyncio
async def test_extract_user_info(bitbucket_manager):
    """Test extracting user information from Bitbucket webhook event."""
    event_data = {
        'actor': {
            'accountId': '5b857d7bb8e3cb58958607cc',
            'username': 'testuser',
            'uuid': '{82ebd5fb-c165-4e68-b5ab-ddab6c8467d1}',
        }
    }

    user_info = bitbucket_manager._extract_user_info(event_data)

    assert user_info is not None
    assert user_info.bitbucket_id == '5b857d7bb8e3cb58958607cc'
    assert user_info.username == 'testuser'
    assert user_info.keycloak_user_id is None
    # Verify the integer user_id is stable (hash-based)
    expected_hash = int(
        hashlib.sha256('5b857d7bb8e3cb58958607cc'.encode()).hexdigest()[:8], 16
    )
    assert user_info.user_id == expected_hash


@pytest.mark.asyncio
async def test_extract_user_info_no_username(bitbucket_manager):
    """Test extracting user info when username is missing."""
    event_data = {
        'actor': {
            'accountId': '5b857d7bb8e3cb58958607cc',
            'uuid': '{82ebd5fb-c165-4e68-b5ab-ddab6c8467d1}',
        }
    }

    user_info = bitbucket_manager._extract_user_info(event_data)

    assert user_info is not None
    assert user_info.bitbucket_id == '5b857d7bb8e3cb58958607cc'
    assert user_info.username == '5b857d7bb8e3cb58958607cc'  # Falls back to account_id


@pytest.mark.asyncio
async def test_extract_user_info_missing_actor(bitbucket_manager):
    """Test extracting user info when actor is missing."""
    event_data: dict = {}

    user_info = bitbucket_manager._extract_user_info(event_data)

    assert user_info is None


@pytest.mark.asyncio
async def test_receive_message_pr_comment_with_mention(bitbucket_manager):
    """Test processing a PR comment with @openhands mention."""
    # Mock the token manager responses
    bitbucket_manager.token_manager.get_user_id_from_idp_user_id.return_value = (
        'keycloak-user-123'
    )
    bitbucket_manager.token_manager.get_idp_token_from_idp_user_id.return_value = (
        'test-token'
    )

    # Create a PR comment event with mention
    event_data = {
        'actor': {
            'accountId': '5b857d7bb8e3cb58958607cc',
            'username': 'testuser',
        },
        'comment': {'content': {'raw': '@openhands help me fix this bug'}},
        'workspace': {'slug': 'test-workspace'},
        'repository': {'slug': 'test-repo'},
        'pullrequest': {'id': 1},
    }

    message = Message(
        source=SourceType.BITBUCKET, message={'event': event_data, 'context': {}}
    )

    with patch.object(bitbucket_manager, '_check_write_access', return_value=True):
        await bitbucket_manager.receive_message(message)

    # Verify user lookup was attempted
    bitbucket_manager.token_manager.get_user_id_from_idp_user_id.assert_called_once_with(
        '5b857d7bb8e3cb58958607cc', ProviderType.BITBUCKET
    )

    # Verify token retrieval was attempted
    bitbucket_manager.token_manager.get_idp_token_from_idp_user_id.assert_called_once_with(
        '5b857d7bb8e3cb58958607cc', ProviderType.BITBUCKET
    )


@pytest.mark.asyncio
async def test_receive_message_no_mention(bitbucket_manager):
    """Test that comments without @openhands mention are ignored."""
    event_data = {
        'actor': {
            'accountId': '5b857d7bb8e3cb58958607cc',
            'username': 'testuser',
        },
        'comment': {'content': {'raw': 'This is just a regular comment'}},
        'workspace': {'slug': 'test-workspace'},
        'repository': {'slug': 'test-repo'},
    }

    message = Message(
        source=SourceType.BITBUCKET, message={'event': event_data, 'context': {}}
    )

    await bitbucket_manager.receive_message(message)

    # Verify no user lookup was attempted
    bitbucket_manager.token_manager.get_user_id_from_idp_user_id.assert_not_called()


@pytest.mark.asyncio
async def test_receive_message_user_not_found(bitbucket_manager):
    """Test handling when user is not found in Keycloak."""
    # Mock the token manager to return None (user not found)
    bitbucket_manager.token_manager.get_user_id_from_idp_user_id.return_value = None

    event_data = {
        'actor': {
            'accountId': '5b857d7bb8e3cb58958607cc',
            'username': 'testuser',
        },
        'comment': {'content': {'raw': '@openhands help me'}},
        'workspace': {'slug': 'test-workspace'},
        'repository': {'slug': 'test-repo'},
    }

    message = Message(
        source=SourceType.BITBUCKET, message={'event': event_data, 'context': {}}
    )

    await bitbucket_manager.receive_message(message)

    # Verify user lookup was attempted
    bitbucket_manager.token_manager.get_user_id_from_idp_user_id.assert_called_once()

    # Verify token retrieval was NOT attempted (user not found)
    bitbucket_manager.token_manager.get_idp_token_from_idp_user_id.assert_not_called()


@pytest.mark.asyncio
async def test_receive_message_wrong_source(bitbucket_manager):
    """Test that non-Bitbucket messages are rejected."""
    message = Message(source=SourceType.GITHUB, message={'event': {}, 'context': {}})

    await bitbucket_manager.receive_message(message)

    # Verify no processing occurred
    bitbucket_manager.token_manager.get_user_id_from_idp_user_id.assert_not_called()


@pytest.mark.asyncio
async def test_send_response_to_pr(bitbucket_manager):
    """Test sending response back to Bitbucket PR."""
    import os

    # Set the webhook URL environment variable
    with patch.dict(
        os.environ, {'FORGE_APP_WEBHOOK_URL': 'https://forge.example.com/webhook'}
    ):
        with patch(
            'integrations.bitbucket.bitbucket_manager.aiohttp.ClientSession'
        ) as mock_session_class:
            # Mock the response
            mock_response = AsyncMock()
            mock_response.status = 200

            # Mock the ClientSession and its async context manager behavior
            mock_session = MagicMock()
            mock_session_class.return_value.__aenter__.return_value = mock_session

            # Mock session.post() to return an async context manager
            mock_session.post.return_value.__aenter__.return_value = mock_response

            # Send a response
            result = await bitbucket_manager.send_response_to_pr(
                'test-workspace', 'test-repo', 123, 'Test message'
            )

            assert result is True

            # Verify the request was made with correct data
            mock_session.post.assert_called_once()
            call_args = mock_session.post.call_args
            assert call_args[0][0] == 'https://forge.example.com/webhook'
            assert call_args[1]['json'] == {
                'workspace': 'test-workspace',
                'repo': 'test-repo',
                'prId': 123,
                'message': 'Test message',
            }


@pytest.mark.asyncio
async def test_send_response_no_webhook_url(bitbucket_manager):
    """Test that sending response fails gracefully when webhook URL is not configured."""
    import os

    # Ensure FORGE_APP_WEBHOOK_URL is not set
    with patch.dict(os.environ, {}, clear=True):
        result = await bitbucket_manager.send_response_to_pr(
            'test-workspace', 'test-repo', 123, 'Test message'
        )

        assert result is False
