from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request, status
from server.utils.rate_limit_utils import (
    RATE_LIMIT_IP_SECONDS,
    RATE_LIMIT_USER_SECONDS,
    check_rate_limit_by_user_id,
)


@pytest.fixture
def mock_request():
    """Create a mock request object."""
    request = MagicMock(spec=Request)
    request.client = MagicMock()
    request.client.host = '192.168.1.1'
    return request


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    redis = AsyncMock()
    redis.set = AsyncMock(return_value=True)  # First call succeeds (key doesn't exist)
    return redis


@pytest.mark.asyncio
async def test_rate_limit_by_user_id_first_request_succeeds(mock_request, mock_redis):
    """Test that first request with user_id succeeds and sets rate limit key."""
    # Arrange
    user_id = 'test_user_id'
    key_prefix = 'email_resend'

    with (
        patch('server.utils.rate_limit_utils.sio') as mock_sio,
        patch('server.utils.rate_limit_utils.logger') as mock_logger,
    ):
        mock_sio.manager.redis = mock_redis

        # Act
        await check_rate_limit_by_user_id(
            request=mock_request, key_prefix=key_prefix, user_id=user_id
        )

        # Assert
        mock_redis.set.assert_called_once_with(
            f'{key_prefix}:{user_id}', 1, nx=True, ex=RATE_LIMIT_USER_SECONDS
        )
        mock_logger.warning.assert_not_called()
        mock_logger.info.assert_not_called()


@pytest.mark.asyncio
async def test_rate_limit_by_user_id_second_request_within_window_fails(
    mock_request, mock_redis
):
    """Test that second request with same user_id within rate limit window fails."""
    # Arrange
    user_id = 'test_user_id'
    key_prefix = 'email_resend'
    mock_redis.set = AsyncMock(return_value=False)  # Key already exists

    with (
        patch('server.utils.rate_limit_utils.sio') as mock_sio,
        patch('server.utils.rate_limit_utils.logger') as mock_logger,
    ):
        mock_sio.manager.redis = mock_redis

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await check_rate_limit_by_user_id(
                request=mock_request, key_prefix=key_prefix, user_id=user_id
            )

        assert exc_info.value.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        assert 'Too many requests' in exc_info.value.detail
        assert f'{RATE_LIMIT_USER_SECONDS // 60} minutes' in exc_info.value.detail
        mock_logger.info.assert_called_once()


@pytest.mark.asyncio
async def test_rate_limit_by_ip_when_user_id_is_none(mock_request, mock_redis):
    """Test that rate limiting falls back to IP address when user_id is None."""
    # Arrange
    key_prefix = 'email_resend'

    with (
        patch('server.utils.rate_limit_utils.sio') as mock_sio,
        patch('server.utils.rate_limit_utils.logger') as mock_logger,
    ):
        mock_sio.manager.redis = mock_redis

        # Act
        await check_rate_limit_by_user_id(
            request=mock_request, key_prefix=key_prefix, user_id=None
        )

        # Assert
        mock_redis.set.assert_called_once_with(
            f'{key_prefix}:ip:{mock_request.client.host}',
            1,
            nx=True,
            ex=RATE_LIMIT_IP_SECONDS,
        )
        mock_logger.warning.assert_not_called()


@pytest.mark.asyncio
async def test_rate_limit_by_ip_second_request_within_window_fails(
    mock_request, mock_redis
):
    """Test that second request from same IP within rate limit window fails."""
    # Arrange
    key_prefix = 'email_resend'
    mock_redis.set = AsyncMock(return_value=False)  # Key already exists

    with (
        patch('server.utils.rate_limit_utils.sio') as mock_sio,
    ):
        mock_sio.manager.redis = mock_redis

        # Act & Assert
        with pytest.raises(HTTPException) as exc_info:
            await check_rate_limit_by_user_id(
                request=mock_request, key_prefix=key_prefix, user_id=None
            )

        assert exc_info.value.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        assert f'{RATE_LIMIT_IP_SECONDS // 60} minutes' in exc_info.value.detail


@pytest.mark.asyncio
async def test_rate_limit_redis_unavailable_fails_open(mock_request):
    """Test that rate limiting fails open when Redis is unavailable."""
    # Arrange
    key_prefix = 'email_resend'
    user_id = 'test_user_id'

    with (
        patch('server.utils.rate_limit_utils.sio') as mock_sio,
        patch('server.utils.rate_limit_utils.logger') as mock_logger,
    ):
        mock_sio.manager.redis = None  # Redis unavailable

        # Act
        await check_rate_limit_by_user_id(
            request=mock_request, key_prefix=key_prefix, user_id=user_id
        )

        # Assert
        mock_logger.warning.assert_called_once_with(
            'Redis unavailable for rate limiting, allowing request'
        )


@pytest.mark.asyncio
async def test_rate_limit_redis_exception_fails_open(mock_request, mock_redis):
    """Test that rate limiting fails open when Redis raises an exception."""
    # Arrange
    key_prefix = 'email_resend'
    user_id = 'test_user_id'
    mock_redis.set = AsyncMock(side_effect=Exception('Redis connection error'))

    with (
        patch('server.utils.rate_limit_utils.sio') as mock_sio,
        patch('server.utils.rate_limit_utils.logger') as mock_logger,
    ):
        mock_sio.manager.redis = mock_redis

        # Act
        await check_rate_limit_by_user_id(
            request=mock_request, key_prefix=key_prefix, user_id=user_id
        )

        # Assert
        mock_logger.warning.assert_called_once()
        assert 'Error checking rate limit' in str(mock_logger.warning.call_args[0][0])


@pytest.mark.asyncio
async def test_rate_limit_custom_key_prefix(mock_request, mock_redis):
    """Test that different key prefixes create different rate limit keys."""
    # Arrange
    user_id = 'test_user_id'
    key_prefix = 'password_reset'

    with patch('server.utils.rate_limit_utils.sio') as mock_sio:
        mock_sio.manager.redis = mock_redis

        # Act
        await check_rate_limit_by_user_id(
            request=mock_request, key_prefix=key_prefix, user_id=user_id
        )

        # Assert
        mock_redis.set.assert_called_once_with(
            f'{key_prefix}:{user_id}', 1, nx=True, ex=RATE_LIMIT_USER_SECONDS
        )


@pytest.mark.asyncio
async def test_rate_limit_custom_rate_limit_seconds(mock_request, mock_redis):
    """Test that custom rate limit seconds are used correctly."""
    # Arrange
    user_id = 'test_user_id'
    key_prefix = 'email_resend'
    custom_user_seconds = 60
    custom_ip_seconds = 180

    with patch('server.utils.rate_limit_utils.sio') as mock_sio:
        mock_sio.manager.redis = mock_redis

        # Act
        await check_rate_limit_by_user_id(
            request=mock_request,
            key_prefix=key_prefix,
            user_id=user_id,
            user_rate_limit_seconds=custom_user_seconds,
            ip_rate_limit_seconds=custom_ip_seconds,
        )

        # Assert
        mock_redis.set.assert_called_once_with(
            f'{key_prefix}:{user_id}', 1, nx=True, ex=custom_user_seconds
        )


@pytest.mark.asyncio
async def test_rate_limit_ip_with_unknown_client(mock_request, mock_redis):
    """Test that rate limiting handles missing client host gracefully."""
    # Arrange
    key_prefix = 'email_resend'
    mock_request.client = None  # No client information

    with patch('server.utils.rate_limit_utils.sio') as mock_sio:
        mock_sio.manager.redis = mock_redis

        # Act
        await check_rate_limit_by_user_id(
            request=mock_request, key_prefix=key_prefix, user_id=None
        )

        # Assert
        mock_redis.set.assert_called_once_with(
            f'{key_prefix}:ip:unknown', 1, nx=True, ex=RATE_LIMIT_IP_SECONDS
        )


@pytest.mark.asyncio
async def test_rate_limit_different_users_have_separate_limits(
    mock_request, mock_redis
):
    """Test that different user_ids have separate rate limit keys."""
    # Arrange
    key_prefix = 'email_resend'
    user_id_1 = 'user_1'
    user_id_2 = 'user_2'

    with patch('server.utils.rate_limit_utils.sio') as mock_sio:
        mock_sio.manager.redis = mock_redis

        # Act
        await check_rate_limit_by_user_id(
            request=mock_request, key_prefix=key_prefix, user_id=user_id_1
        )
        await check_rate_limit_by_user_id(
            request=mock_request, key_prefix=key_prefix, user_id=user_id_2
        )

        # Assert
        assert mock_redis.set.call_count == 2
        # Extract call arguments properly
        call_args_list = [
            (call[0][0], call[0][1], call[1]['nx'], call[1]['ex'])
            for call in mock_redis.set.call_args_list
        ]
        assert (
            f'{key_prefix}:{user_id_1}',
            1,
            True,
            RATE_LIMIT_USER_SECONDS,
        ) in call_args_list
        assert (
            f'{key_prefix}:{user_id_2}',
            1,
            True,
            RATE_LIMIT_USER_SECONDS,
        ) in call_args_list
