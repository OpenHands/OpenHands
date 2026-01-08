"""Tests for distributed_lock module."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from storage.distributed_lock import DistributedLock


@pytest.fixture
def mock_redis():
    """Create a mock Redis client for testing."""
    return MagicMock()


class TestDistributedLock:
    """Tests for the DistributedLock class."""

    @pytest.mark.asyncio
    async def test_acquire_lock_success(self, mock_redis):
        """Test successful lock acquisition."""
        mock_redis.set.return_value = True

        lock = DistributedLock(
            key='test_key',
            timeout_seconds=10,
            redis_client=mock_redis,
        )

        acquired = await lock.acquire(wait=False)

        assert acquired is True
        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        assert call_args[0][0] == 'lock:test_key'
        assert call_args[1]['nx'] is True
        assert call_args[1]['ex'] == 10

    @pytest.mark.asyncio
    async def test_acquire_lock_failure_no_wait(self, mock_redis):
        """Test failed lock acquisition without waiting."""
        mock_redis.set.return_value = False

        lock = DistributedLock(
            key='test_key',
            redis_client=mock_redis,
        )

        acquired = await lock.acquire(wait=False)

        assert acquired is False

    @pytest.mark.asyncio
    async def test_acquire_lock_with_wait_timeout(self, mock_redis):
        """Test lock acquisition with wait that times out."""
        mock_redis.set.return_value = False

        lock = DistributedLock(
            key='test_key',
            max_wait_seconds=0.3,
            retry_delay_seconds=0.1,
            redis_client=mock_redis,
        )

        acquired = await lock.acquire(wait=True)

        assert acquired is False
        # Should have tried multiple times
        assert mock_redis.set.call_count >= 2

    @pytest.mark.asyncio
    async def test_acquire_lock_eventual_success(self, mock_redis):
        """Test lock acquisition that succeeds after a few retries."""
        # First two attempts fail, third succeeds
        mock_redis.set.side_effect = [False, False, True]

        lock = DistributedLock(
            key='test_key',
            max_wait_seconds=1.0,
            retry_delay_seconds=0.05,
            redis_client=mock_redis,
        )

        acquired = await lock.acquire(wait=True)

        assert acquired is True
        assert mock_redis.set.call_count == 3

    @pytest.mark.asyncio
    async def test_release_lock_success(self, mock_redis):
        """Test successful lock release."""
        mock_redis.set.return_value = True
        mock_redis.eval.return_value = 1

        lock = DistributedLock(
            key='test_key',
            redis_client=mock_redis,
        )

        await lock.acquire(wait=False)
        released = await lock.release()

        assert released is True
        mock_redis.eval.assert_called_once()

    @pytest.mark.asyncio
    async def test_release_lock_not_acquired(self, mock_redis):
        """Test release when lock was not acquired."""
        lock = DistributedLock(
            key='test_key',
            redis_client=mock_redis,
        )

        # Try to release without acquiring
        released = await lock.release()

        assert released is False
        mock_redis.eval.assert_not_called()

    @pytest.mark.asyncio
    async def test_release_lock_owned_by_another(self, mock_redis):
        """Test release when lock is owned by another process."""
        mock_redis.set.return_value = True
        mock_redis.eval.return_value = 0  # Lua script returns 0 if not owner

        lock = DistributedLock(
            key='test_key',
            redis_client=mock_redis,
        )

        await lock.acquire(wait=False)
        released = await lock.release()

        assert released is False

    @pytest.mark.asyncio
    async def test_context_manager_success(self, mock_redis):
        """Test using lock as async context manager."""
        mock_redis.set.return_value = True
        mock_redis.eval.return_value = 1

        lock = DistributedLock(
            key='test_key',
            redis_client=mock_redis,
        )

        async with lock as acquired:
            assert acquired is True

        # Lock should be released after context
        mock_redis.eval.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_failure(self, mock_redis):
        """Test context manager when lock cannot be acquired."""
        mock_redis.set.return_value = False

        lock = DistributedLock(
            key='test_key',
            max_wait_seconds=0.1,
            retry_delay_seconds=0.05,
            redis_client=mock_redis,
        )

        async with lock as acquired:
            assert acquired is False

        # Release should not be called since we never acquired
        mock_redis.eval.assert_not_called()

    @pytest.mark.asyncio
    async def test_is_locked(self, mock_redis):
        """Test checking if lock is held."""
        mock_redis.exists.return_value = 1

        lock = DistributedLock(
            key='test_key',
            redis_client=mock_redis,
        )

        assert lock.is_locked() is True
        mock_redis.exists.assert_called_once_with('lock:test_key')

    @pytest.mark.asyncio
    async def test_is_not_locked(self, mock_redis):
        """Test checking when lock is not held."""
        mock_redis.exists.return_value = 0

        lock = DistributedLock(
            key='test_key',
            redis_client=mock_redis,
        )

        assert lock.is_locked() is False

    @pytest.mark.asyncio
    async def test_acquire_redis_error(self, mock_redis):
        """Test handling Redis errors during acquire."""
        mock_redis.set.side_effect = Exception('Redis connection error')

        lock = DistributedLock(
            key='test_key',
            redis_client=mock_redis,
        )

        acquired = await lock.acquire(wait=False)

        assert acquired is False

    @pytest.mark.asyncio
    async def test_release_redis_error(self, mock_redis):
        """Test handling Redis errors during release."""
        mock_redis.set.return_value = True
        mock_redis.eval.side_effect = Exception('Redis connection error')

        lock = DistributedLock(
            key='test_key',
            redis_client=mock_redis,
        )

        await lock.acquire(wait=False)
        released = await lock.release()

        assert released is False

    @pytest.mark.asyncio
    async def test_lock_key_prefix(self, mock_redis):
        """Test that lock key is properly prefixed."""
        mock_redis.set.return_value = True

        lock = DistributedLock(
            key='my_resource',
            redis_client=mock_redis,
        )

        await lock.acquire(wait=False)

        call_args = mock_redis.set.call_args
        assert call_args[0][0] == 'lock:my_resource'

    @pytest.mark.asyncio
    async def test_unique_lock_value(self, mock_redis):
        """Test that each lock instance has a unique value."""
        mock_redis.set.return_value = True

        lock1 = DistributedLock(key='test', redis_client=mock_redis)
        lock2 = DistributedLock(key='test', redis_client=mock_redis)

        assert lock1._lock_value != lock2._lock_value

    @pytest.mark.asyncio
    async def test_creates_redis_client_if_not_provided(self):
        """Test that a Redis client is created if not provided."""
        with patch('storage.distributed_lock.create_redis_client') as mock_create:
            mock_client = MagicMock()
            mock_client.set.return_value = True
            mock_create.return_value = mock_client

            lock = DistributedLock(key='test')

            # Access redis_client property to trigger creation
            _ = lock.redis_client

            mock_create.assert_called_once()
