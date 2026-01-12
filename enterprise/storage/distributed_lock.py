"""Distributed lock implementation using Redis."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING

from server.logger import logger
from storage.redis import create_redis_client

if TYPE_CHECKING:
    import redis


class DistributedLock:
    """A distributed lock using Redis with automatic expiration.

    This lock uses Redis SETNX (SET if Not eXists) to implement a distributed lock.
    The lock automatically expires after a configurable timeout to prevent deadlocks.

    Usage:
        lock = DistributedLock("my_lock_key")
        acquired = await lock.acquire()
        if acquired:
            try:
                # do work
            finally:
                await lock.release()

    Or as a context manager:
        async with DistributedLock("my_lock_key") as acquired:
            if acquired:
                # do work
    """

    def __init__(
        self,
        key: str,
        timeout_seconds: int = 30,
        retry_delay_seconds: float = 0.1,
        max_wait_seconds: float = 10.0,
        redis_client: redis.Redis | None = None,
    ):
        """Initialize a distributed lock.

        Args:
            key: The lock key name
            timeout_seconds: Lock auto-expiration time (prevents deadlocks)
            retry_delay_seconds: Delay between retry attempts when waiting
            max_wait_seconds: Maximum time to wait for lock acquisition
            redis_client: Optional Redis client (creates one if not provided)
        """
        self.key = f'lock:{key}'
        self.timeout_seconds = timeout_seconds
        self.retry_delay_seconds = retry_delay_seconds
        self.max_wait_seconds = max_wait_seconds
        self._redis_client = redis_client
        self._lock_value = str(uuid.uuid4())
        self._acquired = False

    @property
    def redis_client(self) -> redis.Redis:
        if self._redis_client is None:
            self._redis_client = create_redis_client()
        return self._redis_client

    async def acquire(self, wait: bool = True) -> bool:
        """Attempt to acquire the lock.

        Args:
            wait: If True, wait up to max_wait_seconds for the lock.
                  If False, return immediately if lock is not available.

        Returns:
            True if lock was acquired, False otherwise.
        """
        start_time = time.monotonic()

        while True:
            try:
                # Use SET with NX (only set if not exists) and EX (expiration)
                acquired = self.redis_client.set(
                    self.key,
                    self._lock_value,
                    nx=True,
                    ex=self.timeout_seconds,
                )

                if acquired:
                    self._acquired = True
                    logger.debug(
                        'distributed_lock:acquired',
                        extra={'key': self.key, 'lock_value': self._lock_value},
                    )
                    return True

                if not wait:
                    return False

                elapsed = time.monotonic() - start_time
                if elapsed >= self.max_wait_seconds:
                    logger.warning(
                        'distributed_lock:timeout',
                        extra={
                            'key': self.key,
                            'waited_seconds': elapsed,
                            'max_wait_seconds': self.max_wait_seconds,
                        },
                    )
                    return False

                # Wait before retrying
                await asyncio.sleep(self.retry_delay_seconds)

            except Exception as e:
                logger.error(
                    'distributed_lock:acquire_error',
                    extra={'key': self.key, 'error': str(e)},
                )
                # On Redis errors, we should not block - return False to allow fallback
                return False

    async def release(self) -> bool:
        """Release the lock if we own it.

        Uses a Lua script to atomically check ownership and delete.

        Returns:
            True if lock was released, False if we didn't own it or error occurred.
        """
        if not self._acquired:
            return False

        # Lua script to atomically check and delete only if we own the lock
        release_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """

        try:
            result = self.redis_client.eval(
                release_script, 1, self.key, self._lock_value
            )
            self._acquired = False
            if result:
                logger.debug(
                    'distributed_lock:released',
                    extra={'key': self.key, 'lock_value': self._lock_value},
                )
            return bool(result)
        except Exception as e:
            logger.error(
                'distributed_lock:release_error',
                extra={'key': self.key, 'error': str(e)},
            )
            self._acquired = False
            return False

    async def __aenter__(self) -> bool:
        """Async context manager entry - acquires the lock."""
        return await self.acquire()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - releases the lock."""
        await self.release()

    def is_locked(self) -> bool:
        """Check if the lock is currently held (by anyone).

        Returns:
            True if the lock exists in Redis, False otherwise.
        """
        try:
            return self.redis_client.exists(self.key) > 0
        except Exception as e:
            logger.error(
                'distributed_lock:is_locked_error',
                extra={'key': self.key, 'error': str(e)},
            )
            return False
