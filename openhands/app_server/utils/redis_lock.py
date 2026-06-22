import asyncio
import logging

from redis.asyncio.lock import Lock
from redis.exceptions import LockError

from openhands.app_server.utils.redis import get_redis_client_async, redis_exceptions

_logger = logging.getLogger(__name__)

# Re-export so callers can catch lock errors without importing redis directly.
__all__ = [
    'Lock',
    'LockError',
    'RedisLockUnavailable',
    'try_acquire_redis_lock',
    'refresh_lock_periodically',
]


class RedisLockUnavailable(Exception):
    """Raised when Redis cannot be used to evaluate a lock."""


async def try_acquire_redis_lock(key: str, ttl_seconds: int) -> Lock | None:
    """Try to acquire a Redis lock; return None if already held by another caller."""
    redis = get_redis_client_async()
    lock = redis.lock(key, timeout=ttl_seconds)
    try:
        acquired = await lock.acquire(blocking=False)
    except redis_exceptions.RedisError as e:
        raise RedisLockUnavailable from e
    return lock if acquired else None


async def refresh_lock_periodically(lock: Lock, interval: int) -> None:
    """Keep a Redis lock alive by refreshing its TTL every *interval* seconds.

    Intended to run as a background task (via ``asyncio.create_task``) alongside
    a long-running operation.  Cancel the task when the operation finishes; the
    caller is responsible for releasing the lock afterwards.
    """
    try:
        while True:
            await asyncio.sleep(interval)
            try:
                await lock.reacquire()
            except LockError:
                _logger.warning(
                    'redis_lock:periodic_refresh_failed', extra={'key': lock.name}
                )
    except asyncio.CancelledError:
        pass
