import asyncio
import logging
import secrets
from dataclasses import dataclass
from typing import Any

from openhands.app_server.utils.redis import get_redis_client_async, redis_exceptions

_logger = logging.getLogger(__name__)

_RELEASE_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
end
return 0
"""

_REFRESH_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("expire", KEYS[1], ARGV[2])
end
return 0
"""


class RedisLockUnavailable(Exception):
    """Raised when Redis cannot be used to evaluate a lock."""


@dataclass
class RedisLock:
    redis: Any
    key: str
    token: str
    ttl_seconds: int

    async def refresh(self) -> bool:
        try:
            refreshed = await self.redis.eval(
                _REFRESH_SCRIPT, 1, self.key, self.token, self.ttl_seconds
            )
            return bool(refreshed)
        except redis_exceptions.RedisError:
            _logger.warning('redis_lock:refresh_failed', extra={'key': self.key})
            return False

    async def release(self) -> bool:
        try:
            released = await self.redis.eval(_RELEASE_SCRIPT, 1, self.key, self.token)
            return bool(released)
        except redis_exceptions.RedisError:
            _logger.warning('redis_lock:release_failed', extra={'key': self.key})
            return False


async def try_acquire_redis_lock(key: str, ttl_seconds: int) -> RedisLock | None:
    """Acquire a Redis lock or return None when another holder owns it."""
    redis = get_redis_client_async()
    token = secrets.token_urlsafe(24)
    try:
        acquired = await redis.set(key, token, nx=True, ex=ttl_seconds)
    except redis_exceptions.RedisError as e:
        raise RedisLockUnavailable from e

    if not acquired:
        return None

    return RedisLock(
        redis=redis,
        key=key,
        token=token,
        ttl_seconds=ttl_seconds,
    )


async def refresh_lock_periodically(lock: RedisLock, interval: int) -> None:
    """Keep a Redis lock alive by refreshing its TTL every *interval* seconds.

    Intended to run as a background task (via ``asyncio.create_task``) alongside
    a long-running operation.  Cancel the task when the operation finishes; the
    caller is responsible for releasing the lock afterwards.
    """
    try:
        while True:
            await asyncio.sleep(interval)
            if not await lock.refresh():
                _logger.warning(
                    'redis_lock:periodic_refresh_failed', extra={'key': lock.key}
                )
    except asyncio.CancelledError:
        pass
