import importlib
import logging
import os
import secrets
import threading
from dataclasses import dataclass
from typing import Any

from redis import asyncio as aioredis
from redis import exceptions as redis_exceptions

_logger = logging.getLogger(__name__)
_redis_client_async: aioredis.Redis | None = None
_redis_lock = threading.Lock()

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


def _get_redis_client_async():
    try:
        redis_module = importlib.import_module('storage.redis')
        return redis_module.get_redis_client_async()
    except ImportError:
        pass

    global _redis_client_async
    if _redis_client_async is None:
        with _redis_lock:
            if _redis_client_async is None:
                _redis_client_async = aioredis.Redis(
                    host=os.environ.get('REDIS_HOST', 'localhost'),
                    port=int(os.environ.get('REDIS_PORT', '6379')),
                    password=os.environ.get('REDIS_PASSWORD', ''),
                    db=int(os.environ.get('REDIS_DB', '0')),
                    socket_timeout=2,
                )
    return _redis_client_async


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
    redis = _get_redis_client_async()
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
