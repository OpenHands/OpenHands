import os

from redis import asyncio as aioredis
from redis import exceptions as redis_exceptions
from redis import Redis

# Redis configuration
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', '6379'))
REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD', '')
REDIS_DB = int(os.environ.get('REDIS_DB', '0'))
REDIS_SOCKET_TIMEOUT = 2

_redis_client: Redis | None = None
_redis_client_async: aioredis.Redis | None = None


def _get_redis_kwargs():
    """Return common kwargs for Redis client creation."""
    return {
        'host': REDIS_HOST,
        'port': REDIS_PORT,
        'password': REDIS_PASSWORD,
        'db': REDIS_DB,
        'socket_timeout': REDIS_SOCKET_TIMEOUT,
    }


def get_redis_client() -> Redis:
    """Get a shared synchronous Redis client, lazily initialized.

    Returns:
        A Redis client for synchronous operations.
    """
    global _redis_client
    if _redis_client is None:
        _redis_client = Redis(**_get_redis_kwargs())
    return _redis_client


def get_redis_client_async() -> aioredis.Redis:
    """Get a shared asynchronous Redis client, lazily initialized.

    Returns:
        An aioredis client for asynchronous operations.
    """
    global _redis_client_async
    if _redis_client_async is None:
        _redis_client_async = aioredis.Redis(**_get_redis_kwargs())
    return _redis_client_async


def get_redis_authed_url():
    return f'redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}'


__all__ = [
    'Redis',
    'aioredis',
    'get_redis_client',
    'get_redis_client_async',
    'get_redis_authed_url',
    'redis_exceptions',
]
