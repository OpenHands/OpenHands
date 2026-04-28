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


def _get_redis_kwargs():
    """Return common kwargs for Redis client creation."""
    return {
        'host': REDIS_HOST,
        'port': REDIS_PORT,
        'password': REDIS_PASSWORD,
        'db': REDIS_DB,
        'socket_timeout': REDIS_SOCKET_TIMEOUT,
    }


def create_redis_client() -> Redis:
    """Create a synchronous Redis client.

    Returns:
        A Redis client for synchronous operations.
    """
    return Redis(**_get_redis_kwargs())


def create_redis_client_async() -> aioredis.Redis:
    """Create an asynchronous Redis client.

    Returns:
        An aioredis client for asynchronous operations.
    """
    return aioredis.Redis(**_get_redis_kwargs())


def get_redis_authed_url():
    return f'redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}'


__all__ = [
    'Redis',
    'aioredis',
    'create_redis_client',
    'create_redis_client_async',
    'get_redis_authed_url',
    'redis_exceptions',
]
