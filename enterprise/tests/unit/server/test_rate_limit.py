"""Tests for the rate limit functionality with in-memory storage."""

import time
from unittest import mock

import limits
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from server.rate_limit import (
    RateLimiter,
    RateLimitException,
    RateLimitResult,
    _rate_limit_exceeded_handler,
    setup_rate_limit_handler,
)
from starlette.requests import Request
from starlette.responses import Response


@pytest.fixture
def rate_limiter():
    """Create a test rate limiter."""
    backend = limits.aio.storage.MemoryStorage()
    strategy = limits.aio.strategies.FixedWindowRateLimiter(backend)
    return RateLimiter(strategy, '1/second')


@pytest.fixture
def test_app(rate_limiter):
    """Create a FastAPI app with rate limiting for testing."""
    app = FastAPI()
    setup_rate_limit_handler(app)

    @app.get('/test')
    async def test_endpoint(request: Request):
        await rate_limiter.hit('test', 'user123')
        return {'message': 'success'}

    @app.get('/test-with-different-user')
    async def test_endpoint_different_user(request: Request, user_id: str = 'user123'):
        await rate_limiter.hit('test', user_id)
        return {'message': 'success'}

    return app


@pytest.fixture
def test_client(test_app):
    """Create a test client for the FastAPI app."""
    return TestClient(test_app)


@pytest.mark.asyncio
async def test_rate_limiter_hit_success(rate_limiter):
    """Test that hitting the rate limiter works when under the limit."""
    # Should not raise an exception
    await rate_limiter.hit('test', 'user123')


@pytest.mark.asyncio
async def test_rate_limiter_hit_exceeded(rate_limiter):
    """Test that hitting the rate limiter raises an exception when over the limit."""
    # First hit should succeed
    await rate_limiter.hit('test', 'user123')

    # Second hit should fail
    with pytest.raises(RateLimitException) as exc_info:
        await rate_limiter.hit('test', 'user123')

    # Check the exception details
    assert exc_info.value.status_code == 429
    assert '1 per 1 second' in exc_info.value.detail


def test_rate_limit_endpoint_success(test_client):
    """Test that the endpoint works when under the rate limit."""
    response = test_client.get('/test')
    assert response.status_code == 200
    assert response.json() == {'message': 'success'}


def test_rate_limit_endpoint_exceeded(test_client):
    """Test that the endpoint returns 429 when rate limit is exceeded."""
    # First request should succeed
    test_client.get('/test')

    # Second request should fail with 429
    response = test_client.get('/test')
    assert response.status_code == 429
    assert 'Rate limit exceeded' in response.json()['error']

    # Check headers
    assert 'X-RateLimit-Limit' in response.headers
    assert 'X-RateLimit-Remaining' in response.headers
    assert 'X-RateLimit-Reset' in response.headers
    assert 'Retry-After' in response.headers


def test_rate_limit_different_users(test_client):
    """Test that rate limits are applied per user."""
    # First user hits limit
    test_client.get('/test-with-different-user?user_id=user1')
    response = test_client.get('/test-with-different-user?user_id=user1')
    assert response.status_code == 429

    # Second user should still be able to make requests
    response = test_client.get('/test-with-different-user?user_id=user2')
    assert response.status_code == 200


def test_rate_limit_result_headers():
    """Test that rate limit headers are added correctly."""
    result = RateLimitResult(
        description='10 per 1 minute',
        remaining=5,
        reset_time=int(time.time()) + 30,
        retry_after=10,
    )

    # Mock response
    response = mock.MagicMock(spec=Response)
    response.headers = {}

    # Add headers
    result.add_headers(response)

    # Check headers
    assert response.headers['X-RateLimit-Limit'] == '10 per 1 minute'
    assert response.headers['X-RateLimit-Remaining'] == '5'
    assert 'X-RateLimit-Reset' in response.headers
    assert response.headers['Retry-After'] == '10'


def test_rate_limit_exception_handler():
    """Test the rate limit exception handler."""
    request = mock.MagicMock(spec=Request)

    # Create a rate limit result
    result = RateLimitResult(
        description='10 per 1 minute',
        remaining=0,
        reset_time=int(time.time()) + 30,
        retry_after=30,
    )

    # Create an exception
    exception = RateLimitException(result)

    # Call the handler
    response = _rate_limit_exceeded_handler(request, exception)

    # Check the response
    assert response.status_code == 429
    assert 'Rate limit exceeded: 10 per 1 minute' in response.body.decode()

    # Check headers
    assert response.headers['X-RateLimit-Limit'] == '10 per 1 minute'
    assert response.headers['X-RateLimit-Remaining'] == '0'
    assert 'X-RateLimit-Reset' in response.headers
    assert 'Retry-After' in response.headers
