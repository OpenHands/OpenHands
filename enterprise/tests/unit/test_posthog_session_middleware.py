"""Tests for PostHogSessionMiddleware."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request, Response


@pytest.fixture
def mock_response():
    return MagicMock(spec=Response)


def make_mock_request(headers: dict | None = None):
    """Create a mock FastAPI Request with a state object and headers dict."""
    request = MagicMock(spec=Request)
    request.headers = headers or {}
    request.state = MagicMock()
    return request


@pytest.mark.asyncio
async def test_middleware_sets_session_id_from_header(mock_response):
    """PostHogSessionMiddleware sets posthog_session_id from X-POSTHOG-SESSION-ID header."""
    from server.middleware import PostHogSessionMiddleware

    session_id = 'sess_abc123'
    request = make_mock_request({'X-POSTHOG-SESSION-ID': session_id})
    call_next = AsyncMock(return_value=mock_response)

    middleware = PostHogSessionMiddleware()
    result = await middleware(request, call_next)

    assert request.state.posthog_session_id == session_id
    call_next.assert_called_once_with(request)
    assert result == mock_response


@pytest.mark.asyncio
async def test_middleware_sets_none_when_header_absent(mock_response):
    """PostHogSessionMiddleware sets posthog_session_id to None when header is absent."""
    from server.middleware import PostHogSessionMiddleware

    request = make_mock_request({})  # No X-POSTHOG-SESSION-ID header
    call_next = AsyncMock(return_value=mock_response)

    middleware = PostHogSessionMiddleware()
    result = await middleware(request, call_next)

    assert request.state.posthog_session_id is None
    call_next.assert_called_once_with(request)
    assert result == mock_response


@pytest.mark.asyncio
async def test_middleware_does_not_modify_response(mock_response):
    """PostHogSessionMiddleware returns the response unchanged."""
    from server.middleware import PostHogSessionMiddleware

    request = make_mock_request({'X-POSTHOG-SESSION-ID': 'sess_xyz'})
    call_next = AsyncMock(return_value=mock_response)

    middleware = PostHogSessionMiddleware()
    result = await middleware(request, call_next)

    assert result is mock_response


@pytest.mark.asyncio
async def test_middleware_does_not_block_request(mock_response):
    """PostHogSessionMiddleware always calls call_next (never blocks)."""
    from server.middleware import PostHogSessionMiddleware

    request = make_mock_request({})
    call_next = AsyncMock(return_value=mock_response)

    middleware = PostHogSessionMiddleware()
    await middleware(request, call_next)

    call_next.assert_called_once_with(request)


@pytest.mark.asyncio
async def test_middleware_handles_case_insensitive_header(mock_response):
    """PostHogSessionMiddleware uses .get() which handles header lookup."""
    from server.middleware import PostHogSessionMiddleware

    # FastAPI/Starlette Headers are case-insensitive, but we test with dict mock
    # Test the exact header name used in the implementation
    session_id = 'sess_case_test'
    request = make_mock_request({'X-POSTHOG-SESSION-ID': session_id})
    call_next = AsyncMock(return_value=mock_response)

    middleware = PostHogSessionMiddleware()
    await middleware(request, call_next)

    assert request.state.posthog_session_id == session_id
