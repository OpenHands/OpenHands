"""Tests for ConcurrencyLimitError.

This module tests the ConcurrencyLimitError exception including:
- HTTP 429 status code
- Structured error detail
- Limit and current count in response
"""

from fastapi import status

from openhands.app_server.errors import ConcurrencyLimitError


def _make_detail(limit: int, current: int) -> dict:
    """Helper to create standard concurrency limit error detail."""
    return {
        'error': 'CONCURRENCY_LIMIT_REACHED',
        'message': (
            f'You have reached your limit of {limit} '
            'concurrent conversations. Please close an existing '
            'conversation to start a new one.'
        ),
        'limit': limit,
        'current': current,
    }


class TestConcurrencyLimitError:
    """Test cases for ConcurrencyLimitError exception."""

    def test_creates_with_correct_status_code(self):
        """Test that ConcurrencyLimitError has HTTP 429 status code."""
        error = ConcurrencyLimitError(detail=_make_detail(5, 5))

        assert error.status_code == status.HTTP_429_TOO_MANY_REQUESTS

    def test_creates_detail_message(self):
        """Test that ConcurrencyLimitError creates appropriate detail message."""
        error = ConcurrencyLimitError(detail=_make_detail(3, 3))

        assert error.detail['error'] == 'CONCURRENCY_LIMIT_REACHED'
        assert 'limit of 3' in error.detail['message']
        assert error.detail['limit'] == 3
        assert error.detail['current'] == 3

    def test_includes_limit_and_current_in_detail(self):
        """Test that limit and current values are included in detail."""
        error = ConcurrencyLimitError(detail=_make_detail(10, 10))

        assert error.detail['limit'] == 10
        assert error.detail['current'] == 10

    def test_custom_detail(self):
        """Test that custom detail is used."""
        custom_detail = {'custom': 'error message'}
        error = ConcurrencyLimitError(detail=custom_detail)

        assert error.detail == custom_detail

    def test_headers_are_optional(self):
        """Test that headers parameter is optional."""
        error = ConcurrencyLimitError(detail=_make_detail(5, 5))

        # Should not raise, headers defaults to None
        assert error.headers is None

    def test_custom_headers_are_passed_through(self):
        """Test that custom headers are passed to parent."""
        custom_headers = {'Retry-After': '60'}
        error = ConcurrencyLimitError(detail=_make_detail(5, 5), headers=custom_headers)

        assert error.headers == custom_headers

    def test_message_includes_close_conversation_hint(self):
        """Test that error message hints to close existing conversation."""
        error = ConcurrencyLimitError(detail=_make_detail(5, 5))

        assert 'close an existing conversation' in error.detail['message']

    def test_different_limits_produce_different_messages(self):
        """Test that different limit values produce different messages."""
        error_3 = ConcurrencyLimitError(detail=_make_detail(3, 3))
        error_10 = ConcurrencyLimitError(detail=_make_detail(10, 10))

        assert 'limit of 3' in error_3.detail['message']
        assert 'limit of 10' in error_10.detail['message']
        assert error_3.detail['message'] != error_10.detail['message']

    def test_is_http_exception(self):
        """Test that ConcurrencyLimitError is an HTTPException."""
        from fastapi import HTTPException

        error = ConcurrencyLimitError(detail=_make_detail(5, 5))

        assert isinstance(error, HTTPException)

    def test_is_openhands_error(self):
        """Test that ConcurrencyLimitError is an OpenHandsError."""
        from openhands.app_server.errors import OpenHandsError

        error = ConcurrencyLimitError(detail=_make_detail(5, 5))

        assert isinstance(error, OpenHandsError)
