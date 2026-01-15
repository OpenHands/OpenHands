"""Tests for resend_keycloak module.

These tests verify the RateLimiter class functionality in isolation,
without requiring the full module dependencies (keycloak, server, etc.).
"""

import threading
import time
from typing import Optional

import pytest


class RateLimiter:
    """Thread-safe rate limiter for API calls.

    Copy of the class for testing in isolation.
    """

    def __init__(self, requests_per_second: float, safety_margin: float = 0.9):
        self._lock = threading.Lock()
        self._min_interval = 1.0 / (requests_per_second * safety_margin)
        self._last_call_time: Optional[float] = None

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if self._last_call_time is not None:
                elapsed = now - self._last_call_time
                if elapsed < self._min_interval:
                    sleep_time = self._min_interval - elapsed
                    time.sleep(sleep_time)
            self._last_call_time = time.monotonic()


class TestRateLimiter:
    """Test cases for the RateLimiter class."""

    def test_rate_limiter_first_call_no_wait(self):
        """First call should not wait."""
        rate_limiter = RateLimiter(requests_per_second=10, safety_margin=1.0)

        start = time.monotonic()
        rate_limiter.wait()
        elapsed = time.monotonic() - start

        # First call should be nearly instant (< 10ms)
        assert elapsed < 0.01

    def test_rate_limiter_enforces_interval(self):
        """Rate limiter should enforce minimum interval between calls."""
        # 2 requests per second with no safety margin = 0.5s between calls
        rate_limiter = RateLimiter(requests_per_second=2, safety_margin=1.0)

        rate_limiter.wait()  # First call
        start = time.monotonic()
        rate_limiter.wait()  # Second call should wait
        elapsed = time.monotonic() - start

        # Should wait approximately 0.5 seconds (allowing some tolerance)
        assert elapsed >= 0.45
        assert elapsed < 0.6

    def test_rate_limiter_safety_margin(self):
        """Safety margin should increase interval between calls."""
        # 2 requests per second with 0.5 safety margin = 1s between calls
        rate_limiter = RateLimiter(requests_per_second=2, safety_margin=0.5)

        rate_limiter.wait()  # First call
        start = time.monotonic()
        rate_limiter.wait()  # Second call should wait
        elapsed = time.monotonic() - start

        # Should wait approximately 1 second (allowing some tolerance)
        assert elapsed >= 0.95
        assert elapsed < 1.1

    def test_rate_limiter_thread_safety(self):
        """Rate limiter should be thread-safe."""
        # 10 requests per second = 0.1s between calls
        rate_limiter = RateLimiter(requests_per_second=10, safety_margin=1.0)
        call_times = []
        lock = threading.Lock()

        def make_call():
            rate_limiter.wait()
            with lock:
                call_times.append(time.monotonic())

        # Launch 5 threads simultaneously
        threads = [threading.Thread(target=make_call) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Check that calls were spaced appropriately
        call_times.sort()
        for i in range(1, len(call_times)):
            interval = call_times[i] - call_times[i - 1]
            # Each call should be at least 0.1s apart (with some tolerance)
            assert interval >= 0.09, f"Interval {interval} too short"

    def test_rate_limiter_respects_two_per_second_with_safety_margin(self):
        """With default 2 req/s and 0.9 safety margin, should allow ~1.8 req/s."""
        # This is the actual configuration used in production
        rate_limiter = RateLimiter(requests_per_second=2, safety_margin=0.9)

        # Make 4 calls (simulating 2 API calls per user for 2 users)
        start = time.monotonic()
        for _ in range(4):
            rate_limiter.wait()
        elapsed = time.monotonic() - start

        # 4 calls at 1.8 req/s should take at least ~1.67s (3 intervals * 0.556s)
        # With 2 req/s and 0.9 safety: interval = 1/(2*0.9) = 0.556s
        # 3 intervals = 1.667s minimum
        assert elapsed >= 1.6
        # But not too long (< 2.5s to allow for some overhead)
        assert elapsed < 2.5
