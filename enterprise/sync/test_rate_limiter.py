"""Tests for the RateLimiter class."""

import threading
import time

import pytest


class RateLimiter:
    """Thread-safe rate limiter for API calls.

    This is a copy of the RateLimiter class from resend_keycloak.py for testing purposes.
    """

    def __init__(self, requests_per_second: float, safety_margin: float = 0.1):
        self._lock = threading.Lock()
        self._last_call_time: float = 0.0
        effective_rate = requests_per_second * (1 - safety_margin)
        self._min_interval = 1.0 / effective_rate

    def wait(self) -> None:
        with self._lock:
            current_time = time.time()
            time_since_last_call = current_time - self._last_call_time
            if time_since_last_call < self._min_interval:
                sleep_time = self._min_interval - time_since_last_call
                time.sleep(sleep_time)
            self._last_call_time = time.time()


class TestRateLimiter:
    """Tests for the RateLimiter class."""

    def test_rate_limiter_enforces_minimum_interval(self):
        """Test that rate limiter enforces minimum interval between calls."""
        # 2 requests per second with 10% safety margin = 1.8 req/s
        # Minimum interval = 1/1.8 = ~0.556 seconds
        rate_limiter = RateLimiter(requests_per_second=2.0, safety_margin=0.1)

        start_time = time.time()
        rate_limiter.wait()
        first_call_time = time.time()
        rate_limiter.wait()
        second_call_time = time.time()

        # First call should be immediate
        assert first_call_time - start_time < 0.1

        # Second call should wait for minimum interval (~0.556s)
        interval = second_call_time - first_call_time
        assert interval >= 0.5, f"Expected interval >= 0.5s, got {interval}s"

    def test_rate_limiter_no_wait_after_sufficient_time(self):
        """Test that rate limiter doesn't wait if enough time has passed."""
        rate_limiter = RateLimiter(requests_per_second=2.0, safety_margin=0.1)

        rate_limiter.wait()
        # Wait longer than minimum interval
        time.sleep(0.7)

        start_time = time.time()
        rate_limiter.wait()
        elapsed = time.time() - start_time

        # Should not wait since we already waited 0.7s
        assert elapsed < 0.1, f"Expected no wait, but waited {elapsed}s"

    def test_rate_limiter_thread_safety(self):
        """Test that rate limiter is thread-safe."""
        rate_limiter = RateLimiter(requests_per_second=2.0, safety_margin=0.1)
        call_times = []
        lock = threading.Lock()

        def make_call():
            rate_limiter.wait()
            with lock:
                call_times.append(time.time())

        threads = [threading.Thread(target=make_call) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Sort call times
        call_times.sort()

        # Check that intervals between consecutive calls are >= minimum interval
        min_interval = 1.0 / (2.0 * 0.9)  # ~0.556s
        for i in range(1, len(call_times)):
            interval = call_times[i] - call_times[i - 1]
            # Allow some tolerance for timing variations
            assert (
                interval >= min_interval * 0.9
            ), f"Interval {interval}s is less than expected {min_interval}s"

    def test_rate_limiter_with_different_rates(self):
        """Test rate limiter with different rate configurations."""
        # Test with 1 request per second
        rate_limiter = RateLimiter(requests_per_second=1.0, safety_margin=0.0)

        rate_limiter.wait()
        start = time.time()
        rate_limiter.wait()
        elapsed = time.time() - start

        # Should wait approximately 1 second
        assert 0.9 <= elapsed <= 1.2, f"Expected ~1s wait, got {elapsed}s"

    def test_rate_limiter_safety_margin(self):
        """Test that safety margin is applied correctly."""
        # 10 req/s with 50% safety margin = 5 req/s effective
        # Minimum interval = 0.2s
        rate_limiter = RateLimiter(requests_per_second=10.0, safety_margin=0.5)

        rate_limiter.wait()
        start = time.time()
        rate_limiter.wait()
        elapsed = time.time() - start

        # Should wait approximately 0.2 seconds
        assert 0.15 <= elapsed <= 0.3, f"Expected ~0.2s wait, got {elapsed}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
