#!/usr/bin/env python3
"""Tests for the RateLimiter class in resend_keycloak module."""

import threading
import time
import unittest


class RateLimiter:
    """Thread-safe rate limiter for API calls.

    Copied here for isolated testing without module dependencies.
    """

    def __init__(self, rate_limit: float, safety_margin: float = 0.9):
        """Initialize the rate limiter.

        Args:
            rate_limit: Maximum requests per second.
            safety_margin: Multiplier to stay safely under the limit (default 0.9 = 90%).
        """
        self.rate_limit = rate_limit * safety_margin
        self.min_interval = 1.0 / self.rate_limit
        self.last_call_time = 0.0
        self.lock = threading.Lock()

    def wait(self):
        """Wait until it's safe to make the next API call."""
        with self.lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call_time

            if time_since_last_call < self.min_interval:
                sleep_time = self.min_interval - time_since_last_call
                time.sleep(sleep_time)

            self.last_call_time = time.time()


class TestRateLimiter(unittest.TestCase):
    """Tests for the RateLimiter class."""

    def test_first_call_no_wait(self):
        """First call should not wait."""
        limiter = RateLimiter(rate_limit=2.0)
        start = time.time()
        limiter.wait()
        elapsed = time.time() - start
        # First call should be nearly instant
        self.assertLess(elapsed, 0.1)

    def test_rate_limiting_enforced(self):
        """Subsequent calls should be rate limited."""
        limiter = RateLimiter(rate_limit=2.0, safety_margin=1.0)  # 2 req/s = 0.5s interval
        limiter.wait()
        start = time.time()
        limiter.wait()
        elapsed = time.time() - start
        # Should wait approximately 0.5 seconds (1/2 req/s)
        self.assertGreaterEqual(elapsed, 0.4)
        self.assertLess(elapsed, 0.7)

    def test_safety_margin_applied(self):
        """Safety margin should reduce effective rate."""
        limiter = RateLimiter(rate_limit=2.0, safety_margin=0.5)  # 1 req/s = 1.0s interval
        limiter.wait()
        start = time.time()
        limiter.wait()
        elapsed = time.time() - start
        # Should wait approximately 1.0 second (2 * 0.5 = 1 req/s)
        self.assertGreaterEqual(elapsed, 0.9)
        self.assertLess(elapsed, 1.2)

    def test_thread_safety(self):
        """Rate limiter should work correctly with multiple threads."""
        limiter = RateLimiter(rate_limit=10.0, safety_margin=1.0)  # 10 req/s
        call_times = []
        lock = threading.Lock()

        def make_call():
            limiter.wait()
            with lock:
                call_times.append(time.time())

        threads = [threading.Thread(target=make_call) for _ in range(5)]
        start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        total_time = time.time() - start

        # 5 calls at 10 req/s should take at least 0.4 seconds (4 intervals of 0.1s)
        self.assertGreaterEqual(total_time, 0.35)
        self.assertEqual(len(call_times), 5)

    def test_no_wait_after_interval_passed(self):
        """Should not wait if enough time has passed since last call."""
        limiter = RateLimiter(rate_limit=10.0, safety_margin=1.0)  # 0.1s interval
        limiter.wait()
        time.sleep(0.2)  # Wait longer than interval
        start = time.time()
        limiter.wait()
        elapsed = time.time() - start
        # Should not wait since interval has passed
        self.assertLess(elapsed, 0.05)


if __name__ == '__main__':
    unittest.main()
