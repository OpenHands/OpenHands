# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
import time
from typing import Any, Callable

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from openhands.core.exceptions import LLMNoResponseError
from openhands.core.logger import openhands_logger as logger
from openhands.utils.tenacity_stop import stop_if_should_exit

# Circuit breaker constants
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
CIRCUIT_BREAKER_COOLDOWN_SECONDS = 30.0


class LLMCircuitBreakerError(RuntimeError):
    """Raised when the LLM circuit breaker is open due to consecutive failures.

    The circuit opens after CIRCUIT_BREAKER_FAILURE_THRESHOLD consecutive
    failures and remains open for CIRCUIT_BREAKER_COOLDOWN_SECONDS, during
    which all requests are immediately rejected to avoid cascade delays.
    """

    def __init__(self, cooldown_remaining: float) -> None:
        super().__init__(
            f'LLM circuit breaker is open after {CIRCUIT_BREAKER_FAILURE_THRESHOLD} '
            f'consecutive failures. Retry in {cooldown_remaining:.1f}s.'
        )
        self.cooldown_remaining = cooldown_remaining


class RetryMixin:
    """Mixin class for retry logic with circuit breaker protection.

    The circuit breaker tracks consecutive failures across all retry-decorated
    calls on the LLM instance. State attributes (_cb_consecutive_failures,
    _cb_opened_at) are lazily initialized on first access via _cb_failures
    and _cb_opened properties to work correctly as a mixin without requiring
    super().__init__() calls in the host class.
    """

    @property
    def _cb_failures(self) -> int:
        return getattr(self, '_cb_consecutive_failures', 0)

    @_cb_failures.setter
    def _cb_failures(self, value: int) -> None:
        self._cb_consecutive_failures = value

    @property
    def _cb_open_time(self) -> float:
        return getattr(self, '_cb_opened_at', 0.0)

    @_cb_open_time.setter
    def _cb_open_time(self, value: float) -> None:
        self._cb_opened_at = value

    def _check_circuit_breaker(self) -> None:
        """Check if the circuit breaker is open and raise if so.

        The circuit breaker opens after CIRCUIT_BREAKER_FAILURE_THRESHOLD
        consecutive failures. Once open, requests are rejected immediately
        for CIRCUIT_BREAKER_COOLDOWN_SECONDS to prevent cascade delays
        during provider outages.

        After the cooldown period, the circuit transitions to half-open:
        the next request is allowed through. If it succeeds, the circuit
        closes; if it fails, the circuit reopens.
        """
        if self._cb_failures < CIRCUIT_BREAKER_FAILURE_THRESHOLD:
            return

        elapsed = time.monotonic() - self._cb_open_time
        if elapsed < CIRCUIT_BREAKER_COOLDOWN_SECONDS:
            remaining = CIRCUIT_BREAKER_COOLDOWN_SECONDS - elapsed
            logger.warning(
                f'LLM circuit breaker is open ({self._cb_failures} '
                f'consecutive failures). Cooldown: {remaining:.1f}s remaining.'
            )
            raise LLMCircuitBreakerError(cooldown_remaining=remaining)

        # Cooldown has elapsed: transition to half-open (allow one attempt)
        logger.info(
            'LLM circuit breaker cooldown elapsed, allowing next request (half-open).'
        )

    def _record_circuit_breaker_success(self) -> None:
        """Record a successful LLM call, resetting the circuit breaker."""
        if self._cb_failures > 0:
            logger.info(
                f'LLM circuit breaker reset after success '
                f'(was at {self._cb_failures} consecutive failures).'
            )
        self._cb_failures = 0
        self._cb_open_time = 0.0

    def _record_circuit_breaker_failure(self) -> None:
        """Record a failed LLM call, potentially opening the circuit breaker."""
        self._cb_failures = self._cb_failures + 1
        if self._cb_failures >= CIRCUIT_BREAKER_FAILURE_THRESHOLD:
            if self._cb_open_time == 0.0:
                self._cb_open_time = time.monotonic()
                logger.warning(
                    f'LLM circuit breaker opened after '
                    f'{self._cb_failures} consecutive failures. '
                    f'Requests will be rejected for {CIRCUIT_BREAKER_COOLDOWN_SECONDS}s.'
                )
            else:
                # Reopen: reset the cooldown timer
                self._cb_open_time = time.monotonic()

    def retry_decorator(self, **kwargs: Any) -> Callable:
        """Create a LLM retry decorator with customizable parameters. This is used for 429 errors, and a few other exceptions in LLM classes.

        Includes circuit breaker protection: after 5 consecutive failures,
        requests are immediately rejected for a 30-second cooldown period.

        Args:
            **kwargs: Keyword arguments to override default retry behavior.
                      Keys: num_retries, retry_exceptions, retry_min_wait, retry_max_wait, retry_multiplier

        Returns:
            A retry decorator with the parameters customizable in configuration.
        """
        num_retries = kwargs.get('num_retries')
        retry_exceptions: tuple = kwargs.get('retry_exceptions', ())
        retry_min_wait = kwargs.get('retry_min_wait')
        retry_max_wait = kwargs.get('retry_max_wait')
        retry_multiplier = kwargs.get('retry_multiplier')
        retry_listener = kwargs.get('retry_listener')

        def before_sleep(retry_state: Any) -> None:
            self.log_retry_attempt(retry_state)
            if retry_listener:
                retry_listener(retry_state.attempt_number, num_retries)

            # Check if the exception is LLMNoResponseError
            exception = retry_state.outcome.exception()
            if isinstance(exception, LLMNoResponseError):
                if hasattr(retry_state, 'kwargs'):
                    # Only change temperature if it's zero or not set
                    current_temp = retry_state.kwargs.get('temperature', 0)
                    if current_temp == 0:
                        retry_state.kwargs['temperature'] = 1.0
                        logger.warning(
                            'LLMNoResponseError detected with temperature=0, setting temperature to 1.0 for next attempt.'
                        )
                    else:
                        logger.warning(
                            f'LLMNoResponseError detected with temperature={current_temp}, keeping original temperature'
                        )

        inner_retry_decorator: Callable = retry(
            before_sleep=before_sleep,
            stop=stop_after_attempt(num_retries) | stop_if_should_exit(),
            reraise=True,
            retry=(
                retry_if_exception_type(retry_exceptions)
            ),  # retry only for these types
            wait=wait_exponential(
                multiplier=retry_multiplier,
                min=retry_min_wait,
                max=retry_max_wait,
            ),
        )

        def circuit_breaker_decorator(func: Callable) -> Callable:
            """Wrap the retry-decorated function with circuit breaker logic."""
            import asyncio
            import inspect

            retried_func = inner_retry_decorator(func)

            async def async_wrapper(*args: Any, **kw: Any) -> Any:
                self._check_circuit_breaker()
                try:
                    result = await retried_func(*args, **kw)
                    self._record_circuit_breaker_success()
                    return result
                except Exception:
                    self._record_circuit_breaker_failure()
                    raise

            async def async_gen_wrapper(*args: Any, **kw: Any) -> Any:
                self._check_circuit_breaker()
                try:
                    async for item in retried_func(*args, **kw):
                        yield item
                    self._record_circuit_breaker_success()
                except Exception:
                    self._record_circuit_breaker_failure()
                    raise

            def sync_wrapper(*args: Any, **kw: Any) -> Any:
                self._check_circuit_breaker()
                try:
                    result = retried_func(*args, **kw)
                    self._record_circuit_breaker_success()
                    return result
                except Exception:
                    self._record_circuit_breaker_failure()
                    raise

            if inspect.isasyncgenfunction(func):
                setattr(async_gen_wrapper, '__wrapped__', func)
                return async_gen_wrapper
            elif asyncio.iscoroutinefunction(func):
                setattr(async_wrapper, '__wrapped__', func)
                return async_wrapper
            else:
                setattr(sync_wrapper, '__wrapped__', func)
                return sync_wrapper

        return circuit_breaker_decorator

    def log_retry_attempt(self, retry_state: Any) -> None:
        """Log retry attempts."""
        exception = retry_state.outcome.exception()

        # Add retry attempt and max retries to the exception for later use
        if hasattr(retry_state, 'retry_object') and hasattr(
            retry_state.retry_object, 'stop'
        ):
            # Get the max retries from the stop_after_attempt
            stop_condition = retry_state.retry_object.stop

            # Handle both single stop conditions and stop_any (combined conditions)
            stop_funcs = []
            if hasattr(stop_condition, 'stops'):
                # This is a stop_any object with multiple stop conditions
                stop_funcs = stop_condition.stops
            else:
                # This is a single stop condition
                stop_funcs = [stop_condition]

            for stop_func in stop_funcs:
                if hasattr(stop_func, 'max_attempts'):
                    # Add retry information to the exception
                    exception.retry_attempt = retry_state.attempt_number
                    exception.max_retries = stop_func.max_attempts
                    break

        logger.error(
            f'{exception}. Attempt #{retry_state.attempt_number} | You can customize retry values in the configuration.',
        )
