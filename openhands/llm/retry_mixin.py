# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
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


class RetryMixin:
    """Mixin class for retry logic."""

    def retry_decorator(self, num_retries: int, **kwargs: Any) -> Callable:
        """Create a LLM retry decorator with customizable parameters. This is used for 429 errors, and a few other exceptions in LLM classes.

        Args:
            num_retries: Number of retry attempts.
            **kwargs: Keyword arguments to override default retry behavior.
                      Keys: retry_exceptions, retry_min_wait, retry_max_wait, retry_multiplier

        Returns:
            A retry decorator with the parameters customizable in configuration.
        """
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

        retry_decorator: Callable = retry(
            before_sleep=before_sleep,
            stop=stop_after_attempt(num_retries) | stop_if_should_exit(),
            reraise=True,
            retry=(
                retry_if_exception_type(retry_exceptions)
            ),  # retry only for these types
            # TODO: Type errors here probably mean the defaults are being ignored.
            wait=wait_exponential(
                multiplier=retry_multiplier,  # type: ignore[arg-type]
                min=retry_min_wait,  # type: ignore[arg-type]
                max=retry_max_wait,  # type: ignore[arg-type]
            ),
        )
        return retry_decorator

    def _get_diagnostic_hint(self, exception: Any, exception_type: str) -> str:
        """Generate diagnostic hints based on exception type.

        Args:
            exception: The exception that occurred
            exception_type: The name of the exception type

        Returns:
            A human-readable diagnostic hint
        """
        # Timeout errors
        if 'Timeout' in exception_type or 'timeout' in str(exception).lower():
            return (
                "Connection timeout - possible causes: slow network, service overload, or local server not running. "
                "Check base_url is accessible. For Ollama: ensure 'ollama serve' is running."
            )

        # Connection errors
        if 'Connection' in exception_type or 'ECONNREFUSED' in str(exception):
            return (
                "Cannot connect to LLM service - verify base_url is correct and service is running. "
                "For local models (Ollama, LM Studio): check if server is listening on the configured port."
            )

        # Rate limit errors
        if 'RateLimit' in exception_type or '429' in str(exception):
            return (
                "Rate limit exceeded - too many requests. Try increasing retry wait times in configuration "
                "or reduce request frequency."
            )

        # Service unavailable errors
        if 'ServiceUnavailable' in exception_type or '503' in str(exception):
            return (
                "LLM service temporarily unavailable. Server may be restarting, out of capacity, or under maintenance. "
                "Retries will be attempted with exponential backoff."
            )

        # No response errors
        if 'NoResponse' in exception_type:
            return (
                "LLM returned no response - may be an API issue or service overload. "
                "Temperature will be adjusted to add randomness for next attempt."
            )

        # Malformed/invalid response
        if 'Malformed' in exception_type or 'Invalid' in exception_type:
            return (
                "LLM returned malformed response - possible API change or internal server error. "
                "Check server logs and ensure model is compatible."
            )

        # Default hint
        return "Check your LLM configuration and network connectivity. Retrying with exponential backoff."

    def log_retry_attempt(self, retry_state: Any) -> None:
        """Log retry attempts with diagnostic hints."""
        exception = retry_state.outcome.exception()
        exception_type = type(exception).__name__

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

        # Generate diagnostic message based on exception type
        diagnostic_hint = self._get_diagnostic_hint(exception, exception_type)

        logger.error(
            f'{exception_type}: {str(exception)[:200]}. Attempt #{retry_state.attempt_number} | {diagnostic_hint}'
        )
