from collections.abc import Callable
from typing import (
    Any,
    Literal,
)

import litellm
from lmnr import (
    Instruments,
    Laminar,
    LaminarLiteLLMCallback,
    observe as laminar_observe,
)
from opentelemetry import trace

from openhands.sdk.logger import get_logger
from openhands.sdk.observability.utils import get_env


logger = get_logger(__name__)


def maybe_init_laminar():
    """Initialize Laminar if the environment variables are set.

    Example configuration:
    OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://otel-collector:4317/v1/traces

    # comma separated, key=value url-encoded pairs
    OTEL_EXPORTER_OTLP_TRACES_HEADERS="Authorization=Bearer%20<KEY>,X-Key=<CUSTOM_VALUE>"

    # grpc is assumed if not specified
    OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf # or grpc/protobuf
    # or
    OTEL_EXPORTER=otlp_http # or otlp_grpc
    """
    if should_enable_observability():
        if _is_otel_backend_laminar():
            Laminar.initialize()
        else:
            # Do not enable browser session replays for non-laminar backends
            Laminar.initialize(
                disabled_instruments=[
                    Instruments.BROWSER_USE_SESSION,
                    Instruments.PATCHRIGHT,
                    Instruments.PLAYWRIGHT,
                ],
            )
        litellm.callbacks.append(LaminarLiteLLMCallback())
    else:
        logger.debug(
            "Observability/OTEL environment variables are not set. "
            "Skipping Laminar initialization."
        )


def observe[**P, R](
    *,
    name: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    ignore_input: bool = False,
    ignore_output: bool = False,
    span_type: Literal["DEFAULT", "LLM", "TOOL"] = "DEFAULT",
    ignore_inputs: list[str] | None = None,
    input_formatter: Callable[P, str] | None = None,
    output_formatter: Callable[[R], str] | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    preserve_global_context: bool = False,
    **kwargs: dict[str, Any],
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        return laminar_observe(
            name=name,
            session_id=session_id,
            user_id=user_id,
            ignore_input=ignore_input,
            ignore_output=ignore_output,
            span_type=span_type,
            ignore_inputs=ignore_inputs,
            input_formatter=input_formatter,
            output_formatter=output_formatter,
            metadata=metadata,
            tags=tags,
            preserve_global_context=preserve_global_context,
            **kwargs,
        )(func)

    return decorator


def should_enable_observability():
    keys = [
        "LMNR_PROJECT_API_KEY",
        "OTEL_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
    ]
    if any(get_env(key) for key in keys):
        return True
    if Laminar.is_initialized():
        return True
    return False


def _is_otel_backend_laminar():
    """Simple heuristic to check if the OTEL backend is Laminar.
    Caveat: This will still be True if another backend uses the same
    authentication scheme, and the user uses LMNR_PROJECT_API_KEY
    instead of OTEL_HEADERS to authenticate.
    """
    key = get_env("LMNR_PROJECT_API_KEY")
    return key is not None and key != ""


class SpanManager:
    """Manages a stack of active spans and their associated tokens."""

    def __init__(self):
        self._stack: list[trace.Span] = []

    def start_active_span(self, name: str, session_id: str | None = None) -> None:
        """Start a new active span and push it to the stack."""
        span = Laminar.start_active_span(name)
        if session_id:
            Laminar.set_trace_session_id(session_id)
        self._stack.append(span)

    def end_active_span(self) -> None:
        """End the most recent active span by popping it from the stack."""
        if not self._stack:
            logger.warning("Attempted to end active span, but stack is empty")
            return

        try:
            span = self._stack.pop()
            if span and span.is_recording():
                span.end()
        except IndexError:
            logger.warning("Attempted to end active span, but stack is empty")
            return


_span_manager: SpanManager | None = None


def _get_span_manager() -> SpanManager:
    global _span_manager
    if _span_manager is None:
        _span_manager = SpanManager()
    return _span_manager


def start_active_span(name: str, session_id: str | None = None) -> None:
    """Start a new active span using the global span manager."""
    _get_span_manager().start_active_span(name, session_id)


def end_active_span() -> None:
    """End the most recent active span using the global span manager."""
    try:
        _get_span_manager().end_active_span()
    except Exception:
        logger.debug("Error ending active span")
        pass
