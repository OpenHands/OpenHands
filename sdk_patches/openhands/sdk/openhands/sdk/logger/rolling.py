# rolling_view.py
import logging
import sys
from collections import deque
from collections.abc import Callable
from contextlib import contextmanager

from rich.live import Live

from .logger import ENV_JSON, IN_CI


RenderFnType = Callable[[], str]


class _RollingViewHandler(logging.Handler):
    def __init__(self, max_lines: int, use_live: bool):
        super().__init__()
        self._buf: deque[str] = deque(maxlen=max_lines)
        self._use_live: bool = use_live
        self._live: Live | None = None  # set by rolling_log_view when Live is active
        self.render_fn: RenderFnType | None = None

    def emit(self, record: logging.LogRecord):
        msg = self.format(record)
        self._buf.append(msg)

        if self._use_live and self._live:
            # Live mode: repaint using either a custom render_fn or the buffer
            self._live.update(
                self.render_fn() if self.render_fn else "\n".join(self._buf)
            )
            return

        # Non-live paths
        if ENV_JSON:
            # JSON mode: do nothing here; rely on other handlers via propagation
            return

        # CI / non-TTY plain pass-through (avoid double newlines)
        sys.stdout.write(msg + "\n")
        sys.stdout.flush()

    @property
    def snapshot(self) -> str:
        return "\n".join(self._buf)


@contextmanager
def rolling_log_view(
    logger: logging.Logger,
    max_lines: int = 60,
    level: int = logging.INFO,
    propagate: bool = False,
    header: str | None = None,
    footer: str | None = None,
    *,
    json_flush_level: int
    | None = None,  # optional: separate level for the final JSON flush
):
    """
    Temporarily attach a rolling view handler that renders the last N log lines.

    - Local TTY & not CI & not JSON: pretty, live-updating view (Rich.Live)
    - CI / non-TTY: plain line-by-line (no terminal control)
    - JSON mode: buffer only; on exit emit ONE large log record with the full snapshot.
    """
    is_tty = sys.stdout.isatty()
    use_live = (not IN_CI) and is_tty and (not ENV_JSON)

    handler = _RollingViewHandler(max_lines=max_lines, use_live=use_live)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(message)s"))

    prev_propagate = logger.propagate
    # Let other handlers (e.g., your JSON handler) run if needed
    logger.propagate = bool(propagate or ENV_JSON)

    logger.addHandler(handler)

    def _render() -> str:
        parts: list[str] = []
        if header:
            parts.append(header.rstrip())
        parts.append("\n".join(handler._buf))
        if footer:
            parts.append(footer.rstrip())
        return "\n".join(parts)

    try:
        if use_live:
            with Live(_render(), refresh_per_second=8) as live:
                handler._live = live
                handler.render_fn = _render
                yield handler
        else:
            yield handler
    finally:
        final_text = _render()

        # Freeze final frame if Live was active
        if handler._live:
            handler._live.update(final_text)

        # Detach our handler BEFORE flushing to avoid recursion
        logger.removeHandler(handler)
        logger.propagate = prev_propagate

        # JSON mode: emit one big record at exit
        if ENV_JSON:
            logger.log(
                json_flush_level if json_flush_level is not None else level, final_text
            )
