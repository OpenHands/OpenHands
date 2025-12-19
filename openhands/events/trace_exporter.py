import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import httpx

from openhands.core.logger import openhands_logger as logger
from openhands.events.event import Event
from openhands.events.serialization.event import event_to_dict
from openhands.events.stream import EventStream, EventStreamSubscriber


@dataclass(frozen=True)
class TraceExportConfig:
    url: str
    api_key: str | None
    headers: dict[str, str]
    batch_size: int
    flush_interval_s: float
    timeout_s: float
    max_queue_size: int


class TraceExporter:
    """
    Lightweight scaffolding for exporting OpenHands event stream traces to an external HTTP API.

    Design:
    - Subscribes to EventStream and receives already-redacted events (EventStream replaces secrets).
    - Buffers events and sends batched POST requests.
    - Best-effort: failures are logged; events may be dropped under sustained backpressure.
    """

    def __init__(self, event_stream: EventStream, cfg: TraceExportConfig):
        self._event_stream = event_stream
        self._cfg = cfg

        self._q: "queue.Queue[dict[str, Any]]" = queue.Queue(maxsize=cfg.max_queue_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name='trace-exporter', daemon=True)

        self._subscriber_id = EventStreamSubscriber.MAIN
        self._callback_id = f'trace_exporter:{event_stream.sid}'

    def start(self) -> None:
        self._event_stream.subscribe(self._subscriber_id, self._on_event, self._callback_id)
        self._thread.start()
        logger.info(
            'Trace exporter enabled',
            extra={'session_id': self._event_stream.sid, 'trace_export_url': self._cfg.url},
        )

    def stop(self) -> None:
        self._stop.set()
        try:
            self._event_stream.unsubscribe(self._subscriber_id, self._callback_id)
        except Exception:
            # Unsubscribe is best-effort; don't crash shutdown paths.
            pass

    def _on_event(self, event: Event) -> None:
        # Convert to dict; event already had secrets replaced when written to the stream.
        payload = {
            'session_id': self._event_stream.sid,
            'user_id': self._event_stream.user_id,
            'event': event_to_dict(event),
        }
        try:
            self._q.put_nowait(payload)
        except queue.Full:
            logger.warning(
                'Trace exporter queue full; dropping event',
                extra={'session_id': self._event_stream.sid, 'max_queue_size': self._cfg.max_queue_size},
            )

    def _run(self) -> None:
        batch: list[dict[str, Any]] = []
        last_flush = time.monotonic()

        while not self._stop.is_set():
            timeout = max(0.05, self._cfg.flush_interval_s / 4)
            try:
                item = self._q.get(timeout=timeout)
                batch.append(item)
            except queue.Empty:
                pass

            now = time.monotonic()
            should_flush = (
                len(batch) >= self._cfg.batch_size
                or (batch and (now - last_flush) >= self._cfg.flush_interval_s)
            )
            if not should_flush:
                continue

            self._flush(batch)
            batch = []
            last_flush = now

        # final flush on stop
        if batch:
            self._flush(batch)

    def _flush(self, batch: list[dict[str, Any]]) -> None:
        headers = dict(self._cfg.headers or {})
        headers.setdefault('Content-Type', 'application/json')
        if self._cfg.api_key:
            headers.setdefault('Authorization', f'Bearer {self._cfg.api_key}')

        body = {'events': batch}
        try:
            with httpx.Client(timeout=self._cfg.timeout_s) as client:
                resp = client.post(self._cfg.url, json=body, headers=headers)
                if resp.status_code >= 400:
                    logger.warning(
                        'Trace export failed',
                        extra={
                            'session_id': self._event_stream.sid,
                            'status_code': resp.status_code,
                            'response_text': resp.text[:2_000],
                        },
                    )
        except Exception as e:
            logger.warning(
                f'Trace export exception: {type(e).__name__}: {e}',
                extra={'session_id': self._event_stream.sid},
            )

