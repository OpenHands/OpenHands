import asyncio
import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as futures_wait
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from openhands.core.logger import openhands_logger as logger
from openhands.events.event import Event, EventSource
from openhands.events.event_store import EventStore
from openhands.events.serialization.event import event_to_dict
from openhands.io import json
from openhands.storage import FileStore
from openhands.storage.locations import (
    get_conversation_dir,
    get_conversation_event_count_filename,
)
from openhands.utils.async_utils import call_sync_from_async
from openhands.utils.shutdown_listener import should_continue


class EventStreamSubscriber(str, Enum):
    AGENT_CONTROLLER = 'agent_controller'
    RESOLVER = 'openhands_resolver'
    SERVER = 'server'
    RUNTIME = 'runtime'
    MEMORY = 'memory'
    MAIN = 'main'
    TEST = 'test'


async def session_exists(
    sid: str, file_store: FileStore, user_id: str | None = None
) -> bool:
    try:
        await call_sync_from_async(file_store.list, get_conversation_dir(sid, user_id))
        return True
    except FileNotFoundError:
        return False


class EventStream(EventStore):
    secrets: dict[str, str]
    # For each subscriber ID, there is a map of callback functions - useful
    # when there are multiple listeners
    _subscribers: dict[str, dict[str, Callable]]
    _lock: threading.Lock
    _queue: queue.Queue[Event | object]
    _queue_thread: threading.Thread
    _queue_loop: asyncio.AbstractEventLoop | None
    _callback_pool: ThreadPoolExecutor
    _write_page_cache: list[dict]

    # Background writer settings
    _FLUSH_INTERVAL_SECS: float = 0.1  # 100ms
    _FLUSH_BATCH_SIZE: int = 25
    # Shared callback pool size — covers typical subscriber count (controller,
    # server, runtime, memory) with headroom for bursts.
    _CALLBACK_POOL_SIZE: int = 4

    # Sentinel object placed on the queue to signal the processing loop to
    # exit, replacing the previous timeout-based polling approach.
    _SHUTDOWN_SENTINEL = object()

    def __init__(self, sid: str, file_store: FileStore, user_id: str | None = None):
        super().__init__(sid, file_store, user_id)
        self._stop_flag = threading.Event()
        self._queue: queue.Queue[Event | object] = queue.Queue()
        self._queue_loop = None
        self._queue_thread = threading.Thread(target=self._run_queue_loop)
        self._queue_thread.daemon = True
        self._queue_thread.start()
        self._subscribers = {}
        self._lock = threading.Lock()
        self.secrets = {}
        self._write_page_cache = []

        # Single shared thread pool for all subscriber callbacks, replacing
        # the previous per-subscriber ThreadPoolExecutor(max_workers=1).
        # This reduces thread count by ~4x per session.
        self._callback_pool = ThreadPoolExecutor(max_workers=self._CALLBACK_POOL_SIZE)

        # Thread-safe queue for surfacing callback errors to callers.
        # Errors are collected here instead of being silently swallowed.
        self._callback_errors: queue.Queue[tuple[str, str, Exception]] = queue.Queue()

        # Background writer: buffer of (event_id, event_json, data_dict, write_page_ref) tuples
        self._pending_writes: list[tuple[int, str, dict, list[dict]]] = []
        self._write_lock = threading.Lock()
        self._write_ready = threading.Event()
        self._writer_thread = threading.Thread(
            target=self._run_background_writer, daemon=True
        )
        self._writer_thread.start()

    def close(self) -> None:
        self._stop_flag.set()

        # Signal the background writer to wake up and drain remaining writes
        self._write_ready.set()
        if self._writer_thread.is_alive():
            self._writer_thread.join(timeout=5.0)

        # Wake the queue processing loop immediately via sentinel
        self._queue.put(self._SHUTDOWN_SENTINEL)
        if self._queue_thread.is_alive():
            self._queue_thread.join()

        subscriber_ids = list(self._subscribers.keys())
        for subscriber_id in subscriber_ids:
            callback_ids = list(self._subscribers[subscriber_id].keys())
            for callback_id in callback_ids:
                self._clean_up_subscriber(subscriber_id, callback_id)

        # Shut down the shared callback pool
        self._callback_pool.shutdown(wait=False)

        # Clear queue
        while not self._queue.empty():
            self._queue.get()

    def _clean_up_subscriber(self, subscriber_id: str, callback_id: str) -> None:
        if subscriber_id not in self._subscribers:
            logger.warning(f'Subscriber not found during cleanup: {subscriber_id}')
            return
        if callback_id not in self._subscribers[subscriber_id]:
            logger.warning(f'Callback not found during cleanup: {callback_id}')
            return

        del self._subscribers[subscriber_id][callback_id]

    def subscribe(
        self,
        subscriber_id: EventStreamSubscriber,
        callback: Callable[[Event], None],
        callback_id: str,
    ) -> None:
        if subscriber_id not in self._subscribers:
            self._subscribers[subscriber_id] = {}

        if callback_id in self._subscribers[subscriber_id]:
            raise ValueError(
                f'Callback ID on subscriber {subscriber_id} already exists: {callback_id}'
            )

        self._subscribers[subscriber_id][callback_id] = callback

    def unsubscribe(
        self, subscriber_id: EventStreamSubscriber, callback_id: str
    ) -> None:
        if subscriber_id not in self._subscribers:
            logger.warning(f'Subscriber not found during unsubscribe: {subscriber_id}')
            return

        if callback_id not in self._subscribers[subscriber_id]:
            logger.warning(f'Callback not found during unsubscribe: {callback_id}')
            return

        self._clean_up_subscriber(subscriber_id, callback_id)

    def add_event(self, event: Event, source: EventSource) -> None:
        if event.id != Event.INVALID_ID:
            raise ValueError(
                f'Event already has an ID:{event.id}. It was probably added back to the EventStream from inside a handler, triggering a loop.'
            )
        event._timestamp = datetime.now().isoformat()
        event._source = source  # type: ignore [attr-defined]
        with self._lock:
            event._id = self.cur_id  # type: ignore [attr-defined]
            self.cur_id += 1

            # Take a copy of the current write page
            current_write_page = self._write_page_cache

            data = event_to_dict(event)
            data = self._replace_secrets(data)
            current_write_page.append(data)

            # If the page is full, create a new page for future events / other threads to use
            if len(current_write_page) == self.cache_size:
                self._write_page_cache = []

        if event.id is not None:
            # Serialize the event JSON eagerly (cheap CPU work) but defer
            # the file I/O to the background writer thread.
            event_json = json.dumps(data)
            if len(event_json) > 1_000_000:  # Roughly 1MB in bytes, ignoring encoding
                filename = self._get_filename_for_id(event.id, self.user_id)
                logger.warning(
                    f'Saving event JSON over 1MB: {len(event_json):,} bytes, filename: {filename}',
                    extra={
                        'user_id': self.user_id,
                        'session_id': self.sid,
                        'size': len(event_json),
                    },
                )

            with self._write_lock:
                self._pending_writes.append(
                    (event.id, event_json, data, current_write_page)
                )
                pending_count = len(self._pending_writes)

            # Signal the writer if the batch is full; otherwise it will
            # wake on its own after _FLUSH_INTERVAL_SECS.
            if pending_count >= self._FLUSH_BATCH_SIZE:
                self._write_ready.set()

        # Dispatch to subscriber callbacks immediately (no I/O wait)
        self._queue.put(event)

    def _store_cache_page(self, current_write_page: list[dict]):
        """Store a page in the cache. Reading individual events is slow when there are a lot of them, so we use pages."""
        if len(current_write_page) < self.cache_size:
            return
        start = current_write_page[0]['id']
        end = start + self.cache_size
        contents = json.dumps(current_write_page)
        cache_filename = self._get_filename_for_cache(start, end)
        self.file_store.write(cache_filename, contents)

    def _run_background_writer(self) -> None:
        """Background thread that batch-flushes buffered events to the file store.

        Wakes every _FLUSH_INTERVAL_SECS (100ms) or immediately when the
        pending buffer reaches _FLUSH_BATCH_SIZE (25 events).
        """
        while not self._stop_flag.is_set():
            # Wait for either the batch-size signal or the timeout
            self._write_ready.wait(timeout=self._FLUSH_INTERVAL_SECS)
            self._write_ready.clear()
            self._flush_pending_writes()

        # Final drain on shutdown
        self._flush_pending_writes()

    def _flush_pending_writes(self) -> None:
        """Write all buffered events to the file store in one batch."""
        with self._write_lock:
            batch = self._pending_writes
            self._pending_writes = []

        if not batch:
            return

        max_event_id = -1
        seen_pages: set[int] = set()
        for event_id, event_json, data, write_page in batch:
            filename = self._get_filename_for_id(event_id, self.user_id)
            self.file_store.write(filename, event_json)

            # Track the highest event ID for the metadata file
            if event_id > max_event_id:
                max_event_id = event_id

            # Store cache pages (deduplicate by page start id)
            if len(write_page) >= self.cache_size:
                page_start = write_page[0]['id']
                if page_start not in seen_pages:
                    seen_pages.add(page_start)
                    self._store_cache_page(write_page)

        # Persist the event count metadata so that subsequent startups
        # can skip the O(N) directory scan.
        if max_event_id >= 0:
            count_filename = get_conversation_event_count_filename(
                self.sid, self.user_id
            )
            self.file_store.write(count_filename, str(max_event_id + 1))

    def get_callback_errors(self) -> list[tuple[str, str, Exception]]:
        """Drain and return all callback errors accumulated since the last call.

        Returns a list of (subscriber_id, callback_id, exception) tuples.
        This makes subscriber callback bugs visible instead of silently
        swallowed in background threads.
        """
        errors: list[tuple[str, str, Exception]] = []
        while not self._callback_errors.empty():
            try:
                errors.append(self._callback_errors.get_nowait())
            except queue.Empty:
                break
        return errors

    def set_secrets(self, secrets: dict[str, str]) -> None:
        self.secrets = secrets.copy()

    def update_secrets(self, secrets: dict[str, str]) -> None:
        self.secrets.update(secrets)

    def _replace_secrets(
        self, data: dict[str, Any], is_top_level: bool = True
    ) -> dict[str, Any]:
        # Fields that should not have secrets replaced (only at top level - system metadata)
        TOP_LEVEL_PROTECTED_FIELDS = {
            'timestamp',
            'id',
            'source',
            'cause',
            'action',
            'observation',
            'message',
        }

        for key in data:
            if is_top_level and key in TOP_LEVEL_PROTECTED_FIELDS:
                # Skip secret replacement for protected system fields at top level only
                continue
            elif isinstance(data[key], dict):
                data[key] = self._replace_secrets(data[key], is_top_level=False)
            elif isinstance(data[key], str):
                for secret in self.secrets.values():
                    data[key] = data[key].replace(secret, '<secret_hidden>')
        return data

    def _run_queue_loop(self) -> None:
        self._queue_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._queue_loop)
        try:
            self._queue_loop.run_until_complete(self._process_queue())
        finally:
            self._queue_loop.close()

    async def _process_queue(self) -> None:
        while should_continue() and not self._stop_flag.is_set():
            # Blocking get with no timeout — the shutdown sentinel wakes us
            # immediately instead of polling every 100ms.
            item = self._queue.get()
            if item is self._SHUTDOWN_SENTINEL:
                break

            event: Event = item  # type: ignore[assignment]

            # Dispatch each event to all subscriber callbacks via the shared
            # pool. We collect futures so we can wait for all callbacks to
            # finish before processing the next event — this preserves
            # per-event ordering across subscribers.
            futures: list[tuple[Future, str, str]] = []
            for key in sorted(self._subscribers.keys()):
                callbacks = self._subscribers[key]
                # Create a copy of the keys to avoid "dictionary changed size during iteration" error
                callback_ids = list(callbacks.keys())
                for callback_id in callback_ids:
                    # Check if callback_id still exists (might have been removed during iteration)
                    if callback_id in callbacks:
                        callback = callbacks[callback_id]
                        future = self._callback_pool.submit(callback, event)
                        futures.append((future, callback_id, key))

            # Wait for all callbacks to complete before processing next event
            if futures:
                futures_wait([f for f, _, _ in futures])
                for future, callback_id, subscriber_id in futures:
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(
                            f'Error in event callback {callback_id} for subscriber {subscriber_id}: {str(e)}',
                            exc_info=True,
                        )
                        # Store the error so callers can inspect/propagate it
                        self._callback_errors.put_nowait(
                            (subscriber_id, callback_id, e)
                        )
