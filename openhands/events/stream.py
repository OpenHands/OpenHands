import asyncio
import json
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from enum import Enum
from functools import partial
from typing import Any, Callable

from openhands.core.logger import openhands_logger as logger
from openhands.events.event import Event, EventSource
from openhands.events.event_store import EventStore
from openhands.events.serialization.event import event_from_dict, event_to_dict
from openhands.storage import FileStore
from openhands.storage.locations import (
    get_conversation_dir,
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
    _queue: queue.Queue[Event]
    _queue_thread: threading.Thread
    _queue_loop: asyncio.AbstractEventLoop | None
    _thread_pools: dict[str, dict[str, ThreadPoolExecutor]]
    _thread_loops: dict[str, dict[str, asyncio.AbstractEventLoop]]
    _write_page_cache: list[dict]

    def __init__(self, sid: str, file_store: FileStore, user_id: str | None = None):
        super().__init__(sid, file_store, user_id)
        self._stop_flag = threading.Event()
        self._queue: queue.Queue[Event] = queue.Queue()
        self._thread_pools = {}
        self._thread_loops = {}
        self._queue_loop = None
        self._queue_thread = threading.Thread(target=self._run_queue_loop)
        self._queue_thread.daemon = True
        self._queue_thread.start()
        self._subscribers = {}
        self._lock = threading.Lock()
        self.secrets = {}
        self._write_page_cache = []

    def _init_thread_loop(self, subscriber_id: str, callback_id: str) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        if subscriber_id not in self._thread_loops:
            self._thread_loops[subscriber_id] = {}
        self._thread_loops[subscriber_id][callback_id] = loop

    def close(self) -> None:
        self._stop_flag.set()
        if self._queue_thread.is_alive():
            self._queue_thread.join()

        subscriber_ids = list(self._subscribers.keys())
        for subscriber_id in subscriber_ids:
            callback_ids = list(self._subscribers[subscriber_id].keys())
            for callback_id in callback_ids:
                self._clean_up_subscriber(subscriber_id, callback_id)

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
        if (
            subscriber_id in self._thread_loops
            and callback_id in self._thread_loops[subscriber_id]
        ):
            loop = self._thread_loops[subscriber_id][callback_id]
            current_task = asyncio.current_task(loop)
            pending = [
                task for task in asyncio.all_tasks(loop) if task is not current_task
            ]
            for task in pending:
                task.cancel()
            try:
                loop.stop()
                loop.close()
            except Exception as e:
                logger.warning(
                    f'Error closing loop for {subscriber_id}/{callback_id}: {e}'
                )
            del self._thread_loops[subscriber_id][callback_id]

        if (
            subscriber_id in self._thread_pools
            and callback_id in self._thread_pools[subscriber_id]
        ):
            pool = self._thread_pools[subscriber_id][callback_id]
            pool.shutdown()
            del self._thread_pools[subscriber_id][callback_id]

        del self._subscribers[subscriber_id][callback_id]

    def subscribe(
        self,
        subscriber_id: EventStreamSubscriber,
        callback: Callable[[Event], None],
        callback_id: str,
    ) -> None:
        initializer = partial(self._init_thread_loop, subscriber_id, callback_id)
        pool = ThreadPoolExecutor(max_workers=1, initializer=initializer)
        if subscriber_id not in self._subscribers:
            self._subscribers[subscriber_id] = {}
            self._thread_pools[subscriber_id] = {}

        if callback_id in self._subscribers[subscriber_id]:
            raise ValueError(
                f'Callback ID on subscriber {subscriber_id} already exists: {callback_id}'
            )

        self._subscribers[subscriber_id][callback_id] = callback
        self._thread_pools[subscriber_id][callback_id] = pool

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
            event = event_from_dict(data)
            current_write_page.append(data)

            # If the page is full, create a new page for future events / other threads to use
            if len(current_write_page) == self.cache_size:
                self._write_page_cache = []

        if event.id is not None:
            # Write the event to the store - this can take some time
            event_json = json.dumps(data)
            filename = self._get_filename_for_id(event.id, self.user_id)
            if len(event_json) > 1_000_000:  # Roughly 1MB in bytes, ignoring encoding
                logger.warning(
                    f'Saving event JSON over 1MB: {len(event_json):,} bytes, filename: {filename}',
                    extra={
                        'user_id': self.user_id,
                        'session_id': self.sid,
                        'size': len(event_json),
                    },
                )
            self.file_store.write(filename, event_json)

            # Store the cache page last - if it is not present during reads then it will simply be bypassed.
            self._store_cache_page(current_write_page)
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
            event = None
            try:
                event = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # pass each event to each callback in order
            for key in sorted(self._subscribers.keys()):
                callbacks = self._subscribers[key]
                # Create a copy of the keys to avoid "dictionary changed size during iteration" error
                callback_ids = list(callbacks.keys())
                for callback_id in callback_ids:
                    # Check if callback_id still exists (might have been removed during iteration)
                    if callback_id in callbacks:
                        callback = callbacks[callback_id]
                        pool = self._thread_pools[key][callback_id]
                        future = pool.submit(callback, event)
                        future.add_done_callback(
                            self._make_error_handler(callback_id, key)
                        )

    def _make_error_handler(
        self, callback_id: str, subscriber_id: str
    ) -> Callable[[Any], None]:
        def _handle_callback_error(fut: Any) -> None:
            try:
                # This will raise any exception that occurred during callback execution
                fut.result()
            except Exception as e:
                logger.error(
                    f'Error in event callback {callback_id} for subscriber {subscriber_id}: {str(e)}',
                )
                # Re-raise in the main thread so the error is not swallowed
                raise e

        return _handle_callback_error
