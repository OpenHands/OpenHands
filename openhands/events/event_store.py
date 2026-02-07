import json
from dataclasses import dataclass
from typing import Iterable

from openhands.core.logger import openhands_logger as logger
from openhands.events.event import Event, EventSource
from openhands.events.event_filter import EventFilter
from openhands.events.event_store_abc import EventStoreABC
from openhands.events.serialization.event import event_from_dict
from openhands.storage.files import FileStore
from openhands.storage.locations import (
    get_conversation_dir,
    get_conversation_event_count_filename,
    get_conversation_event_filename,
    get_conversation_events_dir,
)
from openhands.utils.shutdown_listener import should_continue


@dataclass(frozen=True)
class _CachePage:
    events: list[dict] | None
    start: int
    end: int

    def covers(self, global_index: int) -> bool:
        if global_index < self.start:
            return False
        if global_index >= self.end:
            return False
        return True

    def get_event(self, global_index: int) -> Event | None:
        # If there was not actually a cached page, return None
        if not self.events:
            return None
        local_index = global_index - self.start
        return event_from_dict(self.events[local_index])


_DUMMY_PAGE = _CachePage(None, 1, -1)


@dataclass
class EventStore(EventStoreABC):
    """A stored list of events backing a conversation"""

    sid: str
    file_store: FileStore
    user_id: str | None
    cache_size: int = 25
    _cur_id: int | None = None  # Private field to cache the calculated value

    @property
    def cur_id(self) -> int:
        """Lazy calculated property for the current event ID."""
        if self._cur_id is None:
            self._cur_id = self._calculate_cur_id()
        return self._cur_id

    @cur_id.setter
    def cur_id(self, value: int) -> None:
        """Setter for cur_id to allow updates."""
        self._cur_id = value

    def _calculate_cur_id(self) -> int:
        """Calculate the current event ID.

        First tries to read from the persisted event_count metadata file
        (O(1)), falling back to the O(N) directory scan for backward
        compatibility with sessions that pre-date the metadata file.
        """
        # Fast path: read from metadata file
        try:
            count_filename = get_conversation_event_count_filename(
                self.sid, self.user_id
            )
            content = self.file_store.read(count_filename)
            count = int(content.strip())
            if count >= 0:
                return count
        except (FileNotFoundError, ValueError):
            pass

        # Slow path: scan the events directory
        events = []
        try:
            events_dir = get_conversation_events_dir(self.sid, self.user_id)
            events = self.file_store.list(events_dir)
        except FileNotFoundError:
            logger.debug(f'No events found for session {self.sid}')

        if not events:
            return 0

        # if we have events, we need to find the highest id to prepare for new events
        max_id = -1
        for event_str in events:
            id = self._get_id_from_filename(event_str)
            if id >= max_id:
                max_id = id
        return max_id + 1

    def search_events(
        self,
        start_id: int = 0,
        end_id: int | None = None,
        reverse: bool = False,
        filter: EventFilter | None = None,
        limit: int | None = None,
    ) -> Iterable[Event]:
        """Retrieve events from the event stream, optionally filtering out events of a given type
        and events marked as hidden.

        Args:
            start_id: The ID of the first event to retrieve. Defaults to 0.
            end_id: The ID of the last event to retrieve. Defaults to the last event in the stream.
            reverse: Whether to retrieve events in reverse order. Defaults to False.
            filter: EventFilter to use

        Yields:
            Events from the stream that match the criteria.
        """
        if end_id is None:
            end_id = self.cur_id
        else:
            end_id += 1  # From inclusive to exclusive

        if reverse:
            step = -1
            start_id, end_id = end_id, start_id
            start_id -= 1
            end_id -= 1
        else:
            step = 1

        cache_page = _DUMMY_PAGE
        num_results = 0
        for index in range(start_id, end_id, step):
            if not should_continue():
                return
            if not cache_page.covers(index):
                cache_page = self._load_cache_page_for_index(index)
            event = cache_page.get_event(index)
            if event is None:
                try:
                    event = self.get_event(index)
                except FileNotFoundError:
                    event = None
            if event:
                if not filter or filter.include(event):
                    yield event
                    num_results += 1
                    if limit and limit <= num_results:
                        return

    def get_event(self, id: int) -> Event:
        filename = self._get_filename_for_id(id, self.user_id)
        content = self.file_store.read(filename)
        data = json.loads(content)
        return event_from_dict(data)

    def get_latest_event(self) -> Event:
        return self.get_event(self.cur_id - 1)

    def get_latest_event_id(self) -> int:
        return self.cur_id - 1

    def filtered_events_by_source(self, source: EventSource) -> Iterable[Event]:
        for event in self.search_events():
            if event.source == source:
                yield event

    def _get_filename_for_id(self, id: int, user_id: str | None) -> str:
        return get_conversation_event_filename(self.sid, id, user_id)

    def _get_filename_for_cache(self, start: int, end: int) -> str:
        return f'{get_conversation_dir(self.sid, self.user_id)}event_cache/{start}-{end}.json'

    def _load_cache_page(self, start: int, end: int) -> _CachePage:
        """Read a page from the cache. Reading individual events is slow when there are a lot of them, so we use pages."""
        cache_filename = self._get_filename_for_cache(start, end)
        try:
            content = self.file_store.read(cache_filename)
            events = json.loads(content)
        except FileNotFoundError:
            events = None
        page = _CachePage(events, start, end)
        return page

    def _load_cache_page_for_index(self, index: int) -> _CachePage:
        offset = index % self.cache_size
        index -= offset
        return self._load_cache_page(index, index + self.cache_size)

    @staticmethod
    def _get_id_from_filename(filename: str) -> int:
        try:
            return int(filename.split('/')[-1].split('.')[0])
        except ValueError:
            logger.warning(f'get id from filename ({filename}) failed.')
            return -1
