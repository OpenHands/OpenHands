import re


BASE_STATE = "base_state.json"
EVENTS_DIR = "events"
EVENT_NAME_RE = re.compile(
    r"^event-(?P<idx>\d{5})-(?P<event_id>[0-9a-fA-F\-]{8,})\.json$"
)
EVENT_FILE_PATTERN = "event-{idx:05d}-{event_id}.json"
