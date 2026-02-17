from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from openhands.events.tool import ToolCallMetadata
from openhands.llm.metrics import Metrics


class EventSource(str, Enum):
    AGENT = 'agent'
    USER = 'user'
    ENVIRONMENT = 'environment'


class FileEditSource(str, Enum):
    LLM_BASED_EDIT = 'llm_based_edit'
    OH_ACI = 'oh_aci'  # openhands-aci


class FileReadSource(str, Enum):
    OH_ACI = 'oh_aci'  # openhands-aci
    DEFAULT = 'default'


@dataclass
class Event:
    INVALID_ID = -1

    @property
    def message(self) -> str | None:
        if hasattr(self, '_message'):
            msg = getattr(self, '_message')
            return str(msg) if msg is not None else None
        return ''

    @property
    def id(self) -> int:
        if hasattr(self, '_id'):
            id_val = getattr(self, '_id')
            return int(id_val) if id_val is not None else Event.INVALID_ID
        return Event.INVALID_ID

    @property
    def timestamp(self) -> str | None:
        if hasattr(self, '_timestamp') and isinstance(self._timestamp, str):
            ts = getattr(self, '_timestamp')
            return str(ts) if ts is not None else None
        return None

    @timestamp.setter
    def timestamp(self, value: datetime) -> None:
        if isinstance(value, datetime):
            self._timestamp = value.isoformat()

    @property
    def source(self) -> EventSource | None:
        if hasattr(self, '_source'):
            src = getattr(self, '_source')
            return EventSource(src) if src is not None else None
        return None

    @property
    def cause(self) -> int | None:
        if hasattr(self, '_cause'):
            cause_val = getattr(self, '_cause')
            return int(cause_val) if cause_val is not None else None
        return None

    @property
    def timeout(self) -> float | None:
        if hasattr(self, '_timeout'):
            timeout_val = getattr(self, '_timeout')
            return float(timeout_val) if timeout_val is not None else None
        return None

    def set_hard_timeout(self, value: float | None, blocking: bool = True) -> None:
        """Set the timeout for the event.

        NOTE, this is a hard timeout, meaning that the event will be blocked
        until the timeout is reached.
        """
        self._timeout = value
        # Check if .blocking is an attribute of the event
        if hasattr(self, 'blocking'):
            # .blocking needs to be set to True if .timeout is set
            self.blocking = blocking

    # optional metadata, LLM call cost of the edit
    @property
    def llm_metrics(self) -> Metrics | None:
        if hasattr(self, '_llm_metrics'):
            metrics = getattr(self, '_llm_metrics')
            return metrics if isinstance(metrics, Metrics) else None
        return None

    @llm_metrics.setter
    def llm_metrics(self, value: Metrics) -> None:
        self._llm_metrics = value

    # optional field, metadata about the tool call, if the event has a tool call
    @property
    def tool_call_metadata(self) -> ToolCallMetadata | None:
        if hasattr(self, '_tool_call_metadata'):
            metadata = getattr(self, '_tool_call_metadata')
            return metadata if isinstance(metadata, ToolCallMetadata) else None
        return None

    @tool_call_metadata.setter
    def tool_call_metadata(self, value: ToolCallMetadata) -> None:
        self._tool_call_metadata = value

    # optional field, the id of the response from the LLM
    @property
    def response_id(self) -> str | None:
        if hasattr(self, '_response_id'):
            return self._response_id  # type: ignore[attr-defined]
        return None

    @response_id.setter
    def response_id(self, value: str) -> None:
        self._response_id = value
