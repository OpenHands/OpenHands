import abc
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from openhands.agent_server.utils import utc_now


class CriticResult(BaseModel):
    """A critic result is a score and a message."""

    score: float
    message: str
    evaluated_at: datetime = Field(default_factory=utc_now)

    @property
    def success(self) -> bool:
        """Whether the agent is successful."""
        return self.score >= 0.5


class BaseCritic(abc.ABC):
    """A critic produces a `CriticResult` from a sequence of conversation events.

    The events list is intentionally untyped so critic implementations can be
    shared between the legacy V0 event stream (``openhands.events.Event``) and
    the V1 SDK event stream (``openhands.sdk.Event``).
    """

    @abc.abstractmethod
    def evaluate(self, events: list[Any], git_patch: str | None = None) -> CriticResult:
        pass
