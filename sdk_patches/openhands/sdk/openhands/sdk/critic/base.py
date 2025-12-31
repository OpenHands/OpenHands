import abc
from collections.abc import Sequence
from typing import ClassVar

from pydantic import BaseModel, Field

from openhands.sdk.event import LLMConvertibleEvent
from openhands.sdk.utils.models import DiscriminatedUnionMixin


class CriticResult(BaseModel):
    """A critic result is a score and a message."""

    THRESHOLD: ClassVar[float] = 0.5

    score: float = Field(
        description="A predicted probability of success between 0 and 1.",
        ge=0.0,
        le=1.0,
    )
    message: str | None = Field(description="An optional message explaining the score.")

    @property
    def success(self) -> bool:
        """Whether the agent is successful."""
        return self.score >= CriticResult.THRESHOLD


class CriticBase(DiscriminatedUnionMixin, abc.ABC):
    """A critic is a function that takes in a list of events,
    optional git patch, and returns a score about the quality of agent's action.
    """

    @abc.abstractmethod
    def evaluate(
        self, events: Sequence[LLMConvertibleEvent], git_patch: str | None = None
    ) -> CriticResult:
        pass
