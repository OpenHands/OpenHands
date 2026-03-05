from dataclasses import dataclass, field

from openhands.core.schema import ObservationType
from openhands.events.observation.observation import Observation


@dataclass
class WarpGrepObservation(Observation):
    """Result of a WarpGrep codebase search operation."""

    observation: str = ObservationType.WARPGREP_SEARCH
    query: str = ''
    results: list[dict] = field(default_factory=list)

    @property
    def message(self) -> str:
        return self.content
