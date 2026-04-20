from __future__ import annotations

from openhands.core.config.condenser_config import BrowserOutputCondenserConfig
from openhands.events.event import Event
from openhands.events.observation import BrowserOutputObservation
from openhands.events.observation.agent import AgentCondensationObservation
from openhands.memory.condenser.condenser import Condensation, Condenser, View
from openhands.sdk.llm import LLMRegistry


class BrowserOutputCondenser(Condenser):
    """A condenser that masks the observations from browser outputs outside of a recent attention window.

    The intent here is to mask just the browser outputs and leave everything else untouched. This is important because currently we provide screenshots and accessibility trees as input to the model for browser observations. These are really large and consume a lot of tokens without any benefits in performance. So we want to mask all such observations from all previous timesteps, and leave only the most recent one in context.
    """

    def __init__(self, attention_window: int = 1):
        self.attention_window = attention_window
        super().__init__()

    def condense(self, view: View) -> View | Condensation:
        """Replace the content of browser observations outside of the attention window with a placeholder."""
        results: list[Event] = []
        cnt: int = 0
        for event in reversed(view):
            if (
                isinstance(event, BrowserOutputObservation)
                and cnt >= self.attention_window
            ):
                results.append(
                    AgentCondensationObservation(
                        f'Visited URL {event.url}\nContent omitted'
                    )
                )
            else:
                results.append(event)
                if isinstance(event, BrowserOutputObservation):
                    cnt += 1

        return View(events=list(reversed(results)))

    @classmethod
    def from_config(
        cls, config: BrowserOutputCondenserConfig, llm_registry: LLMRegistry
    ) -> BrowserOutputCondenser:
        return BrowserOutputCondenser(**config.model_dump(exclude={'type'}))


BrowserOutputCondenser.register_config(BrowserOutputCondenserConfig)
