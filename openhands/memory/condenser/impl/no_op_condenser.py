from __future__ import annotations

from openhands.core.config.condenser_config import NoOpCondenserConfig
from openhands.memory.condenser.condenser import Condensation, Condenser, View
from openhands.sdk.llm import LLMRegistry


class NoOpCondenser(Condenser):
    """A condenser that does nothing to the event sequence."""

    def condense(self, view: View) -> View | Condensation:
        """Returns the list of events unchanged."""
        return view

    @classmethod
    def from_config(
        cls, config: NoOpCondenserConfig, llm_registry: LLMRegistry
    ) -> NoOpCondenser:
        return NoOpCondenser()


NoOpCondenser.register_config(NoOpCondenserConfig)
