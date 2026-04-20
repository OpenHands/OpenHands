from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any

from pydantic import BaseModel

from openhands.controller.state.state import State
from openhands.core.config.condenser_config import CondenserConfig
from openhands.core.logger import openhands_logger as logger
from openhands.events.action.agent import CondensationAction
from openhands.memory.view import View
from openhands.sdk.llm import LLMRegistry

CONDENSER_METADATA_KEY = 'condenser_meta'
"""Key identifying where metadata is stored in a `State` object's `extra_data` field."""


def get_condensation_metadata(state: State) -> list[dict[str, Any]]:
    """Utility function to retrieve a list of metadata batches from a `State`.

    Args:
        state: The state to retrieve metadata from.

    Returns:
        list[dict[str, Any]]: A list of metadata batches, each representing a condensation.
    """
    if CONDENSER_METADATA_KEY in state.extra_data:
        return state.extra_data[CONDENSER_METADATA_KEY]
    return []


CONDENSER_REGISTRY: dict[type[CondenserConfig], type[Condenser]] = {}
"""Registry of condenser configurations to their corresponding condenser classes."""


class Condensation(BaseModel):
    """Produced by a condenser to indicate the history has been condensed."""

    action: CondensationAction


class Condenser(ABC):
    """Abstract condenser interface.

    Condensers take a list of `Event` objects and reduce them into a potentially smaller list.

    Agents can use condensers to reduce the amount of events they need to consider when deciding which action to take. To use a condenser, agents can call the `condensed_history` method on the current `State` being considered and use the results instead of the full history.

    If the condenser returns a `Condensation` instead of a `View`, the agent should return `Condensation.action` instead of producing its own action. On the next agent step the condenser will use that condensation event to produce a new `View`.
    """

    def __init__(self):
        self._metadata_batch: dict[str, Any] = {}
        self._llm_metadata: dict[str, Any] = {}

    def add_metadata(self, key: str, value: Any) -> None:
        """Add information to the current metadata batch.

        Any key/value pairs added to the metadata batch will be recorded in the `State` at the end of the current condensation.

        Args:
            key: The key to store the metadata under.

            value: The metadata to store.
        """
        self._metadata_batch[key] = value

    def write_metadata(self, state: State) -> None:
        """Write the current batch of metadata to the `State`.

        Resets the current metadata batch: any metadata added after this call will be stored in a new batch and written to the `State` at the end of the next condensation.
        """
        if CONDENSER_METADATA_KEY not in state.extra_data:
            state.extra_data[CONDENSER_METADATA_KEY] = []
        if self._metadata_batch:
            state.extra_data[CONDENSER_METADATA_KEY].append(self._metadata_batch)

        # Since the batch has been written, clear it for the next condensation
        self._metadata_batch = {}

    @contextmanager
    def metadata_batch(self, state: State):
        """Context manager to ensure batched metadata is always written to the `State`."""
        try:
            yield
        finally:
            self.write_metadata(state)

    @abstractmethod
    def condense(self, View) -> View | Condensation:
        """Condense a sequence of events into a potentially smaller list.

        New condenser strategies should override this method to implement their own condensation logic. Call `self.add_metadata` in the implementation to record any relevant per-condensation diagnostic information.

        Args:
            View: A view of the history containing all events that should be condensed.

        Returns:
            View | Condensation: A condensed view of the events or an event indicating the history has been condensed.
        """

    def condensed_history(self, state: State) -> View | Condensation:
        """Condense the state's history."""
        if hasattr(self, 'llm'):
            model_name = self.llm.config.model
        else:
            model_name = 'unknown'

        self._llm_metadata = state.to_llm_metadata(
            model_name=model_name, agent_name='condenser'
        )
        with self.metadata_batch(state):
            return self.condense(state.view)

    @property
    def llm_metadata(self) -> dict[str, Any]:
        """Metadata to be passed to the LLM when using this condenser.

        This metadata is used to provide context about the condensation process and can be used by the LLM to understand how the history was condensed.
        """
        if not self._llm_metadata:
            logger.warning(
                'LLM metadata is empty. Ensure to set it in the condenser implementation.'
            )
        return self._llm_metadata

    @classmethod
    def register_config(cls, configuration_type: type[CondenserConfig]) -> None:
        """Register a new condenser configuration type.

        Instances of registered configuration types can be passed to `from_config` to create instances of the corresponding condenser.

        Args:
            configuration_type: The type of configuration used to create instances of the condenser.

        Raises:
            ValueError: If the configuration type is already registered.
        """
        if configuration_type in CONDENSER_REGISTRY:
            raise ValueError(
                f'Condenser configuration {configuration_type} is already registered'
            )
        CONDENSER_REGISTRY[configuration_type] = cls

    @classmethod
    def from_config(
        cls, config: CondenserConfig, llm_registry: LLMRegistry
    ) -> Condenser:
        """Create a condenser from a configuration object.

        Args:
            config: Configuration for the condenser.

        Returns:
            Condenser: A condenser instance.

        Raises:
            ValueError: If the condenser type is not recognized.
        """
        try:
            condenser_class = CONDENSER_REGISTRY[type(config)]
            return condenser_class.from_config(config, llm_registry)
        except KeyError:
            raise ValueError(f'Unknown condenser config: {config}')


class RollingCondenser(Condenser, ABC):
    """Base class for a specialized condenser strategy that applies condensation to a rolling history.

    The rolling history is generated by `View.from_events`, which analyzes all events in the history and produces a `View` object representing what will be sent to the LLM.

    If `should_condense` says so, the condenser is then responsible for generating a `Condensation` object from the `View` object. This will be added to the event history which should -- when given to `get_view` -- produce the condensed `View` to be passed to the LLM.
    """

    @abstractmethod
    def should_condense(self, view: View) -> bool:
        """Determine if a view should be condensed."""

    @abstractmethod
    def get_condensation(self, view: View) -> Condensation:
        """Get the condensation from a view."""

    def condense(self, view: View) -> View | Condensation:
        # If we trigger the condenser-specific condensation threshold, compute and return
        # the condensation.
        if self.should_condense(view):
            return self.get_condensation(view)

        # Otherwise we're safe to just return the view.
        else:
            return view
