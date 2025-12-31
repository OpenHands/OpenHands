from collections.abc import Callable
from typing import ClassVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from openhands.sdk.llm.llm import LLM
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)


class RegistryEvent(BaseModel):
    llm: LLM

    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True,
    )


class LLMRegistry:
    """A minimal LLM registry for managing LLM instances by usage ID.

    This registry provides a simple way to manage multiple LLM instances,
    avoiding the need to recreate LLMs with the same configuration.
    """

    registry_id: str
    retry_listener: Callable[[int, int], None] | None

    def __init__(
        self,
        retry_listener: Callable[[int, int], None] | None = None,
    ):
        """Initialize the LLM registry.

        Args:
            retry_listener: Optional callback for retry events.
        """
        self.registry_id = str(uuid4())
        self.retry_listener = retry_listener
        self._usage_to_llm: dict[str, LLM] = {}
        self.subscriber: Callable[[RegistryEvent], None] | None = None

    def subscribe(self, callback: Callable[[RegistryEvent], None]) -> None:
        """Subscribe to registry events.

        Args:
            callback: Function to call when LLMs are created or updated.
        """
        self.subscriber = callback

    def notify(self, event: RegistryEvent) -> None:
        """Notify subscribers of registry events.

        Args:
            event: The registry event to notify about.
        """
        if self.subscriber:
            try:
                self.subscriber(event)
            except Exception as e:
                logger.warning(f"Failed to emit event: {e}")

    @property
    def usage_to_llm(self) -> dict[str, LLM]:
        """Access the internal usage-ID-to-LLM mapping."""

        return self._usage_to_llm

    def add(self, llm: LLM) -> None:
        """Add an LLM instance to the registry.

        Args:
            llm: The LLM instance to register.

        Raises:
            ValueError: If llm.usage_id already exists in the registry.
        """
        usage_id = llm.usage_id
        if usage_id in self._usage_to_llm:
            message = (
                f"Usage ID '{usage_id}' already exists in registry. "
                "Use a different usage_id on the LLM or "
                "call get() to retrieve the existing LLM."
            )
            raise ValueError(message)

        self._usage_to_llm[usage_id] = llm
        self.notify(RegistryEvent(llm=llm))
        logger.debug(
            f"[LLM registry {self.registry_id}]: Added LLM for usage {usage_id}"
        )

    def get(self, usage_id: str) -> LLM:
        """Get an LLM instance from the registry.

        Args:
            usage_id: Unique identifier for the LLM usage slot.

        Returns:
            The LLM instance.

        Raises:
            KeyError: If usage_id is not found in the registry.
        """
        if usage_id not in self._usage_to_llm:
            raise KeyError(
                f"Usage ID '{usage_id}' not found in registry. "
                "Use add() to register an LLM first."
            )

        logger.info(
            f"[LLM registry {self.registry_id}]: Retrieved LLM for usage {usage_id}"
        )
        return self._usage_to_llm[usage_id]

    def list_usage_ids(self) -> list[str]:
        """List all registered usage IDs."""

        return list(self._usage_to_llm.keys())
