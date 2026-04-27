from __future__ import annotations

from abc import ABC, abstractmethod

from openhands.core.config.openhands_config import OpenHandsConfig
from openhands.storage.data_models.secrets import Secrets


class SecretsStore(ABC):
    """Abstract base class for storing user secrets.

    This is an extension point in OpenHands that allows applications to customize how
    user secrets are stored. Applications can substitute their own implementation by:
    1. Creating a class that inherits from SecretsStore
    2. Implementing all required methods
    3. Setting server_config.secret_store_class to the fully qualified name of the class

    The class is instantiated via get_impl() in openhands.server.shared.py.

    The implementation may or may not support multiple users depending on the environment.
    """

    @abstractmethod
    async def load(self) -> Secrets | None:
        """Load secrets."""

    @abstractmethod
    async def store(self, secrets: Secrets) -> None:
        """Store secrets."""

    @classmethod
    @abstractmethod
    async def get_instance(
        cls, config: OpenHandsConfig, user_id: str | None
    ) -> SecretsStore:
        """Get a store for the user represented by the token given."""
