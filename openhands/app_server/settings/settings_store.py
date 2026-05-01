from __future__ import annotations

from abc import ABC, abstractmethod

from openhands.app_server.settings.settings_models import Settings


class SettingsStore(ABC):
    """Abstract base class for storing user settings.

    This is an extension point in OpenHands that allows applications to customize how
    user settings are stored. Applications can substitute their own implementation by:
    1. Creating a class that inherits from SettingsStore
    2. Implementing all required methods
    3. Setting server_config.settings_store_class to the fully qualified name of the class

    The class is instantiated via get_impl() in openhands.server.shared.py.

    The implementation may or may not support multiple users depending on the environment.
    """

    @abstractmethod
    async def load(self) -> Settings | None:
        """Load session init data."""

    @abstractmethod
    async def store(self, settings: Settings) -> None:
        """Store session init data."""

    @classmethod
    @abstractmethod
    async def get_instance(cls, user_id: str | None) -> SettingsStore:
        """Get a store for the user represented by the token given.

        TODO: This method should be replaced with dependency injection.
        """
