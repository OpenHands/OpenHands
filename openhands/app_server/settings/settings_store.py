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

    The class is instantiated via get_impl() in openhands.app_server.shared.py.

    The implementation may or may not support multiple users depending on the environment.
    """

    @abstractmethod
    async def load(
        self,
        *,
        resolve_agent_profile: bool = False,
        override_agent_profile_id: str | None = None,
    ) -> Settings | None:
        """Load session init data.

        By default returns the PERSISTED (user-authored) settings — the view
        every load() -> store() round-trip must operate on.
        ``resolve_agent_profile=True`` opts into the *effective* launch view:
        the active Agent Profile (cloud-only concept) resolves and replaces
        ``agent_settings``; the result must never be passed to ``store()``.
        ``override_agent_profile_id`` is a one-off launch override and implies
        resolution. Implementations without the Agent Profile concept ignore
        both.
        """

    @abstractmethod
    async def store(self, settings: Settings) -> None:
        """Store session init data."""

    @classmethod
    @abstractmethod
    async def get_instance(cls, user_id: str | None) -> SettingsStore:
        """Get a store for the user represented by the token given.

        TODO: This method should be replaced with dependency injection.
        """

    @abstractmethod
    async def get_org_marketplaces(self, user_id: str | None) -> list[dict]:
        """Get organization-level marketplaces.

        Returns:
            List of marketplace dictionaries from the org's registered_marketplaces.
            Returns empty list if no org marketplaces are available (OSS mode) or if user_id is None.
        """
