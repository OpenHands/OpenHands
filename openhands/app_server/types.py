from abc import ABC, abstractmethod
from typing import Any

from openhands.app_server.config_api.config_models import AppMode


class ServerConfigInterface(ABC):
    @abstractmethod
    def verify_config(self) -> None:
        """Verify configuration settings."""
        raise NotImplementedError

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """Configure attributes for frontend"""
        raise NotImplementedError


class MissingSettingsError(ValueError):
    """Raised when settings are missing or not found."""

    pass


class LLMAuthenticationError(ValueError):
    """Raised when there is an issue with LLM authentication."""

    pass


class SessionExpiredError(ValueError):
    """Raised when the user's authentication session has expired."""

    pass


__all__ = [
    'AppMode',
    'ServerConfigInterface',
    'MissingSettingsError',
    'LLMAuthenticationError',
    'SessionExpiredError',
]
