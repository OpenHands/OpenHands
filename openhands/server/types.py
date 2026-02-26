# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
# This module belongs to the old V0 web server. The V1 application server lives under openhands/app_server/.
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, ClassVar, Protocol


class AppMode(Enum):
    OPENHANDS = 'oss'
    SAAS = 'saas'

    # Backwards-compatible alias (deprecated): prefer AppMode.OPENHANDS
    OSS = 'oss'


class SessionMiddlewareInterface(Protocol):
    """Protocol for session middleware classes."""

    pass


class ServerConfigInterface(ABC):
    CONFIG_PATH: ClassVar[str | None]
    APP_MODE: ClassVar[AppMode]
    POSTHOG_CLIENT_KEY: ClassVar[str]
    GITHUB_CLIENT_ID: ClassVar[str]
    ATTACH_SESSION_MIDDLEWARE_PATH: ClassVar[str]

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
