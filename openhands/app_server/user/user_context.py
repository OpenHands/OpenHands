from abc import ABC, abstractmethod

from openhands.app_server.integrations.provider import (
    PROVIDER_TOKEN_TYPE,
    ProviderHandler,
    ProviderType,
)
from openhands.app_server.integrations.service_types import UserGitInfo
from openhands.app_server.services.injector import Injector
from openhands.app_server.user.user_models import (
    UserInfo,
)
from openhands.sdk.secret import SecretSource
from openhands.sdk.utils.models import DiscriminatedUnionMixin


class UserContext(ABC):
    """Service for managing users."""

    # Read methods

    @abstractmethod
    async def get_user_id(self) -> str | None:
        """Get the user id"""

    @abstractmethod
    async def get_user_email(self) -> str | None:
        """Get the email for the current user, if available.

        Returns the user's email address for attribution in observability
        traces (e.g. Laminar). In SaaS/enterprise mode this is typically
        the Keycloak email; in OSS mode or for admin-scoped contexts this
        returns ``None`` so callers can fall back to the internal user id.

        Note: this value is considered PII and may be forwarded to
        third-party observability services. Treat it accordingly when
        adding new callers.
        """

    @abstractmethod
    async def get_user_info(self) -> UserInfo:
        """Get the user info."""

    @abstractmethod
    async def get_authenticated_git_url(
        self, repository: str, is_optional: bool = False
    ) -> str:
        """Get an authenticated git URL for a repository.

        Args:
            repository: Repository name (owner/repo)
            is_optional: If True, logs at debug level instead of error level
                when repository is not found. Use for optional repositories.
        """

    @abstractmethod
    async def get_provider_tokens(
        self, as_env_vars: bool = False
    ) -> PROVIDER_TOKEN_TYPE | dict[str, str] | None:
        """Get the latest tokens for all provider types.

        Args:
            as_env_vars: When True, return a ``dict[str, str]`` mapping env
                var names (e.g. ``github_token``) to plain-text token values.
                When False (default), return the raw provider token mapping.
        """

    @abstractmethod
    async def get_latest_token(self, provider_type: ProviderType) -> str | None:
        """Get the latest token for the provider type given"""

    @abstractmethod
    async def get_secrets(self) -> dict[str, SecretSource]:
        """Get custom secrets and github provider secrets for the conversation."""

    @abstractmethod
    async def get_mcp_api_key(self) -> str | None:
        """Get an MCP API Key."""

    @abstractmethod
    async def get_user_git_info(self) -> UserGitInfo | None:
        """Get an User Meta"""

    @abstractmethod
    async def get_default_sandbox_spec_id(self) -> str | None:
        """Get the user's preferred default sandbox spec ID, or None to use the global default."""

    async def get_provider_handler(self) -> ProviderHandler:
        """Get a ProviderHandler bound to this user's provider tokens.

        Not all contexts can build one (e.g. admin-scoped contexts without
        provider tokens). Such contexts leave this unimplemented; callers are
        expected to degrade gracefully when it raises.
        """
        raise NotImplementedError

    async def get_max_concurrent_sandboxes(self, default: int = 10) -> int:
        """Get the user's maximum concurrent sandboxes limit.

        This method returns the effective limit for concurrent sandboxes for the user.
        The resolution order is:
        1. User-specific override (if set)
        2. Organization default (if in enterprise/SaaS mode)
        3. The provided default value (OSS mode fallback)

        Args:
            default: The fallback limit if no user/org-specific limit is set.

        Returns:
            The effective maximum number of concurrent sandboxes allowed.
        """
        return default


class UserContextInjector(DiscriminatedUnionMixin, Injector[UserContext], ABC):
    """Injector for user contexts."""

    pass
