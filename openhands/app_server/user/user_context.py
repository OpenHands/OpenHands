from abc import ABC, abstractmethod

from openhands.app_server.services.injector import Injector
from openhands.app_server.user.user_models import (
    UserInfo,
)
from openhands.integrations.provider import PROVIDER_TOKEN_TYPE, ProviderType
from openhands.sdk.secret import SecretSource
from openhands.sdk.utils.models import DiscriminatedUnionMixin


class UserContext(ABC):
    """Service for managing users."""

    # Read methods

    @abstractmethod
    async def get_user_id(self) -> str | None:
        """Get the user id"""

    @abstractmethod
    async def get_user_info(self) -> UserInfo:
        """Get the user info."""

    @abstractmethod
    async def get_authenticated_git_url(self, repository: str) -> str:
        """Get the provider tokens for the user"""

    @abstractmethod
    async def get_provider_tokens(self) -> PROVIDER_TOKEN_TYPE | None:
        """Get the latest tokens for all provider types"""

    @abstractmethod
    async def get_latest_token(self, provider_type: ProviderType) -> str | None:
        """Get the latest token for the provider type given"""

    @abstractmethod
    async def get_secrets(self) -> dict[str, SecretSource]:
        """Get custom secrets and github provider secrets for the conversation."""

    @abstractmethod
    async def get_mcp_api_key(self) -> str | None:
        """Get an MCP API Key."""


class UserContextInjector(DiscriminatedUnionMixin, Injector[UserContext], ABC):
    """Injector for user contexts."""

    pass
