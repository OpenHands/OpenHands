"""HTTP Client Protocol for Git Service Integrations."""

from abc import ABC, abstractmethod
from typing import Any

from httpx import AsyncClient, HTTPError, HTTPStatusError
from pydantic import SecretStr

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.service_types import (
    AuthenticationError,
    RateLimitError,
    RequestMethod,
    ResourceNotFoundError,
    UnknownException,
)


class HTTPClient(ABC):
    """Abstract base class defining the HTTP client interface for Git service integrations.

    This class abstracts the common HTTP client functionality needed by all
    Git service providers (GitHub, GitLab, Bitbucket, Azure DevOps) while keeping inheritance in place.
    """

    # Default attributes (subclasses may override)
    token: SecretStr = SecretStr('')
    refresh: bool = False
    external_auth_id: str | None = None
    external_auth_token: SecretStr | None = None
    external_token_manager: bool = False
    base_domain: str | None = None

    # Provider identification must be implemented by subclasses
    @property
    @abstractmethod
    def provider(self) -> str: ...

    # Abstract methods that concrete classes must implement
    @abstractmethod
    async def get_latest_token(self) -> SecretStr | None:
        """Get the latest working token for the service."""
        ...

    @abstractmethod
    async def _get_headers(self) -> dict[str, Any]:
        """Get HTTP headers for API requests."""
        ...

    @abstractmethod
    async def _make_request(
        self,
        url: str,
        params: dict | None = None,
        method: RequestMethod = RequestMethod.GET,
    ) -> tuple[Any, dict]:
        """Make an HTTP request to the Git service API."""
        ...

    def _has_token_expired(self, status_code: int) -> bool:
        """Check if the token has expired based on HTTP status code."""
        return status_code == 401

    async def execute_request(
        self,
        client: AsyncClient,
        url: str,
        headers: dict,
        params: dict | None,
        method: RequestMethod = RequestMethod.GET,
    ):
        """Execute an HTTP request using the provided client."""
        if method == RequestMethod.POST:
            return await client.post(url, headers=headers, json=params)
        return await client.get(url, headers=headers, params=params)

    def handle_http_status_error(
        self, e: HTTPStatusError
    ) -> (
        AuthenticationError | RateLimitError | ResourceNotFoundError | UnknownException
    ):
        """Handle HTTP status errors and convert them to appropriate exceptions."""
        if e.response.status_code == 401:
            return AuthenticationError(f'Invalid {self.provider} token')
        elif e.response.status_code == 404:
            return ResourceNotFoundError(
                f'Resource not found on {self.provider} API: {e}'
            )
        elif e.response.status_code == 429:
            logger.warning(f'Rate limit exceeded on {self.provider} API: {e}')
            return RateLimitError(f'{self.provider} API rate limit exceeded')

        logger.warning(f'Status error on {self.provider} API: {e}')
        return UnknownException(f'Unknown error: {e}')

    def handle_http_error(self, e: HTTPError) -> UnknownException:
        """Handle general HTTP errors."""
        logger.warning(f'HTTP error on {self.provider} API: {type(e).__name__} : {e}')
        return UnknownException(f'HTTP error {type(e).__name__} : {e}')
