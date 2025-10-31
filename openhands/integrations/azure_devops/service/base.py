from abc import abstractmethod
from typing import Any
from urllib.parse import quote

from pydantic import SecretStr

from openhands.integrations.protocols.http_client import HTTPClient
from openhands.integrations.service_types import (
    BaseGitService,
    RequestMethod,
)


class AzureDevOpsMixinBase(BaseGitService, HTTPClient):
    """Declares common attributes and method signatures used across Azure DevOps mixins."""

    organization: str

    @property
    @abstractmethod
    def base_url(self) -> str:
        """Get the base URL for Azure DevOps API calls."""
        ...

    async def _get_headers(self) -> dict:
        """Retrieve the Azure DevOps token from settings store to construct the headers."""
        if not self.token:
            latest_token = await self.get_latest_token()
            if latest_token:
                self.token = latest_token

        return {
            'Authorization': f'Bearer {self.token.get_secret_value() if self.token else ""}',
            'Content-Type': 'application/json',
        }

    async def get_latest_token(self) -> SecretStr | None:  # type: ignore[override]
        return self.token

    async def _make_request(
        self,
        url: str,
        params: dict | None = None,
        method: RequestMethod = RequestMethod.GET,
    ) -> tuple[Any, dict]:  # type: ignore[override]
        """Make HTTP request to Azure DevOps API."""
        raise NotImplementedError('Implemented in AzureDevOpsServiceImpl')

    def _parse_repository(self, repository: str) -> tuple[str, str, str]:
        """Parse repository string into organization, project, and repo name."""
        raise NotImplementedError('Implemented in AzureDevOpsServiceImpl')

    def _truncate_comment(self, comment: str, max_length: int = 1000) -> str:
        """Truncate comment to max length."""
        raise NotImplementedError('Implemented in AzureDevOpsServiceImpl')

    @staticmethod
    def _encode_url_component(component: str) -> str:
        """URL-encode a component for use in Azure DevOps API URLs.

        Args:
            component: The string component to encode (e.g., repo name, project name, org name)

        Returns:
            URL-encoded string with spaces and special characters properly encoded
        """
        return quote(component, safe='')
