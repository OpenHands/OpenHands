import os
from typing import Any

import httpx
from pydantic import SecretStr

from openhands.integrations.azure_devops.service.branches import (
    AzureDevOpsBranchesMixin,
)
from openhands.integrations.azure_devops.service.features import (
    AzureDevOpsFeaturesMixin,
)
from openhands.integrations.azure_devops.service.prs import AzureDevOpsPRsMixin
from openhands.integrations.azure_devops.service.repos import AzureDevOpsReposMixin
from openhands.integrations.azure_devops.service.resolver import (
    AzureDevOpsResolverMixin,
)
from openhands.integrations.azure_devops.service.work_items import (
    AzureDevOpsWorkItemsMixin,
)
from openhands.integrations.protocols.http_client import HTTPClient
from openhands.integrations.service_types import (
    BaseGitService,
    GitService,
    ProviderType,
    RequestMethod,
)
from openhands.utils.import_utils import get_impl


class AzureDevOpsServiceImpl(
    AzureDevOpsResolverMixin,
    AzureDevOpsReposMixin,
    AzureDevOpsBranchesMixin,
    AzureDevOpsPRsMixin,
    AzureDevOpsWorkItemsMixin,
    AzureDevOpsFeaturesMixin,
    BaseGitService,
    HTTPClient,
    GitService,
):
    """Azure DevOps service implementation using modular mixins.

    This class inherits functionality from specialized mixins:
    - AzureDevOpsResolverMixin: PR/work item comment resolution
    - AzureDevOpsReposMixin: Repository operations
    - AzureDevOpsBranchesMixin: Branch operations
    - AzureDevOpsPRsMixin: Pull request operations
    - AzureDevOpsWorkItemsMixin: Work item operations (unique to Azure DevOps)
    - AzureDevOpsFeaturesMixin: Microagents, suggested tasks, user info

    This is an extension point in OpenHands that allows applications to customize Azure DevOps
    integration behavior. Applications can substitute their own implementation by:
    1. Creating a class that inherits from GitService
    2. Implementing all required methods
    3. Setting OPENHANDS_AZURE_DEVOPS_SERVICE_CLS environment variable

    The class is instantiated via get_impl() at module load time.
    """

    token: SecretStr = SecretStr('')
    refresh = False
    organization: str = ''

    def __init__(
        self,
        user_id: str | None = None,
        external_auth_id: str | None = None,
        external_auth_token: SecretStr | None = None,
        token: SecretStr | None = None,
        external_token_manager: bool = False,
        base_domain: str | None = None,
    ):
        self.user_id = user_id
        self.external_token_manager = external_token_manager

        if token:
            self.token = token

        if base_domain:
            # Parse organization from base_domain
            # Strip URL prefix if present (e.g., "https://dev.azure.com/org" -> "org")
            domain_path = base_domain
            if '://' in domain_path:
                # Remove protocol and domain, keep only path
                domain_path = domain_path.split('://', 1)[1]
                if '/' in domain_path:
                    domain_path = domain_path.split('/', 1)[1]

            # Format expected: organization (e.g., "contoso")
            # Take first part only (in case user still enters org/project)
            parts = domain_path.split('/')
            if len(parts) >= 1:
                self.organization = parts[0]

    async def get_installations(self) -> list[str]:
        """Get Azure DevOps organizations.

        For Azure DevOps, 'installations' are equivalent to organizations.
        Since authentication is per-organization, return the current organization.
        """
        return [self.organization]

    @property
    def provider(self) -> str:
        return ProviderType.AZURE_DEVOPS.value

    @property
    def base_url(self) -> str:
        """Get the base URL for Azure DevOps API calls."""
        return f'https://dev.azure.com/{self.organization}'

    @staticmethod
    def _is_oauth_token(token: str) -> bool:
        """Check if a token is an OAuth JWT token (from SSO) vs a PAT.

        OAuth tokens from Azure AD/Entra ID are JWTs with the format:
        header.payload.signature (three base64url-encoded parts separated by dots)

        PATs are opaque tokens without this structure.

        Args:
            token: The token string to check

        Returns:
            True if the token appears to be a JWT (OAuth), False if it's a PAT
        """
        # JWTs have exactly 3 parts separated by dots
        parts = token.split('.')
        return len(parts) == 3 and all(len(part) > 0 for part in parts)

    async def _get_azure_devops_headers(self) -> dict[str, Any]:
        """Retrieve the Azure DevOps authentication headers.

        Supports two authentication methods:
        1. OAuth 2.0 (Bearer token) - Used for SSO/SaaS mode with Keycloak/Azure AD
        2. Personal Access Token (Basic auth) - Used for self-hosted mode

        The method automatically detects the token type:
        - OAuth tokens are JWTs (header.payload.signature format) -> uses Bearer auth
        - PATs are opaque strings -> uses Basic auth

        Returns:
            dict: HTTP headers with appropriate Authorization header
        """
        if not self.token:
            latest_token = await self.get_latest_token()
            if latest_token:
                self.token = latest_token

        token_value = self.token.get_secret_value()

        # Detect token type and use appropriate authentication method
        if self._is_oauth_token(token_value):
            # OAuth 2.0 access token from SSO (Azure AD/Keycloak broker)
            # Use Bearer authentication as per OAuth 2.0 spec
            auth_header = f'Bearer {token_value}'
        else:
            # Personal Access Token (PAT) for self-hosted deployments
            # Use Basic authentication with empty username and PAT as password
            import base64

            auth_str = base64.b64encode(f':{token_value}'.encode()).decode()
            auth_header = f'Basic {auth_str}'

        return {
            'Authorization': auth_header,
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        }

    async def _get_headers(self) -> dict[str, Any]:
        """Retrieve the Azure DevOps headers."""
        return await self._get_azure_devops_headers()

    def _has_token_expired(self, status_code: int) -> bool:
        return status_code == 401

    async def get_latest_token(self) -> SecretStr | None:
        return self.token

    async def _make_request(
        self,
        url: str,
        params: dict | None = None,
        method: RequestMethod = RequestMethod.GET,
    ) -> tuple[Any, dict]:
        try:
            async with httpx.AsyncClient() as client:
                azure_devops_headers = await self._get_azure_devops_headers()

                # Make initial request
                response = await self.execute_request(
                    client=client,
                    url=url,
                    headers=azure_devops_headers,
                    params=params,
                    method=method,
                )

                # Handle token refresh if needed
                if self.refresh and self._has_token_expired(response.status_code):
                    await self.get_latest_token()
                    azure_devops_headers = await self._get_azure_devops_headers()
                    response = await self.execute_request(
                        client=client,
                        url=url,
                        headers=azure_devops_headers,
                        params=params,
                        method=method,
                    )

                response.raise_for_status()
                headers = {}
                if 'Link' in response.headers:
                    headers['Link'] = response.headers['Link']

                return response.json(), headers

        except httpx.HTTPStatusError as e:
            raise self.handle_http_status_error(e)
        except httpx.HTTPError as e:
            raise self.handle_http_error(e)

    def _parse_repository(self, repository: str) -> tuple[str, str, str]:
        """Parse repository string into organization, project, and repo name.

        Args:
            repository: Repository string in format organization/project/repo

        Returns:
            Tuple of (organization, project, repo_name)
        """
        parts = repository.split('/')
        if len(parts) < 3:
            raise ValueError(
                f'Invalid repository format: {repository}. Expected format: organization/project/repo'
            )
        return parts[0], parts[1], parts[2]


# Dynamic class loading to support custom implementations (e.g., SaaS)
azure_devops_service_cls = os.environ.get(
    'OPENHANDS_AZURE_DEVOPS_SERVICE_CLS',
    'openhands.integrations.azure_devops.azure_devops_service.AzureDevOpsServiceImpl',
)
AzureDevOpsServiceImpl = get_impl(  # type: ignore[misc]
    AzureDevOpsServiceImpl, azure_devops_service_cls
)
