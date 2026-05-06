import base64
import re
from typing import Any

import httpx
from pydantic import SecretStr

from openhands.app_server.integrations.protocols.http_client import HTTPClient
from openhands.app_server.integrations.service_types import (
    BaseGitService,
    OwnerType,
    ProviderType,
    Repository,
    RequestMethod,
    User,
)
from openhands.app_server.utils.http_session import httpx_verify_option
from openhands.app_server.utils.logger import openhands_logger as logger


def _is_atlassian_api_token(token_value: str) -> bool:
    """Detect Atlassian API tokens that require Basic auth (not Bearer).

    Atlassian API tokens from https://id.atlassian.com/manage-profile/security
    are exactly 24 lowercase letters and digits. They do NOT contain colons,
    so the naive ':' check routes them to Bearer auth -- which Bitbucket Cloud
    rejects for these tokens.

    Regular Bitbucket personal access tokens (created in Bitbucket settings) are
    typically longer and contain uppercase letters and/or dashes, so they are
    correctly routed to Bearer auth.
    """
    return bool(re.fullmatch(r'[a-z0-9]{24}', token_value))


class BitBucketMixinBase(BaseGitService, HTTPClient):
    """
    Base mixin for BitBucket service containing common functionality
    """

    BASE_URL = 'https://api.bitbucket.org/2.0'

    @staticmethod
    def _resolve_primary_email(emails: list[dict]) -> str | None:
        """Find the primary confirmed email from a list of Bitbucket email objects.

        Bitbucket's /user/emails endpoint returns objects with
        'email', 'is_primary', and 'is_confirmed' keys.
        """
        for entry in emails:
            if entry.get('is_primary') and entry.get('is_confirmed'):
                return entry.get('email')
        return None

    def _extract_owner_and_repo(self, repository: str) -> tuple[str, str]:
        """Extract owner and repo from repository string.

        Args:
            repository: Repository name in format 'workspace/repo_slug'

        Returns:
            Tuple of (owner, repo)

        Raises:
            ValueError: If repository format is invalid
        """
        parts = repository.split('/')
        if len(parts) < 2:
            raise ValueError(f'Invalid repository name: {repository}')

        return parts[-2], parts[-1]

    async def get_latest_token(self) -> SecretStr | None:
        """Get latest working token of the user."""
        return self.token

    def _has_token_expired(self, status_code: int) -> bool:
        return status_code == 401

    async def _get_headers(self) -> dict[str, str]:
        """Get headers for Bitbucket API requests.

        Auth method selection:
        - Token contains ':'  -> Basic auth (Bitbucket App Password format: username:token)
        - Token matches Atlassian API token pattern (24 lowercase chars, no colon)
                              -> Basic auth with 'bitbucket.org' as username
                                (Atlassian tokens from id.atlassian.com must use Basic)
        - Otherwise          -> Bearer auth (Bitbucket personal access tokens)
        """
        if not self.token or not self.token.get_secret_value():
            latest_token = await self.get_latest_token()
            if latest_token:
                self.token = latest_token

        token_value = self.token.get_secret_value()

        # Atlassian API tokens (24 lowercase chars, no colon) require Basic auth
        # even though they don't contain a ':' separator.
        if _is_atlassian_api_token(token_value):
            auth_str = base64.b64encode(f'bitbucket.org:{token_value}'.encode()).decode()
            return {
                'Authorization': f'Basic {auth_str}',
                'Accept': 'application/json',
            }

        # Check if the token contains a colon, which indicates it's in username:password format
        if ':' in token_value:
            auth_str = base64.b64encode(token_value.encode()).decode()
            return {
                'Authorization': f'Basic {auth_str}',
                'Accept': 'application/json',
            }

        return {
            'Authorization': f'Bearer {token_value}',
            'Accept': 'application/json',
        }

    async def _make_request(
        self,
        url: str,
        params: dict | None = None,
        method: RequestMethod = RequestMethod.GET,
    ) -> tuple[Any, dict]:
        """Make a request to the Bitbucket API.

        Args:
            url: The URL to request
            params: Optional parameters for the request
            method: The HTTP method to use

        Returns:
            A tuple of (response_data, response_headers)

        """
        try:
            async with httpx.AsyncClient(verify=httpx_verify_option()) as client:
                bitbucket_headers = await self._get_headers()
                response = await self.execute_request(
                    client, url, bitbucket_headers, params, method
                )
                if self.refresh and self._has_token_expired(response.status_code):
                    await self.get_latest_token()
                    bitbucket_headers = await self._get_headers()
                    response = await self.execute_request(
                        client=client,
                        url=url,
                        headers=bitbucket_headers,
                        params=params,
                        method=method,
                    )
                response.raise_for_status()
                return response.json(), dict(response.headers)
        except httpx.HTTPError as e:
            raise self.handle_http_error(e)

    async def _fetch_paginated_data(
        self, url: str, params: dict, max_items: int
    ) -> list[dict]:
        """Fetch data with pagination support for Bitbucket API.

        Args:
            url: The API endpoint URL
            params: Query parameters for the request
            max_items: Maximum number of items to fetch

        Returns:
            List of data items from all pages
        """
        all_items: list[dict] = []
        current_url = url

        while current_url and len(all_items) < max_items:
            response, _ = await self._make_request(current_url, params)

            # Extract items from response
            page_items = response.get('values', [])
            all_items.extend(page_items)

            # Get next page URL from response
            current_url = response.get('next')

            # Clear params for subsequent requests as they're included in the next URL
            params = {}

        return all_items[:max_items]

    async def get_user_emails(self) -> list[dict]:
        """Fetch the authenticated user's email addresses from Bitbucket.

        Calls GET /user/emails which returns a paginated response with a
        'values' list of email objects containing 'email', 'is_primary',
        and 'is_confirmed' fields.
        """
        url = f'{self.BASE_URL}/user/emails'
        response, _ = await self._make_request(url)
        return response.get('values', [])

    async def get_user(self) -> User:
        """Get the authenticated user's information."""
        url = f'{self.BASE_URL}/user'
        data, _ = await self._make_request(url)

        account_id = data.get('account_id', '')

        email = None
        try:
            emails = await self.get_user_emails()
            email = self._resolve_primary_email(emails)
        except Exception:
            logger.warning(
                'bitbucket:get_user:email_fallback_failed',
                exc_info=True,
            )

        return User(
            id=account_id,
            login=data.get('username', ''),
            avatar_url=data.get('links', {}).get('avatar', {}).get('href', ''),
            name=data.get('display_name'),
            email=email,
        )

    def _parse_repository(
        self, repo: dict, link_header: str | None = None
    ) -> Repository:
        """Parse a Bitbucket API repository response into a Repository object.

        Args:
            repo: The API response data
            link_header: Optional pagination Link header

        Returns:
            A Repository object
        """
        repo_id = repo.get('uuid', '')

        workspace_slug = repo.get('workspace', {}).get('slug', '')
        repo_slug = repo.get('slug', '')
        full_name = (
            f'{workspace_slug}/{repo_slug}' if workspace_slug and repo_slug else ''
        )

        is_public = not repo.get('is_private', True)
        owner_type = OwnerType.ORGANIZATION
        main_branch = repo.get('mainbranch', {}).get('name')

        return Repository(
            id=repo_id,
            full_name=full_name,  # type: ignore[arg-type]
            git_provider=ProviderType.BITBUCKET,
            is_public=is_public,
            stargazers_count=None,  # Bitbucket doesn't have stars
            owner_type=owner_type,
            main_branch=main_branch,
        )