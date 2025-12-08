import base64
from typing import Any

import httpx
from pydantic import SecretStr

from openhands.integrations.protocols.http_client import HTTPClient
from openhands.integrations.service_types import (
    BaseGitService,
    OwnerType,
    ProviderType,
    Repository,
    RequestMethod,
    ResourceNotFoundError,
    User,
)
from openhands.utils.http_session import httpx_verify_option


class BitBucketMixinBase(BaseGitService, HTTPClient):
    """
    Base mixin for BitBucket service containing common functionality
    """

    BASE_URL = 'https://api.bitbucket.org/2.0'

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
        """Get headers for Bitbucket API requests."""
        token_value = self.token.get_secret_value()

        # Check if the token contains a colon, which indicates it's in username:password format
        if ':' in token_value:
            auth_str = base64.b64encode(token_value.encode()).decode()
            return {
                'Authorization': f'Basic {auth_str}',
                'Accept': 'application/json',
            }
        else:
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
        except httpx.HTTPStatusError as e:
            raise self.handle_http_status_error(e)
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

    async def get_user(self) -> User:
        """Get the authenticated user's information."""
        url = f'{self.BASE_URL}/user'
        data, _ = await self._make_request(url)

        account_id = data.get('account_id', '')

        return User(
            id=account_id,
            login=data.get('username', ''),
            avatar_url=data.get('links', {}).get('avatar', {}).get('href', ''),
            name=data.get('display_name'),
            email=None,  # Bitbucket API doesn't return email in this endpoint
        )

    def _parse_repository(
        self, repo: dict, link_header: str | None = None
    ) -> Repository:
        """Parse a Bitbucket API repository response into a Repository object.

        Args:
            repo: Repository data from Bitbucket API
            link_header: Optional link header for pagination

        Returns:
            Repository object
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
            pushed_at=repo.get('updated_on'),
            owner_type=owner_type,
            link_header=link_header,
            main_branch=main_branch,
        )

    async def get_repository_details_from_repo_name(
        self, repository: str
    ) -> Repository:
        """Get repository details from repository name.

        Args:
            repository: Repository name in format 'workspace/repo_slug'

        Returns:
            Repository object with details
        """
        url = f'{self.BASE_URL}/repositories/{repository}'
        data, _ = await self._make_request(url)
        return self._parse_repository(data)

    async def _get_cursorrules_url(self, repository: str) -> str:
        """Get the URL for checking .cursorrules file."""
        # Get repository details to get the main branch
        repo_details = await self.get_repository_details_from_repo_name(repository)
        if not repo_details.main_branch:
            raise ResourceNotFoundError(
                f'Main branch not found for repository {repository}. '
                f'This repository may be empty or have no default branch configured.'
            )
        return f'{self.BASE_URL}/repositories/{repository}/src/{repo_details.main_branch}/.cursorrules'

    async def _get_microagents_directory_url(
        self, repository: str, microagents_path: str
    ) -> str:
        """Get the URL for checking microagents directory."""
        # Get repository details to get the main branch
        repo_details = await self.get_repository_details_from_repo_name(repository)
        if not repo_details.main_branch:
            raise ResourceNotFoundError(
                f'Main branch not found for repository {repository}. '
                f'This repository may be empty or have no default branch configured.'
            )
        return f'{self.BASE_URL}/repositories/{repository}/src/{repo_details.main_branch}/{microagents_path}'

    def _get_microagents_directory_params(self, microagents_path: str) -> dict | None:
        """Get parameters for the microagents directory request. Return None if no parameters needed."""
        return None

    def _is_valid_microagent_file(self, item: dict) -> bool:
        """Check if an item represents a valid microagent file."""
        return (
            item['type'] == 'commit_file'
            and item['path'].endswith('.md')
            and not item['path'].endswith('README.md')
        )

    def _get_file_name_from_item(self, item: dict) -> str:
        """Extract file name from directory item."""
        return item['path'].split('/')[-1]

    def _get_file_path_from_item(self, item: dict, microagents_path: str) -> str:
        """Extract file path from directory item."""
        return item['path']
