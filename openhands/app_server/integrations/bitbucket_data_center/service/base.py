import base64
from typing import Any
from urllib.parse import unquote

import httpx
from pydantic import SecretStr

from openhands.app_server.integrations.protocols.http_client import HTTPClient
from openhands.app_server.integrations.service_types import (
    AuthenticationError,
    BaseGitService,
    OwnerType,
    ProviderType,
    Repository,
    RequestMethod,
    User,
)
from openhands.app_server.utils.http_session import httpx_verify_option
from openhands.app_server.utils.logger import openhands_logger as logger


class BitbucketDCMixinBase(BaseGitService, HTTPClient):
    """
    Base mixin for BitBucket data center service containing common functionality
    """

    BASE_URL: str = ''  # Set dynamically from domain in __init__
    user_id: str | None

    def _repo_api_base(self, owner: str, repo: str) -> str:
        return f'{self.BASE_URL}/projects/{owner}/repos/{repo}'

    @staticmethod
    def _resolve_primary_email(emails: list[dict]) -> str | None:
        """Find the primary confirmed email from a list of Bitbucket data center email objects.

        Bitbucket data center's /user/emails endpoint returns objects with
        'email', 'is_primary', and 'is_confirmed' keys.
        """
        for entry in emails:
            if entry.get('is_primary') and entry.get('is_confirmed'):
                return entry.get('email')
        return None

    def _extract_owner_and_repo(self, repository: str) -> tuple[str, str]:
        """Extract owner and repo from repository string.

        Args:
            repository: Repository name in format 'project/repo_slug'

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
        return False  # DC tokens cannot be refreshed programmatically

    async def _get_headers(self) -> dict[str, str]:
        """Get headers for Bitbucket data center API requests.

        Mirrors the Bitbucket Cloud base: when ``self.token`` is empty
        (the default for services constructed only with ``external_auth_id``),
        resolve the latest token before building the header. Without this,
        the header would be ``Authorization: Basic `` with an empty base64
        payload and httpx rejects it as ``Illegal header value``.

        Tokens in ``username:secret`` form (e.g. ``x-token-auth:<PAT>`` or
        ``user:pass``) use HTTP Basic auth. Raw OAuth 2.0 access tokens
        (no colon) use Bearer per RFC 6750 — this is the format the SaaS
        broker flow returns from Keycloak.
        """
        if not self.token or not self.token.get_secret_value():
            latest_token = await self.get_latest_token()
            if latest_token:
                self.token = latest_token

        token_value = self.token.get_secret_value()

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
        """Make a request to the Bitbucket data center API.

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
                try:
                    data = response.json()
                except ValueError:
                    data = response.text
                return data, dict(response.headers)
        except httpx.HTTPStatusError as e:
            raise self.handle_http_status_error(e) from e
        except httpx.HTTPError as e:
            raise self.handle_http_error(e) from e

    async def verify_access(self) -> None:
        """Verify that the token and host are valid by making a lightweight API call.
        Raises an exception if the token is invalid or the host is unreachable.
        """
        url = f'{self.BASE_URL}/repos'
        await self._make_request(url, {'limit': '1'})

    async def _fetch_paginated_data(
        self, url: str, params: dict, max_items: int
    ) -> list[dict]:
        """Fetch data with pagination support for Bitbucket data center API.

        Args:
            url: The API endpoint URL
            params: Query parameters for the request
            max_items: Maximum number of items to fetch

        Returns:
            List of data items from all pages
        """
        all_items: list[dict] = []
        current_url = url
        base_endpoint = url

        while current_url and len(all_items) < max_items:
            response, _ = await self._make_request(current_url, params)

            # Extract items from response
            page_items = response.get('values', [])
            all_items.extend(page_items)

            if response.get('isLastPage', True):
                break
            next_start = response.get('nextPageStart')
            if next_start is None:
                break
            params = params or {}
            params = dict(params)
            params['start'] = next_start
            current_url = base_endpoint

        return all_items[:max_items]

    async def get_user_emails(self) -> list[dict]:
        """Fetch the authenticated user's email addresses from Bitbucket data center.

        Calls GET /user/emails which returns a paginated response with a
        'values' list of email objects containing 'email', 'is_primary',
        and 'is_confirmed' fields.
        """
        url = f'{self.BASE_URL}/user/emails'
        response, _ = await self._make_request(url)
        return response.get('values', [])

    @staticmethod
    def _select_user_data(users: list[dict], username: str) -> dict | None:
        """Pick the user matching ``username`` from a /users?filter= response.

        The filter parameter is a substring match, so the first result is not
        necessarily the requested user. Match case-insensitively on name,
        email, and slug, falling back to the first result.
        """
        username_folded = username.casefold()
        for user in users:
            for key in ('name', 'emailAddress', 'slug'):
                value = user.get(key)
                if isinstance(value, str) and value.casefold() == username_folded:
                    return user

        return users[0] if users else None

    async def _get_authenticated_username(self) -> str | None:
        """Resolve the authenticated user's username, or None for credentials
        not tied to a user (e.g. project/repo-scoped HTTP access tokens).

        Bitbucket Data Center has no ``/user`` (myself) REST resource, so try
        the applinks whoami servlet first (some servers answer it with 200 and
        an *empty* body for HTTP access tokens), then fall back to the
        URL-encoded ``X-AUSERNAME`` header from a cheap REST call.
        """
        if not self.BASE_URL:
            return None
        base_server_url = self.BASE_URL.rsplit('/rest/api/1.0', 1)[0]
        whoami_url = f'{base_server_url}/plugins/servlet/applinks/whoami'
        try:
            data, _ = await self._make_request(whoami_url)
        except Exception as e:
            logger.warning(f'Bitbucket data center whoami lookup failed: {e}')
            data = None
        if isinstance(data, str) and data.strip():
            return data.strip()

        # Fall back to the X-AUSERNAME header from a cheap REST call.
        try:
            _, headers = await self._make_request(
                f'{self.BASE_URL}/projects', {'limit': '0'}
            )
        except Exception as e:
            logger.warning(f'Bitbucket data center username lookup failed: {e}')
            return None
        username = next(
            (v for k, v in headers.items() if k.lower() == 'x-ausername'), None
        )
        if username:
            return unquote(username)
        return None

    async def get_user(self) -> User:
        """Get the authenticated user's information."""

        user_id = self.user_id
        if not user_id:
            user_id = await self._get_authenticated_username()

        if not user_id:
            # Credentials not tied to a user don't have user info.
            return User(
                id='',
                login='',
                avatar_url='',
                name=None,
                email=None,
            )

        # Query the users API with the username to get id, avatar, and email
        users_url = f'{self.BASE_URL}/users'
        data, _ = await self._make_request(
            users_url, {'filter': user_id, 'avatarSize': 64}
        )
        user_data = self._select_user_data(data.get('values', []), user_id)
        if not user_data:
            if not self.user_id:
                # The username was auto-resolved; degrade to the empty user
                # rather than failing the whole user-info request.
                logger.warning(f'Bitbucket data center user not found: {user_id}')
                return User(
                    id='',
                    login='',
                    avatar_url='',
                    name=None,
                    email=None,
                )
            raise AuthenticationError(f'User not found: {user_id}')

        avatar = user_data.get('avatarUrl', '')
        # Handle relative avatar URLs (Server returns /users/... paths)
        if avatar.startswith('/users'):
            # Strip /rest/api/1.0 from BASE_URL to get the base server URL
            base_server_url = self.BASE_URL.rsplit('/rest/api/1.0', 1)[0]
            avatar = f'{base_server_url}{avatar}'
        display_name = user_data.get('displayName')
        email = user_data.get('emailAddress')
        return User(
            id=str(user_data.get('id') or user_data.get('slug') or user_id),
            login=user_data.get('name') or user_id,
            avatar_url=avatar,
            name=display_name,
            email=email,
        )

    async def _parse_repository(
        self,
        repo: dict,
        link_header: str | None = None,
        fetch_default_branch: bool = False,
    ) -> Repository:
        """Parse a Bitbucket data center API repository response into a Repository object.

        Args:
            repo: Repository data from Bitbucket data center API
            link_header: Optional link header for pagination
            fetch_default_branch: Whether to make an additional API call to fetch the
                default branch. Set to False for listing endpoints to avoid N+1 queries.

        Returns:
            Repository object
        """
        project_key = repo.get('project', {}).get('key', '')
        repo_slug = repo.get('slug', '')

        if not project_key or not repo_slug:
            raise ValueError(
                f'Cannot parse repository: missing project key or slug. '
                f'Got project_key={project_key!r}, repo_slug={repo_slug!r}'
            )

        full_name = f'{project_key}/{repo_slug}'
        is_public = repo.get('public', False)

        main_branch: str | None = None
        if fetch_default_branch:
            try:
                default_branch_url = (
                    f'{self._repo_api_base(project_key, repo_slug)}/default-branch'
                )
                default_branch_data, _ = await self._make_request(default_branch_url)
                main_branch = default_branch_data.get('displayId') or None
            except Exception as e:
                logger.debug(f'Could not fetch default branch for {full_name}: {e}')

        return Repository(
            id=str(repo.get('id', '')),
            full_name=full_name,
            git_provider=ProviderType.BITBUCKET_DATA_CENTER,
            is_public=is_public,
            stargazers_count=None,
            pushed_at=None,
            owner_type=OwnerType.ORGANIZATION,
            link_header=link_header,
            main_branch=main_branch,
        )

    async def get_repository_details_from_repo_name(
        self, repository: str
    ) -> Repository:
        """Get repository details from repository name.

        Args:
            repository: Repository name in format 'project/repo_slug'

        Returns:
            Repository object with details
        """
        owner, repo = self._extract_owner_and_repo(repository)
        url = self._repo_api_base(owner, repo)
        data, _ = await self._make_request(url)
        return await self._parse_repository(data, fetch_default_branch=True)
