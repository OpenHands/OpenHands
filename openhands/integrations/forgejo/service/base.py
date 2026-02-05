from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

import httpx
from pydantic import SecretStr

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.protocols.http_client import HTTPClient
from openhands.integrations.service_types import (
    BaseGitService,
    OwnerType,
    ProviderType,
    Repository,
    RequestMethod,
    UnknownException,
    User,
)
from openhands.utils.http_session import httpx_verify_option


class ForgejoMixinBase(BaseGitService, HTTPClient):
    """Common functionality shared by Forgejo service mixins."""

    DEFAULT_BASE_URL = 'https://codeberg.org/api/v1'
    DEFAULT_DOMAIN = 'codeberg.org'

    token: SecretStr = SecretStr('')
    refresh = False

    def __init__(
        self,
        user_id: str | None = None,
        external_auth_id: str | None = None,
        external_auth_token: SecretStr | None = None,
        token: SecretStr | None = None,
        external_token_manager: bool = False,
        base_domain: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.user_id = user_id
        self.external_auth_id = external_auth_id
        self.external_auth_token = external_auth_token
        self.external_token_manager = external_token_manager

        if token:
            self.token = token
        else:
            env_token = os.environ.get('FORGEJO_TOKEN')
            if env_token:
                self.token = SecretStr(env_token)

        env_base_url = os.environ.get('FORGEJO_BASE_URL')
        self.BASE_URL = self._resolve_base_url(base_url, base_domain, env_base_url)
        self.base_url = self.BASE_URL  # Backwards compatibility for existing usage
        parsed = urlparse(self.BASE_URL)
        self.base_domain = parsed.netloc or self.DEFAULT_DOMAIN
        # Preserve the protocol from BASE_URL (http or https)
        protocol = parsed.scheme or 'https'
        self.web_base_url = f'{protocol}://{self.base_domain}'.rstrip('/')

    @property
    def provider(self) -> str:
        return ProviderType.FORGEJO.value

    async def get_latest_token(self) -> SecretStr | None:
        return self.token

    async def _get_headers(self) -> dict[str, Any]:
        if not self.token:
            latest_token = await self.get_latest_token()
            if latest_token:
                self.token = latest_token

        return {
            'Authorization': f'token {self.token.get_secret_value() if self.token else ""}',
            'Accept': 'application/json',
        }

    async def _make_request(
        self,
        url: str,
        params: dict | None = None,
        method: RequestMethod = RequestMethod.GET,
    ) -> tuple[Any, dict]:
        try:
            async with httpx.AsyncClient(verify=httpx_verify_option()) as client:
                headers = await self._get_headers()
                response = await self.execute_request(
                    client=client,
                    url=url,
                    headers=headers,
                    params=params,
                    method=method,
                )

                if self.refresh and self._has_token_expired(response.status_code):
                    await self.get_latest_token()
                    headers = await self._get_headers()
                    response = await self.execute_request(
                        client=client,
                        url=url,
                        headers=headers,
                        params=params,
                        method=method,
                    )

                response.raise_for_status()
                headers_out: dict[str, str] = {}
                for header in ('Link', 'X-Total-Count', 'X-Total'):
                    if header in response.headers:
                        headers_out[header] = response.headers[header]

                content_type = response.headers.get('Content-Type', '')
                if 'application/json' in content_type:
                    return response.json(), headers_out
                return response.text, headers_out

        except httpx.HTTPStatusError as err:
            raise self.handle_http_status_error(err)
        except httpx.HTTPError as err:
            raise self.handle_http_error(err)

    def _resolve_base_url(
        self,
        explicit_base_url: str | None,
        base_domain: str | None,
        env_base_url: str | None,
    ) -> str:
        for candidate in (
            explicit_base_url,
            base_domain,
            env_base_url,
            self.DEFAULT_BASE_URL,
        ):
            if not candidate:
                continue

            normalized = candidate.strip().rstrip('/')
            if not normalized:
                continue

            if normalized.startswith(('http://', 'https://')):
                url = normalized
            else:
                url = f'https://{normalized}'

            if '/api/' in url:
                return url

            return f'{url}/api/v1'

        return self.DEFAULT_BASE_URL

    async def get_user(self) -> User:  # type: ignore[override]
        url = f'{self.BASE_URL}/user'
        response, _ = await self._make_request(url)

        return User(
            id=str(response.get('id', '')),
            login=response.get('username', ''),
            avatar_url=response.get('avatar_url', ''),
            name=response.get('full_name'),
            email=response.get('email'),
            company=response.get('organization'),
        )

    def _parse_repository(
        self, repo: dict, link_header: str | None = None
    ) -> Repository:
        owner = repo.get('owner') or {}
        owner_type = (
            OwnerType.ORGANIZATION
            if (owner.get('type') or '').lower() == 'organization'
            else OwnerType.USER
        )

        return Repository(
            id=str(repo.get('id', '')),
            full_name=repo.get('full_name', ''),
            stargazers_count=repo.get('stars_count'),
            git_provider=ProviderType.FORGEJO,
            is_public=not repo.get('private', False),
            link_header=link_header,
            pushed_at=repo.get('updated_at') or repo.get('pushed_at'),
            owner_type=owner_type,
            main_branch=repo.get('default_branch'),
        )

    def _split_repo(self, repository: str) -> tuple[str, str]:
        repo_path = repository.strip()
        if repo_path.startswith(('http://', 'https://')):
            parsed = urlparse(repo_path)
            repo_path = parsed.path.lstrip('/')

        parts = [part for part in repo_path.split('/') if part]
        if len(parts) < 2:
            raise ValueError(f'Invalid repository format: {repository}')

        return parts[0], parts[1]

    def _build_repo_api_url(self, owner: str, repo: str, *segments: str) -> str:
        base = f'{self.BASE_URL}/repos/{owner}/{repo}'
        if segments:
            base = f'{base}/{"/".join(segments)}'
        return base

    def _map_sort(self, sort: str) -> str:
        sort_map = {
            'pushed': 'updated',
            'updated': 'updated',
            'created': 'created',
            'full_name': 'name',
        }
        return sort_map.get(sort, 'updated')

    def handle_http_error(self, e: httpx.HTTPError) -> UnknownException:  # type: ignore[override]
        logger.warning(f'HTTP error on {self.provider} API: {type(e).__name__} : {e}')
        return UnknownException(f'HTTP error {type(e).__name__} : {e}')
