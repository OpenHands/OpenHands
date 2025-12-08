import json
from typing import Any, cast

import httpx
from pydantic import SecretStr

from openhands.integrations.protocols.http_client import HTTPClient
from openhands.integrations.service_types import (
    BaseGitService,
    RequestMethod,
    UnknownException,
    User,
)
from openhands.utils.http_session import httpx_verify_option


class GitHubMixinBase(BaseGitService, HTTPClient):
    """
    Declares common attributes and method signatures used across mixins.
    """

    BASE_URL: str
    GRAPHQL_URL: str

    async def _get_headers(self) -> dict:
        """Retrieve the GH Token from settings store to construct the headers."""
        if not self.token:
            latest_token = await self.get_latest_token()
            if latest_token:
                self.token = latest_token

        return {
            'Authorization': f'Bearer {self.token.get_secret_value() if self.token else ""}',
            'Accept': 'application/vnd.github.v3+json',
        }

    async def get_latest_token(self) -> SecretStr | None:  # type: ignore[override]
        return self.token

    async def _make_request(
        self,
        url: str,
        params: dict | None = None,
        method: RequestMethod = RequestMethod.GET,
    ) -> tuple[Any, dict]:  # type: ignore[override]
        try:
            async with httpx.AsyncClient(verify=httpx_verify_option()) as client:
                github_headers = await self._get_headers()

                # Make initial request
                response = await self.execute_request(
                    client=client,
                    url=url,
                    headers=github_headers,
                    params=params,
                    method=method,
                )

                # Handle token refresh if needed
                if self.refresh and self._has_token_expired(response.status_code):
                    await self.get_latest_token()
                    github_headers = await self._get_headers()
                    response = await self.execute_request(
                        client=client,
                        url=url,
                        headers=github_headers,
                        params=params,
                        method=method,
                    )

                response.raise_for_status()
                headers: dict = {}
                if 'Link' in response.headers:
                    headers['Link'] = response.headers['Link']

                return response.json(), headers

        except httpx.HTTPStatusError as e:
            raise self.handle_http_status_error(e)
        except httpx.HTTPError as e:
            raise self.handle_http_error(e)

    async def execute_graphql_query(
        self, query: str, variables: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(verify=httpx_verify_option()) as client:
                github_headers = await self._get_headers()

                response = await client.post(
                    self.GRAPHQL_URL,
                    headers=github_headers,
                    json={'query': query, 'variables': variables},
                )
                response.raise_for_status()

                result = response.json()
                if 'errors' in result:
                    raise UnknownException(
                        f'GraphQL query error: {json.dumps(result["errors"])}'
                    )

                return dict(result)

        except httpx.HTTPStatusError as e:
            raise self.handle_http_status_error(e)
        except httpx.HTTPError as e:
            raise self.handle_http_error(e)

    async def verify_access(self) -> bool:
        url = f'{self.BASE_URL}'
        await self._make_request(url)
        return True

    async def get_user(self):
        url = f'{self.BASE_URL}/user'
        response, _ = await self._make_request(url)

        return User(
            id=str(response.get('id', '')),
            login=cast(str, response.get('login') or ''),
            avatar_url=cast(str, response.get('avatar_url') or ''),
            company=response.get('company'),
            name=response.get('name'),
            email=response.get('email'),
        )
