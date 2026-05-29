from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx
from pydantic import SecretStr


class JungleGridApiError(RuntimeError):
    """Raised when the Jungle Grid API returns an error response."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        self.status_code = status_code
        self.code = code
        super().__init__(message)


@dataclass
class JungleGridClient:
    """Async client boundary for the Jungle Grid API.

    Route details mirror the official Jungle Grid MCP server's REST API.
    """

    base_url: str
    api_key: SecretStr
    httpx_client: httpx.AsyncClient
    workspace_id: str | None = None

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip('/')

    def _headers(self) -> dict[str, str]:
        return {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.api_key.get_secret_value()}',
            'Content-Type': 'application/json',
        }

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if json is not None:
            kwargs['json'] = json
        response = await self.httpx_client.request(
            method,
            f'{self.base_url}{path}',
            headers=self._headers(),
            **kwargs,
        )
        if response.status_code == 204:
            return {}

        data = self._read_json(response)
        if response.is_error:
            raise self._api_error(response.status_code, data)

        unwrapped = self._unwrap_mcp_envelope(data)
        if isinstance(unwrapped, dict):
            return unwrapped
        return {'data': unwrapped}

    def _read_json(self, response: httpx.Response) -> Any:
        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError:
            if response.is_error:
                return {}
            raise JungleGridApiError(
                502,
                'INVALID_API_RESPONSE',
                'Jungle Grid API returned an invalid JSON response.',
            )

    def _api_error(self, status_code: int, data: Any) -> JungleGridApiError:
        parsed = self._parse_api_error(data)
        return JungleGridApiError(
            status_code,
            parsed.get('code') or self._status_code_to_error_code(status_code),
            parsed.get('message') or self._status_code_to_message(status_code),
        )

    def _parse_api_error(self, data: Any) -> dict[str, str | None]:
        if not isinstance(data, dict):
            return {'code': None, 'message': None}
        nested = data.get('error')
        if isinstance(nested, dict):
            return {
                'code': self._clean_string(nested.get('code')),
                'message': self._clean_string(nested.get('message')),
            }
        return {
            'code': self._clean_string(data.get('code')),
            'message': self._clean_string(data.get('message')),
        }

    def _unwrap_mcp_envelope(self, data: Any) -> Any:
        if isinstance(data, dict) and data.get('ok') is True and 'data' in data:
            return data['data']
        return data

    def _clean_string(self, value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        stripped = value.strip()
        return stripped or None

    def _status_code_to_error_code(self, status_code: int) -> str:
        if status_code == 401:
            return 'UNAUTHORIZED'
        if status_code == 403:
            return 'FORBIDDEN'
        if status_code == 404:
            return 'NOT_FOUND'
        if status_code == 429:
            return 'RATE_LIMITED'
        if status_code >= 500:
            return 'UPSTREAM_ERROR'
        return 'API_ERROR'

    def _status_code_to_message(self, status_code: int) -> str:
        if status_code == 401:
            return 'Authentication is required or the token is invalid.'
        if status_code == 403:
            return 'The token is not authorized for this Jungle Grid action.'
        if status_code == 404:
            return 'The requested Jungle Grid resource was not found.'
        if status_code == 429:
            return 'Jungle Grid API rate limit exceeded.'
        if status_code >= 500:
            return 'Jungle Grid API is temporarily unavailable.'
        return f'Jungle Grid API request failed with status {status_code}.'

    async def estimate_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._request('POST', '/v1/mcp/jobs/estimate', json=payload)

    async def submit_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._request('POST', '/v1/mcp/jobs', json=payload)

    async def get_job(self, job_id: str) -> dict[str, Any]:
        return await self._request('GET', f'/v1/mcp/jobs/{self._path(job_id)}')

    async def get_job_status(self, job_id: str) -> dict[str, Any]:
        return await self.get_job(job_id)

    async def get_job_logs(
        self, job_id: str, limit: int | None = None, cursor: str | int | None = None
    ) -> dict[str, Any]:
        params = {}
        if limit is not None:
            params['limit'] = str(limit)
        if cursor is not None and str(cursor).strip():
            params['cursor'] = str(cursor).strip()
        suffix = f'?{httpx.QueryParams(params)}' if params else ''
        return await self._request(
            'GET',
            f'/v1/mcp/jobs/{self._path(job_id)}/logs{suffix}',
        )

    async def cancel_job(
        self, job_id: str, reason: str | None = None
    ) -> dict[str, Any]:
        return await self._request(
            'POST',
            f'/v1/mcp/jobs/{self._path(job_id)}/cancel',
            json={
                'reason': reason.strip()
                if reason and reason.strip()
                else 'Cancelled via OpenHands'
            },
        )

    async def list_artifacts(self, job_id: str) -> dict[str, Any]:
        return await self._request(
            'GET',
            f'/v1/mcp/jobs/{self._path(job_id)}/artifacts',
        )

    async def get_artifact_download_url(
        self, job_id: str, artifact_id: str
    ) -> dict[str, Any]:
        return await self._request(
            'POST',
            f'/v1/mcp/jobs/{self._path(job_id)}'
            + f'/artifacts/{self._path(artifact_id)}'
            + '/download',
        )

    def _path(self, value: str) -> str:
        return quote(value, safe='')
