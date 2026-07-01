"""Dakera memory REST client.

Wraps the three endpoints used by the OpenHands integration:

    POST /v1/memory/store   – persist a new memory
    POST /v1/memory/search  – semantic recall
    POST /v1/memory/forget  – delete specific memories

All methods are async and accept/return plain Python dicts so callers are not
coupled to any Dakera-specific model types.  Errors are logged and swallowed
rather than propagated so that a Dakera outage never blocks a conversation.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from openhands.app_server.integrations.dakera.config import DakeraConfig

_logger = logging.getLogger(__name__)


class DakeraMemoryClient:
    """Async client for the Dakera memory REST API.

    Args:
        config: DakeraConfig instance with URL, key, and agent_id.
        http_client: Optional shared httpx.AsyncClient.  When *None* a new
            client is created per request (suitable for low-frequency calls).
            Pass a shared client in high-throughput contexts.
    """

    def __init__(
        self,
        config: DakeraConfig,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        self._http_client = http_client

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _base_headers(self) -> dict[str, str]:
        return {'Content-Type': 'application/json', **self._config.auth_headers}

    async def _post(
        self, path: str, payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        """POST *payload* to *path* and return the parsed JSON response.

        Returns ``None`` on any error so callers can distinguish a successful
        empty response from a failed request.
        """
        url = f'{self._config.api_url.rstrip("/")}{path}'
        headers = self._base_headers()

        async def _do_request(client: httpx.AsyncClient) -> dict[str, Any]:
            response = await client.post(
                url,
                json=payload,
                headers=headers,
                timeout=self._config.timeout,
            )
            response.raise_for_status()
            return response.json()

        try:
            if self._http_client is not None:
                return await _do_request(self._http_client)
            async with httpx.AsyncClient() as client:
                return await _do_request(client)
        except httpx.HTTPStatusError as exc:
            _logger.warning(
                'Dakera API returned HTTP %d for %s: %s',
                exc.response.status_code,
                path,
                exc.response.text[:200],
            )
        except httpx.TimeoutException:
            _logger.warning(
                'Dakera API request timed out after %.1fs for %s',
                self._config.timeout,
                path,
            )
        except httpx.RequestError as exc:
            _logger.warning('Dakera API connection error for %s: %s', path, exc)
        except Exception as exc:  # pragma: no cover – unexpected
            _logger.warning(
                'Unexpected error calling Dakera API %s: %s', path, exc
            )
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def store(
        self,
        content: str,
        *,
        session_id: str | None = None,
        importance: float | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store a memory in Dakera.

        Args:
            content: Text content to store.
            session_id: Optional conversation / session identifier for grouping.
            importance: Optional importance weight in [0, 1].
            tags: Optional list of string tags.
            metadata: Optional free-form metadata dict.

        Returns:
            The ``memory`` object from the API response, or ``{}`` on failure.
        """
        payload: dict[str, Any] = {
            'content': content,
            'agent_id': self._config.agent_id,
        }
        if session_id is not None:
            payload['session_id'] = session_id
        if importance is not None:
            payload['importance'] = importance
        if tags is not None:
            payload['tags'] = tags
        if metadata is not None:
            payload['metadata'] = metadata

        result = await self._post('/v1/memory/store', payload)
        if result is None:
            return {}
        return result.get('memory', {})

    async def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve semantically similar memories from Dakera.

        Args:
            query: Natural-language search query.
            top_k: Number of results to return.  Falls back to config default.

        Returns:
            List of hit dicts, each with keys ``memory`` and ``score``.
            Returns ``[]`` on any error.
        """
        payload: dict[str, Any] = {
            'agent_id': self._config.agent_id,
            'query': query,
            'top_k': top_k if top_k is not None else self._config.top_k,
        }
        result = await self._post('/v1/memory/search', payload)
        if result is None:
            return []
        return result.get('memories', [])

    async def forget(
        self,
        memory_ids: list[str] | None = None,
    ) -> bool:
        """Delete memories from Dakera.

        Args:
            memory_ids: Optional list of specific memory IDs to delete.
                If omitted, all memories for this agent are deleted.

        Returns:
            ``True`` if the request succeeded, ``False`` otherwise.
        """
        payload: dict[str, Any] = {'agent_id': self._config.agent_id}
        if memory_ids is not None:
            payload['memory_ids'] = memory_ids

        result = await self._post('/v1/memory/forget', payload)
        # _post returns None on error, a dict (possibly empty) on success
        return result is not None
