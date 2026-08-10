"""HTTP client for Findings Service POST /api/pentest/findings."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from shared.session_auth import MissingSessionApiKeyError, session_auth_headers

logger = logging.getLogger(__name__)

FINDINGS_SERVICE_URL_ENV = "FINDINGS_SERVICE_URL"
DEFAULT_FINDINGS_SERVICE_URL = "http://findings-service:8000"
FINDINGS_PATH = "/api/pentest/findings"


class FindingsAuthError(RuntimeError):
    """Findings Service rejected the request (401/403)."""

    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(f"Findings auth failed ({status_code})")


class FindingsClientError(RuntimeError):
    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(f"Findings request failed ({status_code})")


def findings_base_url() -> str:
    return os.environ.get(
        FINDINGS_SERVICE_URL_ENV, DEFAULT_FINDINGS_SERVICE_URL
    ).rstrip("/")


class FindingsClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 30.0,
    ):
        self.base_url = (base_url or findings_base_url()).rstrip("/")
        self._transport = transport
        self._timeout = timeout

    async def post_finding(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        POST a normalized finding.

        201 → created finding body
        409 → idempotent success with existing_finding_id
        401/403 → FindingsAuthError (never swallowed)
        """
        try:
            headers = session_auth_headers()
        except MissingSessionApiKeyError:
            raise FindingsAuthError(401, "SESSION_API_KEY missing") from None

        headers["Content-Type"] = "application/json"
        url = f"{self.base_url}{FINDINGS_PATH}"

        async with httpx.AsyncClient(
            transport=self._transport, timeout=self._timeout
        ) as client:
            resp = await client.post(url, json=payload, headers=headers)

        if resp.status_code == 201:
            return {"status": "created", "finding": resp.json()}

        if resp.status_code == 409:
            detail = resp.json().get("detail", {})
            existing_id = (
                detail.get("existing_finding_id")
                if isinstance(detail, dict)
                else None
            )
            logger.info(
                "Findings dedupe hit (409); treating as idempotent success"
            )
            return {
                "status": "duplicate",
                "existing_finding_id": existing_id,
            }

        if resp.status_code in (401, 403):
            raise FindingsAuthError(resp.status_code, resp.text)

        raise FindingsClientError(resp.status_code, resp.text)
