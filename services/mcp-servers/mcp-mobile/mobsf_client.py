"""OWASP MobSF REST client (upload → scan → report_json).

Auth: ``Authorization: <MOBSF_API_KEY>`` header. Key and URL come only from env —
never hardcoded. Fail-closed when ``MOBSF_URL`` / ``MOBSF_API_KEY`` are missing.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

MOBSF_URL_ENV = "MOBSF_URL"
MOBSF_API_KEY_ENV = "MOBSF_API_KEY"


class MobsfConfigError(RuntimeError):
    """Missing or invalid MobSF configuration."""

    code = "mobsf_config"

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

    def as_dict(self) -> dict[str, Any]:
        return {"error": self.code, "message": self.message}


class MobsfClientError(RuntimeError):
    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(f"MobSF request failed ({status_code})")


def mobsf_base_url() -> str:
    raw = os.environ.get(MOBSF_URL_ENV, "").strip().rstrip("/")
    if not raw:
        raise MobsfConfigError(f"{MOBSF_URL_ENV} is unset (fail-closed)")
    return raw


def mobsf_api_key() -> str:
    key = os.environ.get(MOBSF_API_KEY_ENV, "").strip()
    if not key:
        raise MobsfConfigError(f"{MOBSF_API_KEY_ENV} is unset (fail-closed)")
    return key


class MobsfClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 120.0,
    ):
        self.base_url = (base_url if base_url is not None else mobsf_base_url()).rstrip(
            "/"
        )
        self.api_key = api_key if api_key is not None else mobsf_api_key()
        self._transport = transport
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"Authorization": self.api_key}

    async def upload(self, apk_path: Path) -> dict[str, Any]:
        """POST /api/v1/upload — multipart APK."""
        url = f"{self.base_url}/api/v1/upload"
        # Do not log APK bytes (AppSec).
        logger.info("MobSF upload start path=%s", apk_path.name)
        async with httpx.AsyncClient(
            transport=self._transport, timeout=self._timeout
        ) as client:
            with apk_path.open("rb") as handle:
                files = {
                    "file": (
                        apk_path.name,
                        handle,
                        "application/vnd.android.package-archive",
                    )
                }
                resp = await client.post(url, headers=self._headers(), files=files)
        if resp.status_code >= 400:
            raise MobsfClientError(resp.status_code, resp.text[:500])
        return resp.json()

    async def scan(
        self,
        *,
        hash_value: str,
        scan_type: str = "apk",
        file_name: str,
    ) -> dict[str, Any]:
        """POST /api/v1/scan."""
        url = f"{self.base_url}/api/v1/scan"
        data = {
            "hash": hash_value,
            "scan_type": scan_type,
            "file_name": file_name,
        }
        async with httpx.AsyncClient(
            transport=self._transport, timeout=self._timeout
        ) as client:
            resp = await client.post(url, headers=self._headers(), data=data)
        if resp.status_code >= 400:
            raise MobsfClientError(resp.status_code, resp.text[:500])
        # Scan may return empty body on success
        if not resp.content:
            return {"hash": hash_value, "status": "ok"}
        try:
            return resp.json()
        except ValueError:
            return {"hash": hash_value, "status": "ok", "raw": resp.text[:200]}

    async def report_json(self, *, hash_value: str) -> dict[str, Any]:
        """POST /api/v1/report_json."""
        url = f"{self.base_url}/api/v1/report_json"
        async with httpx.AsyncClient(
            transport=self._transport, timeout=self._timeout
        ) as client:
            resp = await client.post(
                url, headers=self._headers(), data={"hash": hash_value}
            )
        if resp.status_code >= 400:
            raise MobsfClientError(resp.status_code, resp.text[:500])
        return resp.json()

    async def scorecard(self, *, hash_value: str) -> dict[str, Any]:
        """POST /api/v1/scorecard (optional)."""
        url = f"{self.base_url}/api/v1/scorecard"
        async with httpx.AsyncClient(
            transport=self._transport, timeout=self._timeout
        ) as client:
            resp = await client.post(
                url, headers=self._headers(), data={"hash": hash_value}
            )
        if resp.status_code >= 400:
            raise MobsfClientError(resp.status_code, resp.text[:500])
        return resp.json()

    async def upload_scan_report(self, apk_path: Path) -> dict[str, Any]:
        """Full static pipeline: upload → scan → report_json."""
        uploaded = await self.upload(apk_path)
        hash_value = str(uploaded.get("hash") or "").strip()
        if not hash_value:
            raise MobsfClientError(500, "MobSF upload response missing hash")
        file_name = str(uploaded.get("file_name") or apk_path.name)
        scan_type = str(uploaded.get("scan_type") or "apk")
        await self.scan(
            hash_value=hash_value, scan_type=scan_type, file_name=file_name
        )
        report = await self.report_json(hash_value=hash_value)
        return {
            "hash": hash_value,
            "file_name": file_name,
            "scan_type": scan_type,
            "report": report,
        }
