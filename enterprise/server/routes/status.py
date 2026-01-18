import asyncio
from datetime import datetime, timedelta
import httpx
from fastapi import APIRouter, HTTPException
from enterprise.integrations.types import WidgetResponse
from openhands.server.shared import server_config

_cache: dict[str, tuple[dict, datetime]] = {}
_cache_locks: dict[str, asyncio.Lock] = {}

router = APIRouter(prefix="/api", tags=["status"])

async def fetch_incident_status():
    """
    Fetch current status from incident.io widget API

    Returns:
        WidgetResponse: Current incidents and maintenances
    """
    widget_url = server_config.incident_io_widget_url  # type: ignore[attr-defined]
    if not widget_url:
        raise HTTPException(
            status_code=503, detail="Incident.io widget URL not configured"
        )

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                widget_url, timeout=server_config.incident_io_request_timeout_seconds
            )  # type: ignore[attr-defined]
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException:
            raise HTTPException(status_code=503, detail="Incident.io timeout")
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=503, detail=f"Incident.io error: {e.response.status_code}"
            )
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Network Error: {e}")


@router.get("/v1/status", response_model=WidgetResponse)
async def get_incident_status():
    """
    Response is cached to avoid hitting incident.io too frequently.
    Thread-safe: Uses asyncio.Lock to prevent concurrent fetches.
    Cache TTL and timeout are configurable.
    """
    cache_key = "incident_status"
    cache_ttl = timedelta(seconds=server_config.incident_io_cache_ttl_seconds)  # type: ignore[attr-defined]

    if cache_key not in _cache_locks:
        _cache_locks[cache_key] = asyncio.Lock()

    now = datetime.now()
    if cache_key in _cache:
        cached_data, cached_at = _cache[cache_key]
        if now - cached_at < cache_ttl:
            return cached_data

    async with _cache_locks[cache_key]:
        if cache_key in _cache:
            cached_data, cached_at = _cache[cache_key]
            if now - cached_at < cache_ttl:
                return cached_data

        data = await fetch_incident_status()
        _cache[cache_key] = (data, now)
        return data
