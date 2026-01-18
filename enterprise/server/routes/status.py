from fastapi import APIRouter, HTTPException
from typing import Any
import httpx
from datetime import datetime, timedelta
from openhands.server.shared import server_config
from integrations.types import WidgetResponse

_cache: dict[str, tuple[dict[str, Any], datetime]] = {}
CACHE_TTL = timedelta(seconds=60)

router = APIRouter(prefix='/v1', tags=['status'])

async def fetch_incident_status():
    """"
    Fetch current status from incident.io widget API

    Returns:
        WidgetResponse: Current incidents and maintenances
    """
    widget_url = server_config.incident_io_widget_url
    if not widget_url:
        raise HTTPException(status_code=503, detail="Incident.io widget URL not configured")

    fetch_url = f"{widget_url}/api/widget"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(fetch_url, timeout=5.0)
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException:
            raise HTTPException(status_code=503, detail="Incident.io timeout")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=503, detail=f"Incident.io error: {e.response.status_code}")
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Network Error: {e}")

@router.get('/status', response_model=WidgetResponse)
async def get_incident_status():
    """"
    Response is cached for 60 seconds to avoid hitting incident.io too frequently.
    """
    cache_key = "incident_status"
    now = datetime.now()
    if cache_key in _cache:
        cached_data, cached_at = _cache[cache_key]
        if now - cached_at < CACHE_TTL:
            return cached_data

    data = await fetch_incident_status()
    _cache[cache_key] = (data, now)
    return data

