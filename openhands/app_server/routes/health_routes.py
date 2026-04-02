from fastapi import APIRouter

from openhands.runtime.utils.system_stats import get_system_info

router = APIRouter(tags=['Health'])


@router.get('/alive')
async def alive():
    """Endpoint for liveness probes. If this responds then the server is
    considered alive."""
    return {'status': 'ok'}


@router.get('/health')
async def health() -> str:
    return 'OK'


@router.get('/server_info')
async def get_server_info():
    return get_system_info()


@router.get('/ready')
async def ready() -> str:
    """Endpoint for readiness probes. For now this is functionally the same as
    the liveness probe, but should be need to establish further invariants in
    the future, having a separate endpoint will mean we don't need to change
    client code."""
    return 'OK'
