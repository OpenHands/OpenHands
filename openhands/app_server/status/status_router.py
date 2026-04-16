from fastapi import APIRouter

from openhands.app_server.status.system_stats import get_system_info

router = APIRouter(tags=['Status'])


@router.get('/alive')
async def alive():
    """Endpoint for liveness probes.

    If this responds then the server is considered alive.
    """
    return {'status': 'ok'}


@router.get('/health')
async def health() -> str:
    """Health check endpoint.

    Returns 'OK' if the service is healthy and ready to accept requests.
    This is typically used by load balancers and orchestrators (e.g., Kubernetes)
    to determine if the service should receive traffic.
    """
    return 'OK'


@router.get('/server_info')
async def get_server_info():
    """Server information endpoint.

    Returns system information including CPU count, memory usage, and
    other runtime details about the server. Useful for monitoring and
    debugging purposes.
    """
    return get_system_info()


@router.get('/ready')
async def ready() -> str:
    """Endpoint for readiness probes.

    For now this is functionally the same as the liveness probe, but should
    we need to establish further invariants in the future, having a separate
    endpoint will mean we don't need to change client code.
    """
    return 'OK'
