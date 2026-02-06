from fastapi import APIRouter

from openhands.app_server.config import get_global_config
from openhands.app_server.web_client.web_client_models import WebClientConfig

router = APIRouter(prefix='/web-client', tags=['Config'])


@router.get('/config')
async def get_web_client_config() -> WebClientConfig:
    """Get the configuration of the web client.

    This endpoint is typically one of the first invoked, and does not require
    authentication. It provides general settings for the web client independent
    of users.
    """
    config = get_global_config()
    result = await config.web_client.get_web_client_config()
    return result
