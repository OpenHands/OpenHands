

from fastapi import APIRouter

from openhands.app_server.config import get_global_config
from openhands.app_server.web_client_config.web_client_config_models import WebClientConfig


router = APIRouter(prefix='/web_client_config', tags=['Config'])


@router.get('')
async def get_web_client_config() -> WebClientConfig:
    """Get the configuration of the web client."""
    config = get_global_config()
    return config.web_client_config
