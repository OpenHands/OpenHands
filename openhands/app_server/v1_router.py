from fastapi import APIRouter

from openhands.app_server.app_conversation import app_conversation_router
from openhands.app_server.event import event_router
from openhands.app_server.event_callback import (
    webhook_router,
)
from openhands.app_server.pending_messages.pending_message_router import (
    router as pending_message_router,
)
from openhands.app_server.sandbox import sandbox_router, sandbox_spec_router
from openhands.app_server.secrets.secrets_router import (
    router as secrets_router,
)
from openhands.app_server.settings.settings_router import (
    router as settings_router,
)
from openhands.app_server.user import skills_router, user_router
from openhands.app_server.web_client import web_client_router

# Include routers
router = APIRouter(prefix='/api/v1')
router.include_router(event_router.router)
router.include_router(app_conversation_router.router)
router.include_router(pending_message_router)
router.include_router(sandbox_router.router)
router.include_router(sandbox_spec_router.router)
router.include_router(settings_router)
router.include_router(secrets_router)
router.include_router(user_router.router)
router.include_router(skills_router.router)
router.include_router(webhook_router.router)
router.include_router(web_client_router.router)
