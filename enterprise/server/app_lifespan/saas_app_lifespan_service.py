"""SaaS-specific application lifespan service.

Initializes PostHog analytics on startup and flushes buffered events on
clean shutdown so no events are lost when the server exits gracefully.
"""

from __future__ import annotations

import os

from server.constants import IS_FEATURE_ENV

from openhands.analytics import get_analytics_service, init_analytics_service
from openhands.app_server.app_lifespan.app_lifespan_service import AppLifespanService
from openhands.app_server.utils.logger import openhands_logger as logger
from openhands.server.types import AppMode


class SaasAppLifespanService(AppLifespanService):
    """Lifespan service for the SaaS server.

    On enter: initialises the PostHog analytics singleton from environment vars.
    On exit: calls ``analytics_service.shutdown()`` to flush any buffered events.
    """

    async def __aenter__(self):
        api_key = os.environ.get('POSTHOG_CLIENT_KEY', '')
        host = os.environ.get('POSTHOG_HOST', 'https://us.i.posthog.com')

        init_analytics_service(
            api_key=api_key,
            host=host,
            app_mode=AppMode.SAAS,
            is_feature_env=IS_FEATURE_ENV,
        )
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        try:
            svc = get_analytics_service()
            if svc is not None:
                svc.shutdown()
        except Exception:
            logger.exception('Error shutting down analytics service')
