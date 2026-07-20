"""SaaS-specific application lifespan service.

Initializes PostHog analytics on startup and flushes buffered events on
clean shutdown so no events are lost when the server exits gracefully.
Also reconciles user_authorizations table with environment variables.
"""

from __future__ import annotations

import os

from server.constants import IS_FEATURE_ENV

from openhands.analytics import get_analytics_service, init_analytics_service
from openhands.app_server.app_lifespan.app_lifespan_service import AppLifespanService
from openhands.app_server.utils.logger import openhands_logger as logger
from openhands.server.types import AppMode
from storage.user_authorization_store import UserAuthorizationStore


class SaasAppLifespanService(AppLifespanService):
    """Lifespan service for the SaaS server.

    On enter: initialises the PostHog analytics singleton from environment vars,
    and reconciles user_authorizations table with EMAIL_PATTERN_WHITELIST/BLACKLIST.
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

        # Reconcile user_authorizations table with environment variables
        # This ensures that changes to EMAIL_PATTERN_WHITELIST/BLACKLIST env vars
        # take effect even on already-migrated environments
        await self._reconcile_user_authorizations()

        return self

    async def _reconcile_user_authorizations(self) -> None:
        """Reconcile user_authorizations table with environment variables.

        Logs the results of reconciliation for visibility.
        """
        try:
            stats = await UserAuthorizationStore.reconcile_from_environment()
            if stats['whitelists_added'] > 0 or stats['blacklists_added'] > 0:
                logger.info(
                    f'Reconciled user_authorizations: '
                    f"{stats['whitelists_added']} whitelist(s), "
                    f"{stats['blacklists_added']} blacklist(s) processed"
                )
            else:
                logger.debug('No user_authorizations reconciliation needed')
        except Exception:
            logger.exception(
                'Error reconciling user_authorizations from environment variables'
            )

    async def __aexit__(self, exc_type, exc_value, traceback):
        try:
            svc = get_analytics_service()
            if svc is not None:
                svc.shutdown()
        except Exception:
            logger.exception('Error shutting down analytics service', stack_info=True)

        # Release long-lived database resources: the GCP Cloud SQL connector
        # (background cert-refresh tasks + aiohttp ClientSession) and the
        # SQLAlchemy async engine's pool. Without this, every worker respawn
        # leaves the connector's tasks running on the previous event loop.
        try:
            from openhands.app_server.config import get_global_config

            await get_global_config().db_session.close()
        except Exception:
            logger.exception('Error closing DB session injector', stack_info=True)
