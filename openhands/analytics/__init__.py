"""OpenHands analytics package.

Provides a module-level singleton pattern for the AnalyticsService.

Usage::

    from openhands.analytics import init_analytics_service, get_analytics_service

    # At application startup:
    init_analytics_service(api_key=..., host=..., app_mode=..., is_feature_env=...)

    # At call sites:
    svc = get_analytics_service()
    if svc:
        svc.capture(...)
"""

from openhands.analytics.analytics_context import (
    AnalyticsContext,
    resolve_analytics_context,
)
from openhands.analytics.analytics_service import AnalyticsService
from openhands.server.types import AppMode

_analytics_service: AnalyticsService | None = None


def init_analytics_service(
    api_key: str,
    host: str,
    app_mode: AppMode,
    is_feature_env: bool,
) -> AnalyticsService:
    """Create and store the module-level AnalyticsService singleton.

    Returns the newly created instance. Subsequent calls to
    :func:`get_analytics_service` will return the same object.
    """
    global _analytics_service
    _analytics_service = AnalyticsService(
        api_key=api_key,
        host=host,
        app_mode=app_mode,
        is_feature_env=is_feature_env,
    )
    return _analytics_service


def get_analytics_service() -> AnalyticsService | None:
    """Return the module-level AnalyticsService singleton.

    Returns ``None`` if :func:`init_analytics_service` has not been called yet.
    """
    return _analytics_service


__all__ = [
    'AnalyticsContext',
    'AnalyticsService',
    'get_analytics_service',
    'init_analytics_service',
    'resolve_analytics_context',
]
