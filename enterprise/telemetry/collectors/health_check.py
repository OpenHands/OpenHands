"""Health check metrics collector for OpenHands Enterprise Telemetry.

This collector provides basic health and operational status metrics that can
help identify system issues and monitor overall installation health.
"""

import logging
import os
import platform
import time
from datetime import UTC, datetime
from typing import List

from storage.database import session_maker
from telemetry.base_collector import MetricResult, MetricsCollector
from telemetry.registry import register_collector

logger = logging.getLogger(__name__)


@register_collector('health_check')
class HealthCheckCollector(MetricsCollector):
    """Collects basic health and operational status metrics.

    This collector provides system health indicators and operational
    metrics that can help identify issues and monitor installation status.
    """

    _start_time: float = time.time()

    @property
    def collector_name(self) -> str:
        """Return the unique name for this collector."""
        return 'health_check'

    def collect(self) -> List[MetricResult]:
        """Collect health check metrics.

        Returns:
            List of MetricResult objects containing health metrics
        """
        results = []

        try:
            # Collection timestamp
            results.append(
                MetricResult(
                    key='collection_timestamp', value=datetime.now(UTC).isoformat()
                )
            )

            # System information
            results.append(MetricResult(key='platform_system', value=platform.system()))

            results.append(
                MetricResult(key='platform_release', value=platform.release())
            )

            results.append(
                MetricResult(key='python_version', value=platform.python_version())
            )

            # Database connectivity check
            db_healthy = self._check_database_health()
            results.append(MetricResult(key='database_healthy', value=db_healthy))

            # Environment indicators (without exposing sensitive data)
            results.append(
                MetricResult(
                    key='has_github_app_config',
                    value=bool(os.getenv('GITHUB_APP_CLIENT_ID')),
                )
            )

            results.append(
                MetricResult(
                    key='has_keycloak_config',
                    value=bool(os.getenv('KEYCLOAK_SERVER_URL')),
                )
            )

            # Uptime approximation (time since this collector was first loaded)
            uptime_seconds = int(time.time() - self.__class__._start_time)
            results.append(
                MetricResult(key='collector_uptime_seconds', value=uptime_seconds)
            )

            logger.info(f'Collected {len(results)} health check metrics')

        except Exception as e:
            logger.error(f'Failed to collect health check metrics: {e}')
            # Add an error metric instead of failing completely
            results.append(MetricResult(key='health_check_error', value=str(e)))

        return results

    def _check_database_health(self) -> bool:
        """Check if the database is accessible and healthy.

        Returns:
            True if database is healthy, False otherwise
        """
        try:
            with session_maker() as session:
                # Simple query to test database connectivity
                session.execute('SELECT 1')
                return True
        except Exception as e:
            logger.warning(f'Database health check failed: {e}')
            return False
