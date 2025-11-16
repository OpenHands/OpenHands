"""System metrics collector for OpenHands Enterprise Telemetry.

This collector gathers basic system and usage metrics including user counts,
conversation statistics, and system health indicators.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import List

from storage.database import session_maker
from storage.minimal_conversation_metadata import StoredConversationMetadata
from storage.user_settings import UserSettings
from telemetry.base_collector import MetricResult, MetricsCollector
from telemetry.registry import register_collector

logger = logging.getLogger(__name__)


@register_collector('system_metrics')
class SystemMetricsCollector(MetricsCollector):
    """Collects basic system and usage metrics.

    This collector provides essential metrics about the OpenHands Enterprise
    installation including user counts, conversation activity, and system usage.
    """

    @property
    def collector_name(self) -> str:
        """Return the unique name for this collector."""
        return 'system_metrics'

    def collect(self) -> List[MetricResult]:
        """Collect system metrics from the database.

        Returns:
            List of MetricResult objects containing system metrics
        """
        results = []

        try:
            with session_maker() as session:
                # Collect total user count
                total_users = session.query(UserSettings).count()
                results.append(MetricResult(key='total_users', value=total_users))

                # Collect active users (users who have accepted ToS)
                active_users = (
                    session.query(UserSettings)
                    .filter(UserSettings.accepted_tos.isnot(None))
                    .count()
                )
                results.append(MetricResult(key='active_users', value=active_users))

                # Collect total conversations
                total_conversations = session.query(StoredConversationMetadata).count()
                results.append(
                    MetricResult(key='total_conversations', value=total_conversations)
                )

                # Collect conversations in the last 30 days
                thirty_days_ago = datetime.now(UTC) - timedelta(days=30)
                recent_conversations = (
                    session.query(StoredConversationMetadata)
                    .filter(StoredConversationMetadata.created_at >= thirty_days_ago)
                    .count()
                )
                results.append(
                    MetricResult(key='conversations_30d', value=recent_conversations)
                )

                # Collect conversations in the last 7 days
                seven_days_ago = datetime.now(UTC) - timedelta(days=7)
                weekly_conversations = (
                    session.query(StoredConversationMetadata)
                    .filter(StoredConversationMetadata.created_at >= seven_days_ago)
                    .count()
                )
                results.append(
                    MetricResult(key='conversations_7d', value=weekly_conversations)
                )

                # Collect unique active users in the last 30 days
                active_users_30d = (
                    session.query(StoredConversationMetadata.user_id)
                    .filter(StoredConversationMetadata.created_at >= thirty_days_ago)
                    .distinct()
                    .count()
                )
                results.append(
                    MetricResult(key='active_users_30d', value=active_users_30d)
                )

                logger.info(f'Collected {len(results)} system metrics')

        except Exception as e:
            logger.error(f'Failed to collect system metrics: {e}')
            # Re-raise the exception so the collection system can handle it
            raise

        return results
