"""User activity metrics collector for OpenHands Enterprise Telemetry.

This collector gathers detailed user activity and engagement metrics including
conversation patterns, feature usage, and user behavior analytics.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import List

from sqlalchemy import func
from storage.database import session_maker
from storage.minimal_conversation_metadata import StoredConversationMetadata
from storage.user_settings import UserSettings
from telemetry.base_collector import MetricResult, MetricsCollector
from telemetry.registry import register_collector

logger = logging.getLogger(__name__)


@register_collector('user_activity')
class UserActivityCollector(MetricsCollector):
    """Collects detailed user activity and engagement metrics.

    This collector provides insights into how users are engaging with
    OpenHands Enterprise, including conversation patterns, feature usage,
    and user behavior analytics.
    """

    @property
    def collector_name(self) -> str:
        """Return the unique name for this collector."""
        return 'user_activity'

    def collect(self) -> List[MetricResult]:
        """Collect user activity metrics from the database.

        Returns:
            List of MetricResult objects containing user activity metrics
        """
        results = []

        try:
            with session_maker() as session:
                # Calculate time boundaries
                now = datetime.now(UTC)
                thirty_days_ago = now - timedelta(days=30)

                # Average conversations per active user (30 days)
                active_users_30d = (
                    session.query(StoredConversationMetadata.user_id)
                    .filter(StoredConversationMetadata.created_at >= thirty_days_ago)
                    .distinct()
                    .count()
                )

                conversations_30d = (
                    session.query(StoredConversationMetadata)
                    .filter(StoredConversationMetadata.created_at >= thirty_days_ago)
                    .count()
                )

                avg_conversations_per_user = (
                    conversations_30d / active_users_30d if active_users_30d > 0 else 0
                )
                results.append(
                    MetricResult(
                        key='avg_conversations_per_user_30d',
                        value=round(avg_conversations_per_user, 2),
                    )
                )

                # Most popular LLM models (top 5)
                model_usage = (
                    session.query(
                        StoredConversationMetadata.llm_model,
                        func.count(StoredConversationMetadata.llm_model).label('count'),
                    )
                    .filter(StoredConversationMetadata.created_at >= thirty_days_ago)
                    .filter(StoredConversationMetadata.llm_model.isnot(None))
                    .group_by(StoredConversationMetadata.llm_model)
                    .order_by(func.count(StoredConversationMetadata.llm_model).desc())
                    .limit(5)
                    .all()
                )

                model_stats = {}
                for model, count in model_usage:
                    # Clean up model names for telemetry
                    clean_model = (
                        model.replace('/', '_').replace('-', '_')
                        if model
                        else 'unknown'
                    )
                    model_stats[f'model_usage_{clean_model}'] = count

                for key, value in model_stats.items():
                    results.append(MetricResult(key=key, value=value))

                # Git provider usage
                provider_usage = (
                    session.query(
                        StoredConversationMetadata.git_provider,
                        func.count(StoredConversationMetadata.git_provider).label(
                            'count'
                        ),
                    )
                    .filter(StoredConversationMetadata.created_at >= thirty_days_ago)
                    .filter(StoredConversationMetadata.git_provider.isnot(None))
                    .group_by(StoredConversationMetadata.git_provider)
                    .all()
                )

                for provider, count in provider_usage:
                    clean_provider = (
                        provider.lower().replace(' ', '_') if provider else 'unknown'
                    )
                    results.append(
                        MetricResult(key=f'git_provider_{clean_provider}', value=count)
                    )

                # Conversation trigger types
                trigger_usage = (
                    session.query(
                        StoredConversationMetadata.trigger,
                        func.count(StoredConversationMetadata.trigger).label('count'),
                    )
                    .filter(StoredConversationMetadata.created_at >= thirty_days_ago)
                    .filter(StoredConversationMetadata.trigger.isnot(None))
                    .group_by(StoredConversationMetadata.trigger)
                    .all()
                )

                for trigger, count in trigger_usage:
                    clean_trigger = (
                        trigger.lower().replace(' ', '_') if trigger else 'unknown'
                    )
                    results.append(
                        MetricResult(key=f'trigger_{clean_trigger}', value=count)
                    )

                # User engagement metrics
                # Users with multiple conversations (indicates engagement)
                engaged_users = (
                    session.query(StoredConversationMetadata.user_id)
                    .filter(StoredConversationMetadata.created_at >= thirty_days_ago)
                    .group_by(StoredConversationMetadata.user_id)
                    .having(func.count(StoredConversationMetadata.conversation_id) > 1)
                    .count()
                )

                results.append(
                    MetricResult(key='engaged_users_30d', value=engaged_users)
                )

                # Average token usage per conversation (30 days)
                token_stats = (
                    session.query(
                        func.avg(StoredConversationMetadata.total_tokens).label(
                            'avg_tokens'
                        ),
                        func.sum(StoredConversationMetadata.total_tokens).label(
                            'total_tokens'
                        ),
                    )
                    .filter(StoredConversationMetadata.created_at >= thirty_days_ago)
                    .filter(StoredConversationMetadata.total_tokens > 0)
                    .first()
                )

                if token_stats and token_stats.avg_tokens:
                    results.append(
                        MetricResult(
                            key='avg_tokens_per_conversation_30d',
                            value=int(token_stats.avg_tokens),
                        )
                    )
                    results.append(
                        MetricResult(
                            key='total_tokens_30d',
                            value=int(token_stats.total_tokens or 0),
                        )
                    )

                # Users with analytics consent
                analytics_consent_users = (
                    session.query(UserSettings)
                    .filter(UserSettings.user_consents_to_analytics)
                    .count()
                )

                results.append(
                    MetricResult(
                        key='users_with_analytics_consent',
                        value=analytics_consent_users,
                    )
                )

                logger.info(f'Collected {len(results)} user activity metrics')

        except Exception as e:
            logger.error(f'Failed to collect user activity metrics: {e}')
            # Re-raise the exception so the collection system can handle it
            raise

        return results
