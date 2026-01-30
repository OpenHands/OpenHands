from server.logger import logger

from openhands.core.config.openhands_config import OpenHandsConfig
from openhands.core.schema.agent import AgentState
from openhands.events.event import Event
from openhands.events.observation import (
    AgentStateChangedObservation,
)
from openhands.server.monitoring import MonitoringListener


class SaaSMonitoringListener(MonitoringListener):
    """Forward app signals to structured logging for GCP native monitoring."""

    def on_session_event(self, event: Event) -> None:
        """Track metrics about events being added to a Session's EventStream."""
        if (
            isinstance(event, AgentStateChangedObservation)
            and event.agent_state == AgentState.ERROR
        ):
            logger.info(
                'Tracking agent status error',
                extra={'signal': 'saas_agent_status_errors'},
            )

    def on_agent_session_start(self, success: bool, duration: float) -> None:
        """Track an agent session start.

        Success is true if startup completed without error.
        Duration is start time in seconds observed by AgentSession.
        """
        logger.info(
            'Tracking agent session start',
            extra={
                'signal': 'saas_agent_session_start',
                'success': success,
                'duration': duration,
            },
        )

    def on_create_conversation(self) -> None:
        """Track the beginning of conversation creation.

        Does not currently capture whether it succeed.
        """
        logger.info(
            'Tracking create conversation', extra={'signal': 'saas_create_conversation'}
        )

    @classmethod
    def get_instance(
        cls,
        config: OpenHandsConfig,
    ) -> 'SaaSMonitoringListener':
        return cls()
