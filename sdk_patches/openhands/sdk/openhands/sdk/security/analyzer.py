from abc import ABC, abstractmethod

from openhands.sdk.event.base import Event
from openhands.sdk.event.llm_convertible import ActionEvent
from openhands.sdk.logger import get_logger
from openhands.sdk.security.risk import SecurityRisk
from openhands.sdk.utils.models import (
    DiscriminatedUnionMixin,
)


logger = get_logger(__name__)


class SecurityAnalyzerBase(DiscriminatedUnionMixin, ABC):
    """Abstract base class for security analyzers.

    Security analyzers evaluate the risk of actions before they are executed
    and can influence the conversation flow based on security policies.

    This is adapted from OpenHands SecurityAnalyzer but designed to work
    with the agent-sdk's conversation-based architecture.
    """

    @abstractmethod
    def security_risk(self, action: ActionEvent) -> SecurityRisk:
        """Evaluate the security risk of an ActionEvent.

        This is the core method that analyzes an ActionEvent and returns its risk level.
        Implementations should examine the action's content, context, and potential
        impact to determine the appropriate risk level.

        Args:
            action: The ActionEvent to analyze for security risks

        Returns:
            ActionSecurityRisk enum indicating the risk level
        """
        pass

    def analyze_event(self, event: Event) -> SecurityRisk | None:
        """Analyze an event for security risks.

        This is a convenience method that checks if the event is an action
        and calls security_risk() if it is. Non-action events return None.

        Args:
            event: The event to analyze

        Returns:
            ActionSecurityRisk if event is an action, None otherwise
        """
        if isinstance(event, ActionEvent):
            return self.security_risk(event)
        return None

    def should_require_confirmation(
        self, risk: SecurityRisk, confirmation_mode: bool = False
    ) -> bool:
        """Determine if an action should require user confirmation.

        This implements the default confirmation logic based on risk level
        and confirmation mode settings.

        Args:
            risk: The security risk level of the action
            confirmation_mode: Whether confirmation mode is enabled

        Returns:
            True if confirmation is required, False otherwise
        """
        if risk == SecurityRisk.HIGH:
            # HIGH risk actions always require confirmation
            return True
        elif risk == SecurityRisk.UNKNOWN and not confirmation_mode:
            # UNKNOWN risk requires confirmation if no security analyzer is configured
            return True
        elif confirmation_mode:
            # In confirmation mode, all actions require confirmation
            return True
        else:
            # LOW and MEDIUM risk actions don't require confirmation by default
            return False

    def analyze_pending_actions(
        self, pending_actions: list[ActionEvent]
    ) -> list[tuple[ActionEvent, SecurityRisk]]:
        """Analyze all pending actions in a conversation.

        This method gets all unmatched actions from the conversation state
        and analyzes each one for security risks.

        Args:
            conversation: The conversation to analyze

        Returns:
            List of tuples containing (action, risk_level) for each pending action
        """
        analyzed_actions = []

        for action_event in pending_actions:
            try:
                risk = self.security_risk(action_event)
                analyzed_actions.append((action_event, risk))
                logger.debug(f"Action {action_event} analyzed with risk level: {risk}")
            except Exception as e:
                logger.error(f"Error analyzing action {action_event}: {e}")
                # Default to HIGH risk on analysis error for safety
                analyzed_actions.append((action_event, SecurityRisk.HIGH))

        return analyzed_actions
