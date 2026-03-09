# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
from typing import Any

from fastapi import Request

from openhands.events.action.action import Action, ActionSecurityRisk


class SecurityAnalyzer:
    """Security analyzer that analyzes agent actions for security risks."""

    def __init__(self) -> None:
        """Initializes a new instance of the SecurityAnalyzer class."""
        pass

    async def handle_api_request(self, request: Request) -> Any:
        """Handles the incoming API request."""
        raise NotImplementedError(
            'Need to implement handle_api_request method in SecurityAnalyzer subclass'
        )

    async def security_risk(self, action: Action) -> ActionSecurityRisk:
        """Evaluates the Action for security risks and returns the risk level."""
        raise NotImplementedError(
            'Need to implement security_risk method in SecurityAnalyzer subclass'
        )

    def set_event_stream(self, event_stream) -> None:
        """Set the event stream for accessing conversation history.

        Args:
            event_stream: EventStream instance for accessing events
        """
        pass

    async def close(self) -> None:
        """Cleanup resources allocated by the SecurityAnalyzer."""
        pass
