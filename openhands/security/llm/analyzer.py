# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
"""Security analyzer that uses LLM-provided risk assessments."""

from typing import Any

from fastapi import Request

from openhands.core.logger import openhands_logger as logger
from openhands.events.action.action import Action, ActionSecurityRisk
from openhands.security.analyzer import SecurityAnalyzer


class LLMRiskAnalyzer(SecurityAnalyzer):
    """Security analyzer that respects LLM-provided risk assessments."""

    async def handle_api_request(self, request: Request) -> Any:
        """Handles the incoming API request."""
        return {'status': 'ok'}

    async def security_risk(self, action: Action) -> ActionSecurityRisk:
        """Evaluates the Action for security risks and returns the risk level.

        This analyzer checks if the action has a 'security_risk' attribute set by the LLM.
        If it does, it uses that value. Otherwise, it returns UNKNOWN.
        """
        # Check if the action has a security_risk attribute set by the LLM
        if not hasattr(action, 'security_risk'):
            return ActionSecurityRisk.UNKNOWN

        security_risk = getattr(action, 'security_risk')

        if security_risk in {
            ActionSecurityRisk.LOW,
            ActionSecurityRisk.MEDIUM,
            ActionSecurityRisk.HIGH,
        }:
            return security_risk
        elif security_risk == ActionSecurityRisk.UNKNOWN:
            return ActionSecurityRisk.UNKNOWN
        else:
            # Default to UNKNOWN if security_risk value is not recognized
            logger.warning(f'Unrecognized security_risk value: {security_risk}')
            return ActionSecurityRisk.UNKNOWN
