from openhands.sdk.event import ActionEvent
from openhands.sdk.logger import get_logger
from openhands.sdk.security.analyzer import SecurityAnalyzerBase
from openhands.sdk.security.risk import SecurityRisk


logger = get_logger(__name__)


class LLMSecurityAnalyzer(SecurityAnalyzerBase):
    """LLM-based security analyzer.

    This analyzer respects the security_risk attribute that can be set by the LLM
    when generating actions, similar to OpenHands' LLMRiskAnalyzer.

    It provides a lightweight security analysis approach that leverages the LLM's
    understanding of action context and potential risks.
    """

    def security_risk(self, action: ActionEvent) -> SecurityRisk:
        """Evaluate security risk based on LLM-provided assessment.

        This method checks if the action has a security_risk attribute set by the LLM
        and returns it. The LLM may not always provide this attribute but it defaults to
        UNKNOWN if not explicitly set.
        """
        logger.debug(f"Analyzing security risk: {action} -- {action.security_risk}")

        return action.security_risk
