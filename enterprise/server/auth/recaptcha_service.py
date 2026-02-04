import hashlib
import hmac
from dataclasses import dataclass

from google.cloud import recaptchaenterprise_v1
from server.auth.constants import (
    RECAPTCHA_BLOCK_THRESHOLD,
    RECAPTCHA_HMAC_SECRET,
    RECAPTCHA_PROJECT_ID,
    RECAPTCHA_SITE_KEY,
    SUSPICIOUS_LABELS,
)

from openhands.core.logger import openhands_logger as logger


@dataclass
class AssessmentResult:
    """Result of a reCAPTCHA Enterprise assessment."""

    name: str
    score: float
    valid: bool
    action_valid: bool
    reason_codes: list[str]
    account_defender_labels: list[str]
    allowed: bool


class RecaptchaService:
    """Service for creating reCAPTCHA Enterprise assessments."""

    def __init__(self):
        self._client = None
        self.project_id = RECAPTCHA_PROJECT_ID
        self.site_key = RECAPTCHA_SITE_KEY

    @property
    def client(self):
        """Lazily initialize the reCAPTCHA client to avoid credential errors at import time."""
        if self._client is None:
            self._client = recaptchaenterprise_v1.RecaptchaEnterpriseServiceClient()
        return self._client

    def hash_account_id(self, email: str) -> str:
        """Hash email using SHA256-HMAC for Account Defender.

        Args:
            email: The user's email address.

        Returns:
            Hex-encoded HMAC-SHA256 hash of the lowercase email.
        """
        return hmac.new(
            RECAPTCHA_HMAC_SECRET.encode(),
            email.lower().encode(),
            hashlib.sha256,
        ).hexdigest()

    def create_assessment(
        self,
        token: str,
        action: str,
        user_ip: str,
        user_agent: str,
        email: str | None = None,
    ) -> AssessmentResult:
        """Create a reCAPTCHA Enterprise assessment.

        Args:
            token: The reCAPTCHA token from the frontend.
            action: The expected action name (e.g., 'LOGIN').
            user_ip: The user's IP address.
            user_agent: The user's browser user agent.
            email: Optional email for Account Defender hashing.

        Returns:
            AssessmentResult with score, validity, and allowed status.
        """
        event = recaptchaenterprise_v1.Event()
        event.site_key = self.site_key
        event.token = token
        event.user_ip_address = user_ip
        event.user_agent = user_agent
        event.expected_action = action

        # Account Defender: use user_info.account_id (hashed_account_id is deprecated)
        if email:
            user_info = recaptchaenterprise_v1.UserInfo()
            user_info.account_id = self.hash_account_id(email)
            # Also include email as a user identifier for better fraud detection
            user_info.user_ids.append(recaptchaenterprise_v1.UserId(email=email))
            event.user_info = user_info

        assessment = recaptchaenterprise_v1.Assessment()
        assessment.event = event

        request = recaptchaenterprise_v1.CreateAssessmentRequest()
        request.assessment = assessment
        request.parent = f'projects/{self.project_id}'

        response = self.client.create_assessment(request)

        # Capture assessment name for potential annotation later
        # Format: projects/{project_id}/assessments/{assessment_id}
        assessment_name = response.name

        token_properties = response.token_properties
        risk_analysis = response.risk_analysis

        score = risk_analysis.score
        valid = token_properties.valid
        action_valid = token_properties.action == action
        reason_codes = [str(r) for r in risk_analysis.reasons]

        # Extract Account Defender labels
        account_defender_labels = []
        if response.account_defender_assessment:
            account_defender_labels = [
                str(label) for label in response.account_defender_assessment.labels
            ]

        # Check if any suspicious labels are present
        has_suspicious_labels = bool(set(account_defender_labels) & SUSPICIOUS_LABELS)

        # Block if: invalid token, wrong action, low score, OR suspicious Account Defender labels
        allowed = (
            valid
            and action_valid
            and score >= RECAPTCHA_BLOCK_THRESHOLD
            and not has_suspicious_labels
        )

        logger.info(
            'recaptcha_assessment',
            extra={
                'assessment_name': assessment_name,
                'score': score,
                'valid': valid,
                'action_valid': action_valid,
                'reasons': reason_codes,
                'account_defender_labels': account_defender_labels,
                'has_suspicious_labels': has_suspicious_labels,
                'allowed': allowed,
                'user_ip': user_ip,
            },
        )

        return AssessmentResult(
            name=assessment_name,
            score=score,
            valid=valid,
            action_valid=action_valid,
            reason_codes=reason_codes,
            account_defender_labels=account_defender_labels,
            allowed=allowed,
        )


recaptcha_service = RecaptchaService()
