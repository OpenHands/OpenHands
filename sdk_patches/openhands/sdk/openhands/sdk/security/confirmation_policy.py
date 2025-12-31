from abc import ABC, abstractmethod

from pydantic import field_validator

from openhands.sdk.security.risk import SecurityRisk
from openhands.sdk.utils.models import DiscriminatedUnionMixin


class ConfirmationPolicyBase(DiscriminatedUnionMixin, ABC):
    @abstractmethod
    def should_confirm(self, risk: SecurityRisk = SecurityRisk.UNKNOWN) -> bool:
        """Determine if an action with the given risk level requires confirmation.

        This method defines the core logic for determining whether user confirmation
        is required before executing an action based on its security risk level.

        Args:
            risk: The security risk level of the action to be evaluated.
                 Defaults to SecurityRisk.UNKNOWN if not specified.

        Returns:
            True if the action requires user confirmation before execution,
            False if the action can proceed without confirmation.
        """


class AlwaysConfirm(ConfirmationPolicyBase):
    def should_confirm(
        self,
        risk: SecurityRisk = SecurityRisk.UNKNOWN,  # noqa: ARG002
    ) -> bool:
        return True


class NeverConfirm(ConfirmationPolicyBase):
    def should_confirm(
        self,
        risk: SecurityRisk = SecurityRisk.UNKNOWN,  # noqa: ARG002
    ) -> bool:
        return False


class ConfirmRisky(ConfirmationPolicyBase):
    threshold: SecurityRisk = SecurityRisk.HIGH
    confirm_unknown: bool = True

    @field_validator("threshold")
    def validate_threshold(cls, v: SecurityRisk) -> SecurityRisk:
        if v == SecurityRisk.UNKNOWN:
            raise ValueError("Threshold cannot be UNKNOWN")
        return v

    def should_confirm(self, risk: SecurityRisk = SecurityRisk.UNKNOWN) -> bool:
        if risk == SecurityRisk.UNKNOWN:
            return self.confirm_unknown

        # This comparison is reflexive by default, so if the threshold is HIGH we will
        # still require confirmation for HIGH risk actions. And since the threshold is
        # guaranteed to never be UNKNOWN (by the validator), we're guaranteed to get a
        # boolean here.
        return risk.is_riskier(self.threshold)
