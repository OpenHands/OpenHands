from __future__ import annotations

from enum import Enum

from rich.text import Text


class SecurityRisk(str, Enum):
    """Security risk levels for actions.

    Based on OpenHands security risk levels but adapted for agent-sdk.
    Integer values allow for easy comparison and ordering.
    """

    UNKNOWN = "UNKNOWN"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

    @property
    def description(self) -> str:
        """Get a human-readable description of the risk level."""
        descriptions = {
            SecurityRisk.LOW: (
                "Low risk - Safe operation with minimal security impact"
            ),
            SecurityRisk.MEDIUM: (
                "Medium risk - Moderate security impact, review recommended"
            ),
            SecurityRisk.HIGH: (
                "High risk - Significant security impact, confirmation required"
            ),
            SecurityRisk.UNKNOWN: ("Unknown risk - Risk level could not be determined"),
        }
        return descriptions.get(self, "Unknown risk level")

    def __str__(self) -> str:
        return self.name

    def get_color(self) -> str:
        """Get the color for displaying this risk level in Rich text."""
        color_map = {
            SecurityRisk.LOW: "green",
            SecurityRisk.MEDIUM: "yellow",
            SecurityRisk.HIGH: "red",
            SecurityRisk.UNKNOWN: "white",
        }
        return color_map.get(self, "white")

    @property
    def visualize(self) -> Text:
        """Return Rich Text representation of this risk level."""
        content = Text()
        content.append(
            "Predicted Security Risk: ",
            style="bold",
        )
        content.append(
            f"{self.value}\n\n",
            style=f"bold {self.get_color()}",
        )
        return content

    def is_riskier(self, other: SecurityRisk, reflexive: bool = True) -> bool:
        """Check if this risk level is riskier than another.

        Risk levels follow the natural ordering: LOW is less risky than MEDIUM, which is
        less risky than HIGH. UNKNOWN is not comparable to any other level.

        To make this act like a standard well-ordered domain, we reflexively consider
        risk levels to be riskier than themselves. That is:

            for risk_level in list(SecurityRisk):
                assert risk_level.is_riskier(risk_level)

            # More concretely:
            assert SecurityRisk.HIGH.is_riskier(SecurityRisk.HIGH)
            assert SecurityRisk.MEDIUM.is_riskier(SecurityRisk.MEDIUM)
            assert SecurityRisk.LOW.is_riskier(SecurityRisk.LOW)

        This can be disabled by setting the `reflexive` parameter to False.

        Args:
            other (SecurityRisk): The other risk level to compare against.
            reflexive (bool): Whether the relationship is reflexive.

        Raises:
            ValueError: If either risk level is UNKNOWN.
        """
        if self.value == SecurityRisk.UNKNOWN or other.value == SecurityRisk.UNKNOWN:
            raise ValueError("Cannot compare unknown risk levels.")

        # Map risk levels to a well-ordered domain for comparison. No need to map
        # UNKNOWN since we'll already have raised an error by now if either is UNKNOWN.
        risk_order = {
            SecurityRisk.LOW: 1,
            SecurityRisk.MEDIUM: 2,
            SecurityRisk.HIGH: 3,
        }
        return risk_order[self] > risk_order[other] or (reflexive and self == other)
