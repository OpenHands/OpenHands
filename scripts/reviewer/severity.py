"""
Severity level definitions for the Reviewer Agent.

Defines issue severity levels, verdict determination, and automatic
classification rules based on issue patterns.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class Severity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    def __str__(self) -> str:
        return self.value

    @property
    def label(self) -> str:
        labels = {
            Severity.CRITICAL: "🔴 Critical",
            Severity.HIGH: "🟡 High",
            Severity.MEDIUM: "🔵 Medium",
            Severity.LOW: "⚪ Low",
        }
        return labels[self]

    @property
    def sort_order(self) -> int:
        return list(Severity).index(self)


@dataclass
class Issue:
    severity: Severity
    file: str
    line: Optional[int]
    category: str  # security, quality, performance, bilingual, template
    title: str
    description: str
    suggestion: Optional[str] = None

    @property
    def line_str(self) -> str:
        return str(self.line) if self.line is not None else "-"


@dataclass
class ReviewResult:
    template_compliance: dict  # {"passed": bool, "missing": list[str]}
    issues: list[Issue] = field(default_factory=list)
    summary: dict = field(default_factory=lambda: {
        "total": 0,
        "critical": 0,
        "high": 0,
        "medium": 0,
        "verdict": "approve",
    })

    def add_issue(self, issue: Issue) -> None:
        self.issues.append(issue)
        self.summary["total"] += 1
        if issue.severity == Severity.CRITICAL:
            self.summary["critical"] += 1
        elif issue.severity == Severity.HIGH:
            self.summary["high"] += 1
        elif issue.severity == Severity.MEDIUM:
            self.summary["medium"] += 1
        self._update_verdict()

    def _update_verdict(self) -> None:
        if self.summary["critical"] > 0 or self.summary["high"] > 0:
            self.summary["verdict"] = "changes_requested"
        else:
            self.summary["verdict"] = "approve"

    def issues_by_severity(self, severity: Severity) -> list[Issue]:
        return [i for i in self.issues if i.severity == severity]

    def issues_by_category(self, category: str) -> list[Issue]:
        return [i for i in self.issues if i.category == category]


def classify_issue_auto(issue_text: str) -> Optional[Severity]:
    """Auto-classify severity based on keywords in the issue description.

    This is a lightweight heuristic used when the LLM doesn't provide a
    severity rating. Returns None when uncertain (LLM judgement preferred).
    """
    text_lower = issue_text.lower()

    critical_patterns = [
        "hardcoded secret", "hardcoded key", "hardcoded password",
        "hardcoded token", "api key", "sql injection", "xss",
        "command injection", "path traversal", "auth bypass",
        "authentication bypass", "cryptographic key", "data leak",
        "data loss", "csrf", "race condition",
    ]
    for pattern in critical_patterns:
        if pattern in text_lower:
            return Severity.CRITICAL

    high_patterns = [
        "n+1", "missing pagination", "no error handling",
        "unbounded query", "no type hint", "mutation",
        "dead code", "unused import", "no test",
    ]
    for pattern in high_patterns:
        if pattern in text_lower:
            return Severity.HIGH

    return None
