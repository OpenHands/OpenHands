"""Tests for the Reviewer Agent."""

from scripts.reviewer.severity import (
    Severity,
    Issue,
    ReviewResult,
    classify_issue_auto,
)
from scripts.reviewer.comment_builder import build_comment

# ──────────────────────────────────────────────
# Severity Tests
# ──────────────────────────────────────────────


class TestSeverity:
    def test_classify_hardcoded_secret(self):
        assert classify_issue_auto("hardcoded API key found") == Severity.CRITICAL

    def test_classify_sql_injection(self):
        assert classify_issue_auto("SQL injection risk in query") == Severity.CRITICAL

    def test_classify_n_plus_one(self):
        assert classify_issue_auto("N+1 query pattern detected") == Severity.HIGH

    def test_classify_unused_import(self):
        assert classify_issue_auto("unused import 'os'") == Severity.HIGH

    def test_classify_unknown_returns_none(self):
        assert classify_issue_auto("minor style suggestion") is None


# ──────────────────────────────────────────────
# Issue Tests
# ──────────────────────────────────────────────


class TestIssue:
    def test_issue_creation(self):
        issue = Issue(
            severity=Severity.CRITICAL,
            file="src/auth.py",
            line=42,
            category="security",
            title="Hardcoded secret",
            description="API key hardcoded in source",
            suggestion="Use environment variable",
        )
        assert issue.severity == Severity.CRITICAL
        assert issue.file == "src/auth.py"
        assert issue.line == 42
        assert issue.line_str == "42"

    def test_line_str_none(self):
        issue = Issue(
            severity=Severity.MEDIUM,
            file="README.md",
            line=None,
            category="quality",
            title="Typo",
            description="Typo in docs",
        )
        assert issue.line_str == "-"


# ──────────────────────────────────────────────
# ReviewResult Tests
# ──────────────────────────────────────────────


class TestReviewResult:
    def test_empty_result_is_approve(self):
        result = ReviewResult(
            template_compliance={"passed": True, "missing": []},
        )
        assert result.summary["total"] == 0
        assert result.summary["verdict"] == "approve"

    def test_critical_issue_triggers_changes_requested(self):
        result = ReviewResult(
            template_compliance={"passed": True, "missing": []},
        )
        result.add_issue(Issue(
            severity=Severity.CRITICAL,
            file="src/app.py",
            line=10,
            category="security",
            title="Hardcoded key",
            description="Key hardcoded",
        ))
        assert result.summary["total"] == 1
        assert result.summary["critical"] == 1
        assert result.summary["verdict"] == "changes_requested"

    def test_high_issue_triggers_changes_requested(self):
        result = ReviewResult(
            template_compliance={"passed": True, "missing": []},
        )
        result.add_issue(Issue(
            severity=Severity.HIGH,
            file="src/app.py",
            line=20,
            category="quality",
            title="Unused import",
            description="Unused import 'os'",
        ))
        assert result.summary["high"] == 1
        assert result.summary["verdict"] == "changes_requested"

    def test_medium_only_is_approve(self):
        result = ReviewResult(
            template_compliance={"passed": True, "missing": []},
        )
        result.add_issue(Issue(
            severity=Severity.MEDIUM,
            file="src/app.py",
            line=30,
            category="quality",
            title="Long function",
            description="Function exceeds 50 lines",
        ))
        assert result.summary["medium"] == 1
        assert result.summary["verdict"] == "approve"

    def test_issues_by_severity(self):
        result = ReviewResult(
            template_compliance={"passed": True, "missing": []},
        )
        result.add_issue(Issue(
            severity=Severity.CRITICAL, file="a.py", line=1,
            category="security", title="S1", description="",
        ))
        result.add_issue(Issue(
            severity=Severity.HIGH, file="b.py", line=2,
            category="quality", title="S2", description="",
        ))
        result.add_issue(Issue(
            severity=Severity.CRITICAL, file="c.py", line=3,
            category="security", title="S3", description="",
        ))
        assert len(result.issues_by_severity(Severity.CRITICAL)) == 2
        assert len(result.issues_by_severity(Severity.HIGH)) == 1
        assert len(result.issues_by_severity(Severity.MEDIUM)) == 0

    def test_issues_by_category(self):
        result = ReviewResult(
            template_compliance={"passed": True, "missing": []},
        )
        result.add_issue(Issue(
            severity=Severity.HIGH, file="a.py", line=1,
            category="security", title="XSS", description="",
        ))
        result.add_issue(Issue(
            severity=Severity.MEDIUM, file="b.py", line=2,
            category="quality", title="Long func", description="",
        ))
        assert len(result.issues_by_category("security")) == 1
        assert len(result.issues_by_category("quality")) == 1
        assert len(result.issues_by_category("performance")) == 0


# ──────────────────────────────────────────────
# Comment Builder Tests
# ──────────────────────────────────────────────


class TestCommentBuilder:
    def test_empty_comment(self):
        result = ReviewResult(
            template_compliance={"passed": True, "missing": []},
        )
        comment = build_comment(result)
        assert "✅ Approve" in comment
        assert "Reviewer Agent Report" in comment

    def test_template_failure_skips_code_review(self):
        result = ReviewResult(
            template_compliance={"passed": False, "missing": ["Why", "Summary"]},
        )
        comment = build_comment(result)
        assert "❌ **Why**" in comment
        assert "❌ **Summary**" in comment
        assert "⏸️ Code review skipped" in comment

    def test_critical_issues_included(self):
        result = ReviewResult(
            template_compliance={"passed": True, "missing": []},
        )
        result.add_issue(Issue(
            severity=Severity.CRITICAL,
            file="src/auth.py",
            line=42,
            category="security",
            title="Hardcoded API key",
            description="API key found in source",
            suggestion="Move to environment variable",
        ))
        result.add_issue(Issue(
            severity=Severity.HIGH,
            file="src/db.py",
            line=100,
            category="performance",
            title="N+1 query",
            description="Loop-based query should use JOIN",
            suggestion="Use select_related()",
        ))
        comment = build_comment(result)
        assert "🔴 Critical" in comment
        assert "🟡 High" in comment
        assert "`src/auth.py`" in comment
        assert "Move to environment variable" in comment
        assert "⚠️ Changes Requested" in comment

    def test_bilingual_issues_have_section(self):
        result = ReviewResult(
            template_compliance={"passed": True, "missing": []},
        )
        result.add_issue(Issue(
            severity=Severity.LOW,
            file="docs/guide.md",
            line=15,
            category="bilingual",
            title="Spacing",
            description="Chinese/English spacing issue",
        ))
        comment = build_comment(result)
        assert "📝 Bilingual Check" in comment
