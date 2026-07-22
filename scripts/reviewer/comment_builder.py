"""
Comment Builder — formats review results into a markdown PR comment.
"""

from .severity import ReviewResult, Severity


def build_comment(result: ReviewResult) -> str:
    """Build a formatted PR comment from the review result."""
    lines = ["## 🤖 Reviewer Agent Report", ""]

    # ── Template Compliance ──
    lines.append("### PR Template Compliance")
    tc = result.template_compliance
    if tc.get("passed", False):
        lines.append("- ✅ All required fields present")
    else:
        for field in tc.get("missing", []):
            lines.append(f"- ❌ **{field}** — missing or empty")
    lines.append("")

    # If template failed and we skipped code review
    if not tc.get("passed", False) and not result.issues:
        lines.append("> ⏸️ Code review skipped — please fix the template fields above and re-trigger.")
        lines.append("")
        return "\n".join(lines)

    # ── Issues by Severity ──
    severity_order = [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM]
    severity_headers = {
        Severity.CRITICAL: "🔴 Critical",
        Severity.HIGH: "🟡 High",
        Severity.MEDIUM: "🔵 Medium",
    }

    for sev in severity_order:
        issues = result.issues_by_severity(sev)
        if not issues:
            continue
        lines.append(f"### {severity_headers[sev]} ({len(issues)} issues)")
        lines.append("| File | Line | Category | Issue |")
        lines.append("|------|------|----------|-------|")
        for issue in issues:
            file_path = f"`{issue.file}`" if issue.file else "—"
            line_num = issue.line_str
            category = issue.category.capitalize()
            # Shorten description — keep it concise for table
            desc = issue.title if issue.title else issue.description[:80]
            lines.append(f"| {file_path} | {line_num} | {category} | {desc} |")
        lines.append("")

    # ── Bilingual Issues ──
    bilingual = result.issues_by_category("bilingual")
    if bilingual:
        lines.append("### 📝 Bilingual Check")
        lines.append("| File | Line | Issue |")
        lines.append("|------|------|-------|")
        for issue in bilingual:
            file_path = f"`{issue.file}`" if issue.file else "—"
            desc = issue.description[:100]
            lines.append(f"| {file_path} | {issue.line_str} | {desc} |")
        lines.append("")

    # ── Details (for issues with suggestions) ──
    has_details = any(i.suggestion for i in result.issues)
    if has_details:
        lines.append("### 💡 Suggestions")
        lines.append("")
        for issue in result.issues:
            if issue.suggestion:
                loc = f"`{issue.file}:{issue.line_str}`" if issue.file else ""
                lines.append(f"- **{issue.title.strip('.')}** ({issue.severity.label})")
                if loc:
                    lines.append(f"  - Location: {loc}")
                lines.append(f"  - {issue.suggestion}")
                lines.append("")
        lines.append("")

    # ── Summary ──
    s = result.summary
    lines.append("### Summary")
    lines.append(f"- **Total**: {s['total']} | **Critical**: {s['critical']} | **High**: {s['high']} | **Medium**: {s['medium']}")
    verdict_icon = "✅" if s["verdict"] == "approve" else "⚠️"
    verdict_text = "Approve" if s["verdict"] == "approve" else "Changes Requested"
    lines.append(f"- **Verdict**: {verdict_icon} {verdict_text}")
    lines.append("")
    lines.append("---")
    lines.append("*Powered by the Reviewer Agent*")

    return "\n".join(lines)


def build_request_review_comment(verdict: str, reason: str = "") -> str:
    """Build a concise comment when review requires changes but we want to be brief."""
    if verdict == "approve":
        return "## ✅ Reviewer Agent\n\nThe changes look good — no blocking issues found."
    return f"## ⚠️ Reviewer Agent\n\nReview requested changes.\n\n{reason}"
