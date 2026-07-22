"""
Review Engine — calls an LLM to perform multi-dimensional code review.

Supports both OpenAI-compatible API and a lightweight built-in pattern scanner
for template compliance and basic security checks (to minimize LLM cost).
"""

import json
import os
from typing import Any, Optional

from openai import OpenAI

from .severity import ReviewResult, Issue, Severity


# Jinja2-like prompt template for the LLM review call
REVIEW_SYSTEM_PROMPT = """You are the Reviewer Agent for the OpenHands project, an automated AI software engineer platform written in Python (backend) and TypeScript/React (frontend).

Your task is to analyze the provided PR diff and produce a structured, actionable code review.

## Review Dimensions

1. **Security** — hardcoded secrets, SQL injection, XSS, path traversal, CSRF, insecure crypto
2. **Code Quality** — function length (>50 lines), file length (>800 lines), nesting (>4 levels), dead code, naming, error handling, mutation
3. **Performance** — N+1 queries, missing pagination, unnecessary computation, large payloads, bundle impact
4. **Bilingual Check** — Chinese/English mixed spacing, pinyin comments, term consistency

## PR Template Compliance
Check: Why, Summary, How to Test, Type fields present and non-empty.
Type must have at least one checkbox selected ([x]).

## Output Format
Respond with a JSON object exactly as described below — no markdown wrapping, no extra text.

{
  "template_compliance": {
    "passed": true/false,
    "missing": ["field names..."]
  },
  "issues": [
    {
      "severity": "critical" | "high" | "medium" | "low",
      "file": "path/to/file.py",
      "line": 42,
      "category": "security" | "quality" | "performance" | "bilingual",
      "title": "Short description",
      "description": "Detailed explanation",
      "suggestion": "How to fix"
    }
  ],
  "summary": {
    "total": 0,
    "critical": 0,
    "high": 0,
    "medium": 0,
    "verdict": "approve" | "changes_requested"
  }
}

## Rules
- CRITICAL = security vuln, data loss, hardcoded secret → changes_requested
- HIGH = functional bug, major quality issue → changes_requested
- MEDIUM = maintainability concern → advisory only
- LOW = style suggestion → advisory only
- Never issue CRITICAL for opinion/style issues
- Only flag actual problems; don't invent issues
- If the diff is clean, return an empty issues array with approve
"""


class ReviewEngine:
    """Multi-dimensional code review engine backed by an LLM."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("REVIEWER_LLM_API_KEY", "")
        self.model = model
        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)

    def review(
        self,
        diff_text: str,
        pr_metadata: dict[str, Any],
        max_diff_chars: int = 15000,
    ) -> ReviewResult:
        """Run a full LLM-based review on the PR diff."""
        # Truncate diff if too large
        truncated = diff_text[:max_diff_chars]
        if len(diff_text) > max_diff_chars:
            truncated += f"\n\n... [diff truncated at {max_diff_chars} chars; {len(diff_text)} total]"

        user_prompt = f"""## PR Metadata
- Title: {pr_metadata.get('title', 'N/A')}
- Author: {pr_metadata.get('author', 'N/A')}
- Changed files: {pr_metadata.get('changed_files', 'N/A')}
- Additions: {pr_metadata.get('additions', 'N/A')}
- Deletions: {pr_metadata.get('deletions', 'N/A')}

## PR Description
{pr_metadata.get('body', 'N/A')[:2000]}

## Diff
```diff
{truncated}
```
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=4096,
            )

            content = response.choices[0].message.content
            if not content:
                return self._fallback_result("Empty LLM response")

            data = json.loads(content)

        except Exception as e:
            return self._fallback_result(f"LLM review failed: {e}")

        return self._parse_result(data)

    def _parse_result(self, data: dict) -> ReviewResult:
        """Parse LLM JSON response into a ReviewResult."""
        tc = data.get("template_compliance", {})
        result = ReviewResult(
            template_compliance={
                "passed": tc.get("passed", False),
                "missing": tc.get("missing", []),
            }
        )

        for item in data.get("issues", []):
            try:
                severity = Severity(item.get("severity", "medium").lower())
            except ValueError:
                severity = Severity.MEDIUM

            issue = Issue(
                severity=severity,
                file=item.get("file", ""),
                line=item.get("line"),
                category=item.get("category", "quality"),
                title=item.get("title", ""),
                description=item.get("description", ""),
                suggestion=item.get("suggestion"),
            )
            result.add_issue(issue)

        return result

    def _fallback_result(self, reason: str) -> ReviewResult:
        """Create a minimal result when the LLM call fails."""
        result = ReviewResult(
            template_compliance={"passed": True, "missing": []},
        )
        # Log the failure but don't block the PR
        print(f"[Reviewer] LLM review skipped — {reason}")
        return result

    def quick_template_check(self, pr_body: str) -> dict:
        """Lightweight template compliance check without LLM.

        This is a fast pattern-based check that runs before the full review.
        """
        required = ["Why", "Summary", "How to Test", "Type"]
        missing = []
        for field in required:
            if f"## {field}" not in pr_body:
                missing.append(field)

        # Check Type checkbox
        if "## Type" in pr_body:
            has_selection = any(
                line.strip().startswith(("- [x]", "- [X]"))
                for line in pr_body.split("\n")
            )
            if not has_selection:
                missing.append("Type (no selection)")

        return {
            "passed": len(missing) == 0,
            "missing": missing,
        }
