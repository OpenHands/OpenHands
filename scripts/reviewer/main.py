#!/usr/bin/env python3
"""
Reviewer Agent — Main Entry Point.

Orchestrates the full PR review workflow:
1. Parse PR info (from GitHub Actions env or CLI args)
2. Check PR template compliance (fast, pattern-based)
3. If compliant, run LLM-based multi-dimensional review
4. Post results as a PR comment
5. Optionally set PR review status (approve / changes_requested)

Usage:
  python -m scripts.reviewer.main  # in GitHub Actions
  python -m scripts.reviewer.main --pr 42 --repo owner/repo --token ghp_xxx  # local test
"""

import argparse
import json
import os
import sys
from typing import Optional

import requests

from .pr_analyzer import PRAnalyzer
from .review_engine import ReviewEngine
from .comment_builder import build_comment
from .severity import ReviewResult


def post_comment(token: str, repo: str, pr_number: int, comment: str) -> bool:
    """Post a comment on the PR via GitHub API."""
    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    resp = requests.post(url, headers=headers, json={"body": comment})
    if resp.status_code not in (200, 201):
        print(f"[Reviewer] Failed to post comment: {resp.status_code} {resp.text[:200]}")
        return False
    print(f"[Reviewer] Comment posted: {resp.json().get('html_url', '')}")
    return True


def set_review_status(
    token: str,
    repo: str,
    pr_number: int,
    verdict: str,
    commit_sha: str,
) -> bool:
    """Set the PR review status (approve or request changes).

    Requires the pull-requests: write permission.
    """
    if verdict not in ("approve", "changes_requested"):
        return False

    event = "APPROVE" if verdict == "approve" else "REQUEST_CHANGES"
    body = (
        "✅ Reviewer Agent: No blocking issues found."
        if verdict == "approve"
        else "⚠️ Reviewer Agent: Critical or high issues found — changes requested."
    )

    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/reviews"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    payload = {
        "commit_id": commit_sha,
        "event": event,
        "body": body,
    }
    resp = requests.post(url, headers=headers, json=payload)
    if resp.status_code not in (200, 201):
        print(f"[Reviewer] Failed to set review status: {resp.status_code} {resp.text[:200]}")
        return False
    print(f"[Reviewer] Review status set to {event}")
    return True


def run_review(
    token: str,
    repo: str,
    pr_number: int,
    llm_api_key: str,
    llm_model: str = "gpt-4o",
    llm_base_url: Optional[str] = None,
    post_comment_flag: bool = True,
    set_status: bool = False,
) -> ReviewResult:
    """Run the full review pipeline."""
    # Step 1: Analyze PR
    print(f"[Reviewer] Analyzing PR #{pr_number} in {repo}...")
    analyzer = PRAnalyzer(token=token, repo=repo, pr_number=pr_number)
    pr_meta = analyzer.get_pr_metadata()
    print(f"[Reviewer] Title: {pr_meta['title']}")
    print(f"[Reviewer] Files changed: {pr_meta['changed_files']}, "
          f"++{pr_meta['additions']} --{pr_meta['deletions']}")

    # Step 2: Template compliance check (fast, pattern-based)
    print("[Reviewer] Checking PR template compliance...")
    template_result = analyzer.check_template_compliance()

    if not template_result["passed"]:
        print(f"[Reviewer] Template compliance FAILED — missing: {template_result['missing']}")
        result = ReviewResult(template_compliance=template_result)
        # Add a template issue
        from .severity import Issue, Severity
        for field in template_result["missing"]:
            result.add_issue(Issue(
                severity=Severity.HIGH,
                file="PR Description",
                line=None,
                category="template",
                title=f"Missing template field: {field}",
                description=f"The PR template requires a '{field}' section.",
                suggestion=f"Add a '## {field}' section to the PR description.",
            ))

        if post_comment_flag:
            comment = build_comment(result)
            post_comment(token, repo, pr_number, comment)
        return result

    print("[Reviewer] Template compliance PASSED")

    # Step 3: Get diff
    print("[Reviewer] Fetching PR diff...")
    diff_text = analyzer.get_diff()
    print(f"[Reviewer] Diff size: {len(diff_text)} chars")

    # Step 4: LLM review
    print(f"[Reviewer] Running LLM review (model: {llm_model})...")
    engine = ReviewEngine(api_key=llm_api_key, model=llm_model, base_url=llm_base_url)
    result = engine.review(diff_text, pr_meta)

    print(f"[Reviewer] Review complete — {result.summary['total']} issues found, "
          f"verdict: {result.summary['verdict']}")

    # Step 5: Post comment
    if post_comment_flag:
        comment = build_comment(result)
        post_comment(token, repo, pr_number, comment)
        print(f"[Reviewer] Comment posted to PR #{pr_number}")

    # Step 6: Set review status (optional)
    if set_status:
        commit_sha = pr_meta.get("head_sha", "")
        if commit_sha:
            set_review_status(token, repo, pr_number, result.summary["verdict"], commit_sha)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Reviewer Agent — automated PR review")
    parser.add_argument("--pr", type=int, help="PR number")
    parser.add_argument("--repo", help="Repository (owner/repo)")
    parser.add_argument("--token", help="GitHub token")
    parser.add_argument("--llm-api-key", help="LLM API key")
    parser.add_argument("--llm-model", default="gpt-4o", help="LLM model name")
    parser.add_argument("--llm-base-url", help="LLM API base URL (for proxies)")
    parser.add_argument("--set-status", action="store_true", help="Set PR review status")
    parser.add_argument("--dry-run", action="store_true", help="Print comment without posting")

    args = parser.parse_args()

    # Resolve parameters (env → file → CLI)
    token = args.token or os.environ.get("REVIEWER_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
    repo = args.repo or os.environ.get("GITHUB_REPOSITORY", "")
    pr_number = args.pr

    if not pr_number and os.environ.get("GITHUB_EVENT_PATH"):
        try:
            with open(os.environ["GITHUB_EVENT_PATH"]) as f:
                event = json.load(f)
            pr_number = (
                event.get("pull_request", {}).get("number")
                or event.get("issue", {}).get("number")
            )
        except (json.JSONDecodeError, FileNotFoundError):
            pass

    if not token or not repo or not pr_number:
        print("[Reviewer] Missing required params: token, repo, and pr_number")
        print("[Reviewer] Provide via CLI args or environment variables")
        sys.exit(1)

    llm_api_key = args.llm_api_key or os.environ.get("REVIEWER_LLM_API_KEY", token)
    llm_base_url = args.llm_base_url or os.environ.get("REVIEWER_LLM_BASE_URL")

    if args.dry_run:
        # Just build and print the comment, don't post
        analyzer = PRAnalyzer(token=token, repo=repo, pr_number=pr_number)
        pr_meta = analyzer.get_pr_metadata()
        diff_text = analyzer.get_diff()
        engine = ReviewEngine(api_key=llm_api_key, model=args.llm_model, base_url=llm_base_url)
        result = engine.review(diff_text, pr_meta)
        comment = build_comment(result)
        print(comment)
        return

    run_review(
        token=token,
        repo=repo,
        pr_number=pr_number,
        llm_api_key=llm_api_key,
        llm_model=args.llm_model,
        llm_base_url=llm_base_url,
        post_comment_flag=True,
        set_status=args.set_status,
    )


if __name__ == "__main__":
    main()
