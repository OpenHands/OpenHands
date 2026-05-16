#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


OPENHANDS_BASE_URL = os.environ.get("OPENHANDS_BASE_URL", "https://app.all-hands.dev")
REPOSITORY_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$")
GITHUB_API_BASE_URL = os.environ.get("GITHUB_API_BASE_URL", "https://api.github.com")
FAILED_EXECUTION_STATUSES = {
    "error",
    "errored",
    "failed",
    "stopped",
}
SUCCESSFUL_TERMINAL_EXECUTION_STATUSES = {
    "completed",
    "finished",
}
TERMINAL_EXECUTION_STATUSES = (
    FAILED_EXECUTION_STATUSES | SUCCESSFUL_TERMINAL_EXECUTION_STATUSES
)
EVENT_SEARCH_LIMIT = 1000
EVENT_SEARCH_LIMIT_HIT_MESSAGE = (
    f"Event search returned at least {EVENT_SEARCH_LIMIT} events; results may be "
    "incomplete"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start an OpenHands Cloud conversation that checks a GitHub issue "
            "for duplicates."
        )
    )
    parser.add_argument(
        "--repository", required=True, help="Repository in owner/repo form"
    )
    parser.add_argument(
        "--issue-number", required=True, type=int, help="Issue number to inspect"
    )
    parser.add_argument(
        "--output",
        default="duplicate-check-result.json",
        help="Path where the JSON result should be written",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        default=5,
        type=int,
        help="Polling interval while waiting for the conversation to finish",
    )
    parser.add_argument(
        "--max-wait-seconds",
        default=900,
        type=int,
        help=(
            "Maximum time to wait per polling phase; if a start task must be awaited "
            "first, the total runtime can approach twice this value"
        ),
    )
    return parser.parse_args()


def github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "openhands-issue-duplicate-check",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    return headers


def openhands_headers() -> dict[str, str]:
    api_key = os.environ.get("OPENHANDS_API_KEY")
    if not api_key:
        raise RuntimeError("OPENHANDS_API_KEY environment variable is required")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def request_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
) -> Any:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers=headers or {},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"{method} {base_url}{path} failed with HTTP {exc.code}: {error_body}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to parse JSON from {method} {base_url}{path}: {exc}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{method} {base_url}{path} failed: {exc}") from exc


def fetch_issue(repository: str, issue_number: int) -> dict[str, Any]:
    if not REPOSITORY_PATTERN.fullmatch(repository):
        raise ValueError(f"Invalid repository format: {repository}")
    return request_json(
        GITHUB_API_BASE_URL,
        f"/repos/{repository}/issues/{issue_number}",
        headers=github_headers(),
    )


def escape_json_text(value: str | None) -> str:
    return json.dumps(value or "", ensure_ascii=False)


def build_prompt(repository: str, issue: dict[str, Any]) -> str:
    issue_number = issue["number"]
    issue_title = issue.get("title", "")
    issue_body = issue.get("body") or ""
    issue_url = issue.get("html_url", "")
    issue_title_json = escape_json_text(issue_title)
    issue_body_json = escape_json_text(issue_body)

    return "\n".join(
        [
            "You are investigating whether a GitHub issue should be redirected "
            "to an existing issue because it is either:",
            "- an exact or near-exact duplicate, or",
            "- so overlapping in scope that discussion or fix planning would "
            "likely be better kept in one canonical issue.",
            "",
            "Be conservative about auto-close decisions, but do investigate "
            "seriously before deciding.",
            "",
            f"Repository: {repository}",
            f"New issue number: #{issue_number}",
            f"New issue URL: {issue_url}",
            f"New issue title (JSON-escaped string): {issue_title_json}",
            f"New issue body (JSON-escaped string): {issue_body_json}",
            "",
            "Task:",
            "1. Understand the core problem, user-facing outcome, likely root "
            "cause, and requested fix or behavior.",
            "2. Investigate this repository's open issues and issues closed "
            "in the last 90 days for exact duplicates, near-duplicates, or "
            "strong scope overlap.",
            "3. Use multiple search approaches with diverse keywords and "
            "phrasings rather than a single literal search.",
            "4. Ignore pull requests.",
            "5. Distinguish carefully between:",
            "   - duplicate: essentially the same report, request, or root cause",
            "   - overlapping-scope: not identical, but likely to fragment "
            "discussion or produce competing fixes",
            "   - related-but-distinct: similar area, but should stay separate",
            "   - no-match: no strong candidate worth redirecting to",
            "6. Inspect the strongest 1-3 candidates carefully. If needed, "
            "inspect comments on the strongest candidates to disambiguate "
            "false positives.",
            "7. Do not post comments, do not modify files, and do not change "
            "repository state.",
            "8. Useful API shapes include:",
            f"   - GET https://api.github.com/repos/{repository}/issues?state=open&per_page=100",
            "   - GET https://api.github.com/repos/"
            f"{repository}/issues?state=closed&since=<ISO-8601 timestamp>&per_page=100",
            "   - GET https://api.github.com/search/issues?q=<query>",
            f"   - GET https://api.github.com/repos/{repository}/issues/<number>/comments",
            "9. Return exactly one JSON object and nothing else. Do not wrap "
            "it in markdown fences.",
            "",
            "Return schema:",
            "{",
            f'  "issue_number": {issue_number},',
            '  "should_comment": true or false,',
            '  "is_duplicate": true or false,',
            '  "auto_close_candidate": true or false,',
            '  "classification": "duplicate" | "overlapping-scope" | '
            '"related-but-distinct" | "no-match",',
            '  "confidence": "high" | "medium" | "low",',
            '  "summary": "short explanation",',
            '  "canonical_issue_number": 123 or null,',
            '  "candidate_issues": [',
            "    {",
            '      "number": 123,',
            f'      "url": "https://github.com/{repository}/issues/123",',
            '      "title": "issue title",',
            '      "state": "open or closed",',
            '      "closed_at": "ISO timestamp or null",',
            '      "similarity_reason": "why it looks similar"',
            "    }",
            "  ]",
            "}",
            "",
            "Rules:",
            "- `should_comment` should be true only when redirecting the "
            "author would likely help.",
            "- `is_duplicate` should be true only for exact or near-exact duplicates.",
            "- `auto_close_candidate` should be true only when:",
            "  - classification is `duplicate`",
            "  - confidence is `high`",
            "  - one canonical issue clearly stands out",
            "  - a maintainer would likely be comfortable closing this issue "
            "after a waiting period",
            "- For `overlapping-scope`, `auto_close_candidate` must be false.",
            "- `candidate_issues` must contain at most 3 issues, sorted best-first.",
            "- If no strong match exists, return `should_comment: false`, "
            '`classification: "no-match"`, `canonical_issue_number: null`, '
            "and an empty candidate list.",
            "- Be especially careful not to collapse broad meta, tracking, "
            "feedback, or umbrella issues with specific bug reports unless "
            "the new issue clearly belongs in that exact thread.",
        ]
    )


def start_conversation(
    prompt: str, repository: str, issue_number: int
) -> dict[str, Any]:
    body = {
        "title": f"Issue duplicate check #{issue_number}",
        "selected_repository": repository,
        "initial_message": {
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                }
            ]
        },
    }
    return request_json(
        OPENHANDS_BASE_URL,
        "/api/v1/app-conversations",
        method="POST",
        headers=openhands_headers(),
        body=body,
    )


def extract_first_item(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, list):
        first_item = payload[0] if payload else None
        return first_item if isinstance(first_item, dict) else None
    if not isinstance(payload, dict):
        return None

    items = payload.get("items")
    if isinstance(items, list):
        first_item = items[0] if items else None
        return first_item if isinstance(first_item, dict) else None
    return payload


def poll_start_task(
    start_task_id: str, poll_interval_seconds: int, max_wait_seconds: int
) -> dict[str, Any]:
    deadline = time.time() + max_wait_seconds
    while time.time() < deadline:
        payload = request_json(
            OPENHANDS_BASE_URL,
            f"/api/v1/app-conversations/start-tasks?ids={urllib.parse.quote(start_task_id)}",
            headers={"Authorization": openhands_headers()["Authorization"]},
        )
        item = extract_first_item(payload)
        if item is None:
            time.sleep(poll_interval_seconds)
            continue
        status = item.get("status")
        if status == "READY" and item.get("app_conversation_id"):
            return item
        if status in {"ERROR", "FAILED"}:
            raise RuntimeError(f"OpenHands start task failed: {json.dumps(item)}")
        time.sleep(poll_interval_seconds)
    raise TimeoutError(
        f"Timed out waiting for start task {start_task_id} to become ready"
    )


def poll_conversation(
    app_conversation_id: str, poll_interval_seconds: int, max_wait_seconds: int
) -> dict[str, Any]:
    deadline = time.time() + max_wait_seconds
    while time.time() < deadline:
        payload = request_json(
            OPENHANDS_BASE_URL,
            f"/api/v1/app-conversations?ids={urllib.parse.quote(app_conversation_id)}",
            headers={"Authorization": openhands_headers()["Authorization"]},
        )
        item = extract_first_item(payload)
        if item is None:
            time.sleep(poll_interval_seconds)
            continue
        execution_status = str(item.get("execution_status", "")).lower()
        if execution_status in FAILED_EXECUTION_STATUSES:
            raise RuntimeError(
                "OpenHands conversation ended with "
                f"{execution_status}: {json.dumps(item)}"
            )
        if execution_status in SUCCESSFUL_TERMINAL_EXECUTION_STATUSES:
            return item
        time.sleep(poll_interval_seconds)
    raise TimeoutError(
        f"Timed out waiting for conversation {app_conversation_id} to finish running"
    )


def validate_event_search_results(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(events) >= EVENT_SEARCH_LIMIT:
        raise RuntimeError(EVENT_SEARCH_LIMIT_HIT_MESSAGE)
    return events


def fetch_app_server_events(app_conversation_id: str) -> list[dict[str, Any]]:
    payload = request_json(
        OPENHANDS_BASE_URL,
        f"/api/v1/conversation/{urllib.parse.quote(app_conversation_id)}/events/search?limit={EVENT_SEARCH_LIMIT}",
        headers={"Authorization": openhands_headers()["Authorization"]},
    )
    if isinstance(payload, dict):
        items = payload.get("items")
        return validate_event_search_results(items) if isinstance(items, list) else []
    if isinstance(payload, list):
        return validate_event_search_results(payload)
    return []


def fetch_agent_server_events(
    app_conversation_id: str, agent_server_url: str, session_api_key: str
) -> list[dict[str, Any]]:
    payload = request_json(
        agent_server_url,
        f"/api/conversations/{urllib.parse.quote(app_conversation_id)}/events/search?limit={EVENT_SEARCH_LIMIT}",
        headers={"X-Session-API-Key": session_api_key},
    )
    if isinstance(payload, dict):
        items = payload.get("items")
        return validate_event_search_results(items) if isinstance(items, list) else []
    if isinstance(payload, list):
        return validate_event_search_results(payload)
    return []


def fetch_agent_server_final_response(
    app_conversation_id: str, agent_server_url: str, session_api_key: str
) -> str:
    payload = request_json(
        agent_server_url,
        f"/api/conversations/{urllib.parse.quote(app_conversation_id)}/agent_final_response",
        headers={"X-Session-API-Key": session_api_key},
    )
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("response") or "").strip()


def extract_agent_server_url(conversation_url: str) -> str | None:
    marker = "/api/conversations/"
    if marker not in conversation_url:
        return None
    return conversation_url.rsplit(marker, 1)[0]


def extract_last_agent_text(events: list[dict[str, Any]]) -> str:
    agent_events = [
        event
        for event in events
        if event.get("kind") == "MessageEvent" and event.get("source") == "agent"
    ]
    if not agent_events:
        raise RuntimeError(
            "No assistant text message was found in the conversation events"
        )

    llm_message = agent_events[-1].get("llm_message")
    if not isinstance(llm_message, dict):
        raise RuntimeError("Last agent message has no llm_message field")
    content = llm_message.get("content")
    if not isinstance(content, list):
        raise RuntimeError("Last agent message content is not a list")

    text_parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text" and part.get("text"):
            text_parts.append(str(part["text"]))
    if not text_parts:
        raise RuntimeError("Last agent message contains no text content")
    return "".join(text_parts).strip()


def parse_agent_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for start, character in enumerate(cleaned):
            if character != "{":
                continue
            try:
                candidate, end = decoder.raw_decode(cleaned[start:])
            except json.JSONDecodeError:
                continue
            trailing = cleaned[start + end :].strip()
            if trailing not in {"", "```"}:
                continue
            if isinstance(candidate, dict):
                return candidate
        raise ValueError("No valid JSON object found in the agent response")


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(result)
    normalized["should_comment"] = as_bool(normalized.get("should_comment"))
    normalized["is_duplicate"] = as_bool(normalized.get("is_duplicate"))
    normalized["auto_close_candidate"] = as_bool(normalized.get("auto_close_candidate"))

    classification = str(normalized.get("classification") or "no-match").strip().lower()
    if classification not in {
        "duplicate",
        "overlapping-scope",
        "related-but-distinct",
        "no-match",
    }:
        classification = "no-match"
    normalized["classification"] = classification

    confidence = str(normalized.get("confidence") or "low").strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    normalized["confidence"] = confidence

    try:
        canonical_issue_number = normalized.get("canonical_issue_number")
        if canonical_issue_number in {None, ""}:
            normalized["canonical_issue_number"] = None
        else:
            normalized["canonical_issue_number"] = int(str(canonical_issue_number))
    except (TypeError, ValueError):
        normalized["canonical_issue_number"] = None

    candidate_issues = normalized.get("candidate_issues")
    if not isinstance(candidate_issues, list):
        candidate_issues = []
    normalized["candidate_issues"] = candidate_issues[:3]

    if classification not in {"duplicate", "overlapping-scope"}:
        normalized["should_comment"] = False
    if classification != "duplicate":
        normalized["is_duplicate"] = False
        normalized["auto_close_candidate"] = False
    if (
        classification in {"duplicate", "overlapping-scope"}
        and normalized["candidate_issues"]
        and confidence in {"high", "medium"}
    ):
        normalized["should_comment"] = True
    if normalized["auto_close_candidate"] and confidence != "high":
        normalized["auto_close_candidate"] = False
    if normalized["auto_close_candidate"] and not normalized["candidate_issues"]:
        normalized["auto_close_candidate"] = False
    if (
        normalized["auto_close_candidate"]
        and normalized["canonical_issue_number"] is None
    ):
        first_candidate = (
            normalized["candidate_issues"][0] if normalized["candidate_issues"] else {}
        )
        candidate_number = first_candidate.get("number")
        try:
            if candidate_number is None:
                raise ValueError("candidate number is missing")
            normalized["canonical_issue_number"] = int(str(candidate_number))
        except (TypeError, ValueError, AttributeError):
            normalized["auto_close_candidate"] = False

    normalized["summary"] = str(normalized.get("summary") or "").strip()
    return normalized


def main() -> int:
    args = parse_args()
    issue = fetch_issue(args.repository, args.issue_number)
    if issue.get("pull_request"):
        raise RuntimeError(f"#{args.issue_number} is a pull request, not an issue")

    prompt = build_prompt(args.repository, issue)
    start_task = start_conversation(prompt, args.repository, args.issue_number)
    app_conversation_id = start_task.get("app_conversation_id")
    conversation_url = ""

    if not app_conversation_id:
        task_id = start_task.get("id")
        if not task_id:
            raise RuntimeError(f"Missing id in start task response: {start_task}")
        ready_task = poll_start_task(
            task_id,
            args.poll_interval_seconds,
            args.max_wait_seconds,
        )
        app_conversation_id = ready_task.get("app_conversation_id")
        if not app_conversation_id:
            raise RuntimeError(f"Missing app_conversation_id in response: {ready_task}")

    conversation = poll_conversation(
        app_conversation_id,
        args.poll_interval_seconds,
        args.max_wait_seconds,
    )
    conversation_url = (
        conversation.get("conversation_url")
        or f"{OPENHANDS_BASE_URL}/conversations/{app_conversation_id}"
    )
    session_api_key_value = conversation.get("session_api_key")
    if session_api_key_value and not isinstance(session_api_key_value, str):
        raise RuntimeError(
            "session_api_key had unexpected type in the OpenHands conversation: "
            f"{type(session_api_key_value).__name__}"
        )
    session_api_key = session_api_key_value or ""
    agent_server_url = extract_agent_server_url(conversation_url)

    agent_text = ""
    if agent_server_url and session_api_key:
        try:
            agent_text = fetch_agent_server_final_response(
                app_conversation_id,
                agent_server_url,
                session_api_key,
            )
        except RuntimeError:
            agent_text = ""
    if not agent_text:
        events = fetch_app_server_events(app_conversation_id)
        try:
            agent_text = extract_last_agent_text(events)
        except RuntimeError as exc:
            if not session_api_key:
                raise RuntimeError(
                    "App server events did not contain assistant text and "
                    "session_api_key was missing from the OpenHands conversation"
                ) from exc
            if not agent_server_url:
                raise RuntimeError(
                    "App server events did not contain assistant text and cannot "
                    "extract agent server URL from conversation URL: "
                    f"{conversation_url}"
                ) from exc
            events = fetch_agent_server_events(
                app_conversation_id,
                agent_server_url,
                session_api_key,
            )
            agent_text = extract_last_agent_text(events)
    result = normalize_result(parse_agent_json(agent_text))

    result["issue_number"] = args.issue_number
    result["repository"] = args.repository
    result["app_conversation_id"] = app_conversation_id
    result["conversation_url"] = conversation_url
    result["agent_response"] = agent_text

    output_path = Path(args.output)
    try:
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    except OSError as exc:
        raise RuntimeError(f"Failed to write output to {output_path}: {exc}") from exc

    print(
        json.dumps(
            {
                "issue_number": result.get("issue_number"),
                "should_comment": result.get("should_comment"),
                "is_duplicate": result.get("is_duplicate"),
                "auto_close_candidate": result.get("auto_close_candidate"),
                "classification": result.get("classification"),
                "confidence": result.get("confidence"),
                "conversation_url": result.get("conversation_url"),
                "output": str(output_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        raise
