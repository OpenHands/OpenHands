import json
import os
import re
import urllib.parse
import urllib.request

APPROVAL_EMOJI = "white_check_mark"
BLOCK_EMOJI = "no_entry_sign"
APPROVAL_QUORUM = 2
CHANNEL_NAME = "tech-council"
AI_NOTICE = "Automated by an OpenHands AI agent on behalf of the engineering team."


def get_secret(name):
    """Fetch a named secret stored in the agent server."""
    url = os.environ.get("AGENT_SERVER_URL", "").rstrip("/")
    key = os.environ.get("SESSION_API_KEY") or os.environ.get("OH_SESSION_API_KEYS_0", "")
    with urllib.request.urlopen(
        urllib.request.Request(
            f"{url}/api/settings/secrets/{name}",
            headers={"X-Session-API-Key": key},
        )
    ) as response:
        return response.read().decode().strip()


def fire_callback(status="COMPLETED", error=None):
    """Signal run completion. MUST be called on every exit path."""
    url = os.environ.get("AUTOMATION_CALLBACK_URL", "")
    if not url:
        return
    body = {"status": status, "run_id": os.environ.get("AUTOMATION_RUN_ID", "")}
    if error:
        body["error"] = error
    try:
        urllib.request.urlopen(
            urllib.request.Request(
                url,
                data=json.dumps(body).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {os.environ.get('AUTOMATION_CALLBACK_API_KEY', '')}",
                },
            )
        )
    except Exception as exc:
        print(f"Callback error: {exc}")


def request_json(url, *, headers=None, data=None, method=None):
    encoded = json.dumps(data).encode() if data is not None else None
    request = urllib.request.Request(
        url,
        data=encoded,
        headers=headers or {},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode())


def slack_call(token, method, params=None, *, post=False):
    url = f"https://slack.com/api/{method}"
    headers = {"Authorization": f"Bearer {token}"}
    data = None
    if post:
        headers["Content-Type"] = "application/json; charset=utf-8"
        data = params or {}
    elif params:
        url += "?" + urllib.parse.urlencode(params)
    result = request_json(url, headers=headers, data=data)
    if not result.get("ok"):
        raise RuntimeError(f"Slack {method} failed: {result.get('error', 'unknown_error')}")
    return result


def linear_call(token, query, variables=None):
    result = request_json(
        "https://api.linear.app/graphql",
        headers={
            "Authorization": token,
            "Content-Type": "application/json",
        },
        data={"query": query, "variables": variables or {}},
    )
    if result.get("errors"):
        raise RuntimeError(f"Linear GraphQL failed: {result['errors']}")
    return result["data"]


def find_channel(slack_token):
    cursor = None
    while True:
        params = {
            "types": "public_channel,private_channel",
            "exclude_archived": "true",
            "limit": 200,
        }
        if cursor:
            params["cursor"] = cursor
        result = slack_call(slack_token, "conversations.list", params)
        for channel in result.get("channels", []):
            if channel.get("name") == CHANNEL_NAME:
                return channel["id"]
        cursor = result.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            raise RuntimeError(f"Slack channel #{CHANNEL_NAME} was not found")


def channel_members(slack_token, channel_id):
    members = set()
    cursor = None
    while True:
        params = {"channel": channel_id, "limit": 200}
        if cursor:
            params["cursor"] = cursor
        result = slack_call(slack_token, "conversations.members", params)
        members.update(result.get("members", []))
        cursor = result.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            return members


def is_human(slack_token, user_id, cache):
    if user_id not in cache:
        cache[user_id] = slack_call(slack_token, "users.info", {"user": user_id})["user"]
    user = cache[user_id]
    return (
        not user.get("deleted")
        and not user.get("is_bot")
        and not user.get("is_app_user")
        and user.get("id") != "USLACKBOT"
    )


def eligible_reactors(slack_token, reactors, members, excluded_user, cache):
    return {
        user_id
        for user_id in reactors & members
        if user_id != excluded_user and is_human(slack_token, user_id, cache)
    }


def reaction_users(message, emoji_name):
    for reaction in message.get("reactions", []):
        if reaction.get("name") == emoji_name:
            return set(reaction.get("users", []))
    return set()


def council_decision(approvals, blockers):
    if blockers:
        return "blocked"
    if len(approvals) >= APPROVAL_QUORUM:
        return "approved"
    return "waiting"


def reply_thread_timestamp(message, message_timestamp):
    return message.get("thread_ts") or message_timestamp


def parse_approval_message(comments):
    message_pattern = re.compile(
        r"council-approval-message:\s*([A-Z0-9]+)/([0-9.]+)"
    )
    author_pattern = re.compile(r"council-pr-author-slack-user:\s*([A-Z0-9]+)")
    ordered = sorted(
        enumerate(comments),
        key=lambda item: (item[1].get("createdAt") or "", item[0]),
    )
    for _, comment in reversed(ordered):
        body = comment.get("body") or ""
        message_match = message_pattern.search(body)
        author_match = author_pattern.search(body)
        if message_match and author_match:
            return message_match.group(1), message_match.group(2), author_match.group(1)
    return None


def get_review_issues(linear_token):
    query = """
      query CouncilReviewIssues {
        projects(filter: {name: {eq: "Feature Launches"}}, first: 1) {
          nodes {
            issues(
              first: 100
              filter: {labels: {name: {eq: "stage:council-review"}}}
            ) {
              nodes {
                id
                identifier
                title
                description
                team { id name }
                labels { nodes { id name } }
                comments(first: 100) { nodes { body createdAt } }
              }
            }
          }
        }
      }
    """
    projects = linear_call(linear_token, query)["projects"]["nodes"]
    return projects[0]["issues"]["nodes"] if projects else []


def get_approved_labels(linear_token):
    query = """
      query ApprovedLabels {
        issueLabels(filter: {name: {eq: "stage:council-approved"}}, first: 100) {
          nodes { id name team { id name } }
        }
      }
    """
    return linear_call(linear_token, query)["issueLabels"]["nodes"]


def approved_label_for_issue(labels, issue):
    team_id = issue["team"]["id"]
    exact = [label for label in labels if (label.get("team") or {}).get("id") == team_id]
    workspace = [label for label in labels if not label.get("team")]
    candidates = exact or workspace
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one stage:council-approved label for Linear team {issue['team']['name']}"
        )
    return candidates[0]["id"]


def approve_issue(linear_token, issue, approved_label_id, approvers):
    label_ids = [
        label["id"]
        for label in issue["labels"]["nodes"]
        if label["name"] != "stage:council-review"
    ]
    label_ids.append(approved_label_id)
    mutation = """
      mutation ApproveIssue($id: String!, $labelIds: [String!]!, $body: String!) {
        commentCreate(input: {issueId: $id, body: $body}) { success }
        issueUpdate(id: $id, input: {labelIds: $labelIds}) { success }
      }
    """
    body = (
        "Tech council approved this feature via Slack reactions: "
        + ", ".join(f"<@{user_id}>" for user_id in sorted(approvers))
        + ". Waiting for independent verification that the feature flag is enabled in production."
        + f"\n\n_{AI_NOTICE}_"
    )
    linear_call(
        linear_token,
        mutation,
        {"id": issue["id"], "labelIds": label_ids, "body": body},
    )


def update_pr_tracker(github_token, issue):
    match = re.search(
        r"https://github\.com/([^/]+)/([^/]+)/pull/(\d+)",
        issue.get("description") or "",
    )
    if not match:
        raise RuntimeError(f"{issue['identifier']}: no feature PR URL for tracker update")
    owner, repo, number = match.groups()
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "openhands-landing-checklist",
    }
    comments = request_json(
        f"https://api.github.com/repos/{owner}/{repo}/issues/{number}/comments?per_page=100",
        headers=headers,
    )
    tracker = next(
        (comment for comment in comments if "<!-- landing-tracker:v1 -->" in comment.get("body", "")),
        None,
    )
    if not tracker:
        raise RuntimeError(f"{issue['identifier']}: tracker comment not found")
    lines = tracker["body"].splitlines()
    bar = (
        "✅ Review  →  ✅ Merged  →  ✅ In Prod  →  ✅ Bug Bash  →  "
        "✅ Council Review  →  🔄 Council Approved  →  ⬜ Flag On  →  ⬜ GA"
    )
    replaced_bar = False
    for index, line in enumerate(lines):
        if "Review" in line and "Merged" in line and "Flag On" in line and "GA" in line:
            lines[index] = bar
            replaced_bar = True
        elif line.startswith("**Current stage:**"):
            lines[index] = (
                "**Current stage:** Council Approved — awaiting independent verification "
                "that the feature flag is enabled in production."
            )
    if not replaced_bar:
        raise RuntimeError(f"{issue['identifier']}: tracker bar not recognized")
    if not any(AI_NOTICE in line for line in lines):
        lines.extend(["", f"_{AI_NOTICE}_"])
    request_json(
        tracker["url"],
        headers={**headers, "Content-Type": "application/json"},
        data={"body": "\n".join(lines)},
        method="PATCH",
    )


def process_issue(slack_token, linear_token, github_token, channel_id, members, human_cache, issue, approved_label_id):
    approval_message = parse_approval_message(issue["comments"]["nodes"])
    if not approval_message:
        print(f"{issue['identifier']}: no council-approval-message marker")
        return "missing-message"
    message_channel, timestamp, pr_author_slack_user = approval_message
    if message_channel != channel_id:
        print(f"{issue['identifier']}: approval message is not in #{CHANNEL_NAME}")
        return "wrong-channel"

    result = slack_call(
        slack_token,
        "reactions.get",
        {"channel": channel_id, "timestamp": timestamp, "full": "true"},
    )
    message = result.get("message", {})
    approvals = eligible_reactors(
        slack_token,
        reaction_users(message, APPROVAL_EMOJI),
        members,
        pr_author_slack_user,
        human_cache,
    )
    blockers = eligible_reactors(
        slack_token,
        reaction_users(message, BLOCK_EMOJI),
        members,
        pr_author_slack_user,
        human_cache,
    )
    decision = council_decision(approvals, blockers)
    if decision == "blocked":
        print(f"{issue['identifier']}: blocked by {sorted(blockers)}")
        return decision
    if decision == "waiting":
        print(f"{issue['identifier']}: {len(approvals)}/{APPROVAL_QUORUM} approvals")
        return decision

    update_pr_tracker(github_token, issue)
    slack_call(
        slack_token,
        "chat.postMessage",
        {
            "channel": channel_id,
            "thread_ts": reply_thread_timestamp(message, timestamp),
            "text": (
                f"✅ Council approval recorded ({len(approvals)}/{APPROVAL_QUORUM}). "
                "Waiting for the feature flag to be verified on in production."
                f"\n\n_{AI_NOTICE}_"
            ),
        },
        post=True,
    )
    # Advance the stage last so a transient GitHub or Slack failure is retried.
    approve_issue(linear_token, issue, approved_label_id, approvals)
    print(f"{issue['identifier']}: approved by {sorted(approvals)}")
    return "approved"


def main():
    slack_token = get_secret("SLACK_BOT_TOKEN")
    linear_token = get_secret("LINEAR_API_KEY")
    github_token = get_secret("GITHUB_TOKEN")
    channel_id = find_channel(slack_token)
    members = channel_members(slack_token, channel_id)
    issues = get_review_issues(linear_token)
    approved_labels = get_approved_labels(linear_token)
    human_cache = {}
    counts = {}
    errors = []
    for issue in issues:
        try:
            approved_label_id = approved_label_for_issue(approved_labels, issue)
            outcome = process_issue(
                slack_token,
                linear_token,
                github_token,
                channel_id,
                members,
                human_cache,
                issue,
                approved_label_id,
            )
        except Exception as exc:
            outcome = "error"
            errors.append(f"{issue['identifier']}: {exc}")
            print(errors[-1])
        counts[outcome] = counts.get(outcome, 0) + 1
    print(json.dumps({"checked": len(issues), "outcomes": counts}, sort_keys=True))
    if errors:
        raise RuntimeError("; ".join(errors))


if __name__ == "__main__":
    try:
        main()
        fire_callback("COMPLETED")
    except Exception as exc:
        print(f"Council reaction gate failed: {exc}")
        fire_callback("FAILED", str(exc))
        raise
