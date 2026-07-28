# Automation 6: Landing - Council Reaction Gate

**Trigger:** cron, every 5 minutes.

**Implementation:** deterministic custom Python script with no LLM. Reaction
counting, channel membership, and label mutation are fixed rules; invoking an
agent up to 288 times/day would add cost and nondeterminism without benefit.
The source is in `council-reaction-gate/main.py`.

## Approval policy

- Approval message must be in `#tech-council`.
- Approval emoji: `✅` (Slack API name `white_check_mark`).
- Blocking emoji: `🚫` (Slack API name `no_entry_sign`).
- Quorum: **two distinct human approvals**.
- Approver allowlist: current human members of `#tech-council`, fetched from
  Slack on every run. Bot, app-user, deleted-user, and Slackbot reactions do
  not count.
- The original `(feat)` PR author is excluded from both approval and blocking
  vote sets.
- Blocking wins: any valid `🚫` keeps the ticket in
  `stage:council-review`, even if two or more valid approvals exist. Removing
  the blocking reaction allows a future run to approve.
- On approval, move the Linear ticket to `stage:council-approved` - **not**
  `stage:flag-on`. Council intent is separate from proof that the flag was
  enabled in production.

## Contract with Automation 5

Automation 5 posts the explicit approval request, resolves the original PR
assignee to Slack via email, and records both values in one Linear comment:

```text
council-approval-message: <channel-id>/<message-ts>
council-pr-author-slack-user: <slack-user-id>
```

Automation 6 requires both markers and excludes the recorded PR author from
approval and blocking votes. It does not count reactions on the thread
starter or other replies, avoiding accidental approval from unrelated emoji.

## Side effects when quorum is satisfied

1. Replace `stage:council-review` with `stage:council-approved` on the Linear
   ticket and record the approving Slack user IDs in a Linear comment.
2. Reply in the same Slack thread that council approval is recorded and that
   production flag verification is still pending.
3. Advance the original PR tracker to `Council Approved`, with `Flag On` still
   pending.

The transition is idempotent: after the label changes, the issue no longer
appears in the next `stage:council-review` query.

## Required secrets and Slack scopes

- `SLACK_BOT_TOKEN`
  - `chat:write`
  - `reactions:read`
  - `channels:read`
  - `users:read`
  - `groups:read` if `#tech-council` is private
- `LINEAR_API_KEY`
- `GITHUB_TOKEN` with PR-comment write access to all production repos,
  including the private `OpenHands/enterprise` pilot.

The bot must be invited to `#tech-council`.

Linear must have `stage:council-approved` as one workspace label or once for
each participating team. The script prefers the issue team's label and falls
back to a unique workspace label.

## Validate and package

```bash
cd automations/council-reaction-gate
python3 -m py_compile main.py
python3 -m unittest -v test_main.py
tar -czf ../council-reaction-gate.tar.gz main.py
```

The tarball is a local deployment artifact and should not be committed.

## Upload and create (run only when ready to deploy)

```bash
OPENHANDS_HOST="https://app.all-hands.dev"

UPLOAD_RESPONSE="$(curl -sS -X POST \
  "${OPENHANDS_HOST}/api/automation/v1/uploads?name=landing-council-reaction-gate&description=Two-vote%20Slack%20council%20approval" \
  -H "Authorization: Bearer ${OPENHANDS_API_KEY}" \
  -H "Content-Type: application/gzip" \
  --data-binary @automations/council-reaction-gate.tar.gz)"

TARBALL_PATH="$(printf '%s' "$UPLOAD_RESPONSE" | jq -r '.tarball_path')"

test -n "$TARBALL_PATH" && test "$TARBALL_PATH" != "null"

curl -sS -X POST "${OPENHANDS_HOST}/api/automation/v1" \
  -H "Authorization: Bearer ${OPENHANDS_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg tarball_path "$TARBALL_PATH" '{
    name: "Landing: Council Reaction Gate",
    trigger: {
      type: "cron",
      schedule: "*/5 * * * *",
      timezone: "America/Los_Angeles"
    },
    tarball_path: $tarball_path,
    entrypoint: "python3 main.py",
    timeout: 120
  }')"
```

Do not deploy until the `stage:council-approved` Linear label exists and the
Slack token has all scopes above.

## Deliberately separate next transition

A production reconciler still needs to implement:

```text
stage:council-approved -> stage:flag-on
```

For the `OpenHands/enterprise` pilot, it should verify the Enterprise flag in
the released `ghcr.io/openhands/enterprise-server` image and the production
configuration in `OpenHands/saas-deploy` before advancing. Only
`stage:flag-on` triggers the docs reveal and the three-month cleanup clock.
