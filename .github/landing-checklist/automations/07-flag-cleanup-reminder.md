# Automation 7: Landing — Flag Cleanup Reminder

**Trigger:** cron, weekly (this is a long-horizon check — months, not days).
**Slack/tracker:** this is also the automation that posts the closing 🎉
message once a feature reaches `stage:ga` — the last entry in the
`#tech-council` thread and the last update to the PR tracker comment. See
`tracker-format.md`.
**Docs reveal:** this is also where the docs page's `hidden: true` flag
gets cleared — as soon as this automation next runs after a ticket reaches
`stage:flag-on` (not waiting for the 3-month flag-cleanup timer, which is a
separate concern). See `docs-visibility.md`. `OpenHands/docs` is an additional integration repo
outside the seven-repo production allowlist and needs write access (this
automation opens a real PR against it, unlike the read-only grep-and-report
behavior for the production repos).

**Known timing gap:** because this automation is weekly cron, the docs
reveal can lag up to ~7 days behind the verified production flag flip (e.g.
the production reconciler advances the ticket Monday, this doesn't run until
next Monday). If same-day reveal matters, run STEP 0 daily or co-locate it
with the production reconciler that advances `stage:council-approved` to
`stage:flag-on`. That reconciler is not implemented yet.

```bash
curl -X POST "${OPENHANDS_HOST}/api/automation/v1/preset/prompt" \
  -H "Authorization: Bearer ${OPENHANDS_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Landing: Flag Cleanup Reminder",
    "prompt": "Find every Linear issue in the '\''Feature Launches'\'' project with label stage:flag-on.\n\nSTEP 0 — docs reveal (run this for every stage:flag-on issue found, regardless of how long it'\''s been in this stage): check whether a '\''docs revealed'\'' comment already exists on this Linear issue (added by a previous run of this automation). If not:\n   a. Read the docs PR reference recorded on the Linear issue (from Automation 3'\''s verification comment, or the '\''Docs page: ...'\'' line from Automation 1) to find the .mdx file path in OpenHands/docs.\n   b. Clone/check OpenHands/docs, confirm the page still has '\''hidden: true'\'' in its frontmatter, then open a PR against OpenHands/docs removing that line entirely (per docs-visibility.md, do NOT set hidden: false — remove the field). Title the PR something like '\''Reveal docs: <feature name>'\'' and reference the Linear issue.\n   c. Enable auto-merge on that PR after the normal OpenHands/docs required checks pass. Do not bypass branch protection or merge a failing PR. Council approval and verified production enablement make the content decision pre-approved; repository checks still gate the merge.\n   d. Post a '\''docs revealed'\'' comment on the Linear issue linking the merged docs PR, and update the PR'\''s Feature Landing Tracker comment (marker '\''<!-- landing-tracker:v1 -->'\'', GITHUB_TOKEN, PATCH) so the '\''Flag On'\'' segment is ticked and the docs-page checklist detail is marked done. If no docs PR reference can be found at all, skip this step and note it in your summary rather than guessing a file path.\n\nSTEP 1 — social launch evidence: read the '\''Social post:'\'' field on the ticket. If it contains a public x.com, twitter.com, or linkedin.com post URL, tick checklist item 6 in Linear and the PR tracker and preserve the URL as evidence. If it is TBD, missing, or not a supported public URL, leave item 6 unchecked and remind the assignee to coordinate the launch post and record its URL. Do not move the ticket to stage:ga while social evidence is missing.\n\nSTEP 2 — flag cleanup countdown: for each stage:flag-on issue, check the '\''flag verified on in production'\'' date recorded by the production reconciler, and the ENABLE_<FEATURE> flag name recorded on the ticket.\n\nIf 3+ months have elapsed since that date:\n1. Post a comment @-mentioning the assignee (the original PR author — or their manager if they'\''ve left the team, best-effort lookup), asking them to remove the flag from enterprise/server/auth/constants.py, enterprise/server/config.py'\''s FEATURE_FLAGS dict, any Helm values.yaml toggle, and any frontend WebClientFeatureFlags usage, then ship the feature on-by-default.\n2. Grep the relevant repo for the flag name (e.g. `grep -rn ENABLE_<FEATURE>`) and list every file location in your comment, to make the cleanup PR easier to scope (do NOT open a PR that actually removes the flag automatically — flag removal needs human review; this is just a scouting comment listing affected files).\n3. Update the PR'\''s Feature Landing Tracker comment detail line to note the cleanup reminder was sent and the 3-month deadline, per landing-checklist/tracker-format.md (fetch from OpenHands/OpenHands raw, main) for exact formatting. Also read the '\''slack-thread: <channel>/<ts>'\'' comment on this Linear issue and post a lightweight threaded reply to #tech-council (chat.postMessage, SLACK_BOT_TOKEN secret) noting the cleanup nudge went out.\n4. When you have confirmed (in a later run) that the flag no longer appears anywhere in the codebase'\''s '\''main'\'' branch AND STEP 1 has valid social-post evidence:\n   a. Move the label to stage:ga and post a closing summary comment on the Linear issue.\n   b. Update the Feature Landing Tracker comment on the original PR one last time: 8-stage bar becomes fully '\''✅ Review  →  ✅ Merged  →  ✅ In Prod  →  ✅ Bug Bash  →  ✅ Council Review  →  ✅ Council Approved  →  ✅ Flag On  →  ✅ GA'\'', all 6 checklist items ticked, and a closing note that the feature is fully GA'\''d and the flag is removed.\n   c. Post a final threaded reply to #tech-council in the same slack-thread with a closing 🎉 message noting the feature is GA and the flag has been fully removed from the codebase.\n\nEnd every human-facing GitHub, Linear, or Slack body you create with: '\''_Automated by an OpenHands AI agent on behalf of the engineering team._'\''\n\nSummarize how many docs pages were revealed, how many issues were reminded about flag cleanup, and any moved to stage:ga.",
    "trigger": {"type": "cron", "schedule": "0 15 * * 1", "timezone": "America/Los_Angeles"},
    "timeout": 900,
    "repos": [
      {"url": "https://github.com/OpenHands/OpenHands", "ref": "main"},
      {"url": "https://github.com/OpenHands/enterprise", "ref": "main"},
      {"url": "https://github.com/OpenHands/agent-canvas", "ref": "main"},
      {"url": "https://github.com/OpenHands/software-agent-sdk", "ref": "main"},
      {"url": "https://github.com/All-Hands-AI/infra", "ref": "main"},
      {"url": "https://github.com/OpenHands/saas-deploy", "ref": "main"},
      {"url": "https://github.com/OpenHands/OpenHands-Cloud", "ref": "main"},
      {"url": "https://github.com/OpenHands/docs", "ref": "main"}
    ]
  }'
```
