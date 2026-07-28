# Automation 1: Landing — Create Feature Ticket

**Trigger:** GitHub event, `pull_request.opened` (and `edited`, in case the
title is fixed up to add `(feat)` after opening).
**Filter:** title starts with `(feat)` AND repo is in the production
allowlist (`landing-checklist/repos.yml`).

```bash
curl -X POST "${OPENHANDS_HOST}/api/automation/v1/preset/prompt" \
  -H "Authorization: Bearer ${OPENHANDS_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Landing: Create Feature Ticket",
    "prompt": "A GitHub pull request was just opened. The event payload contains repository.full_name, pull_request.number, pull_request.title, pull_request.html_url, pull_request.user.login, and pull_request.body.\n\n1. Fetch the current production repo allowlist from https://raw.githubusercontent.com/OpenHands/OpenHands/main/.github/landing-checklist/repos.yml (GITHUB_TOKEN secret). If repository.full_name is not in production_repos, stop — do nothing.\n2. Confirm pull_request.title matches the regex ^\\(feat\\). If not, stop.\n3. Check whether a Linear issue already references this PR URL (search Linear issues in the '\''Feature Launches'\'' project, across all teams, for pull_request.html_url in the description). If one exists, skip issue creation and just make sure it has label stage:review, then stop.\n4. Otherwise create a new Linear issue in the '\''Feature Launches'\'' project titled after the PR title (strip the leading '\''(feat) '\'' prefix), with label stage:review. Assignee and team: try to match pull_request.user.login to a Linear user by GitHub username/name/email; file the issue under that person'\''s own Linear team (this is also who will be pinged as the bug-bash DRI later, per team process — no rotation, it'\''s always the PR author). If no match is found, assign to the '\''ALL'\'' (All Hands AI) team and leave unassigned with a note asking someone to claim it. Description should contain: the PR link, the repo, and the full 6-item landing checklist copied verbatim from .github/landing-checklist/PULL_REQUEST_TEMPLATE_snippet.md in OpenHands/OpenHands (fetch it) with all 6 boxes unchecked, plus a line '\''Feature flag: TBD'\'' to be filled in once the ENABLE_<FEATURE> flag name is known, plus a line '\''Docs page: TBD (must be added to OpenHands/docs with hidden: true in frontmatter — see landing-checklist/docs-visibility.md)'\'' to be filled in once the docs PR exists, plus a line '\''Social post: TBD'\'' that must eventually contain a public X or LinkedIn URL.\n5. Post TWO things on the GitHub PR (use GITHUB_TOKEN and POST /repos/{owner}/{repo}/issues/{number}/comments), skipping either if it already exists:\n   a. A short one-line comment linking back to the created (or existing) Linear issue, e.g. '\''Tracking this feature'\''s landing checklist in <Linear URL>'\''.\n   b. The Feature Landing Tracker comment: fetch landing-checklist/tracker-format.md from OpenHands/OpenHands (raw, main branch) and follow its '\''GitHub markdown rendering'\'' section exactly, including the literal marker line '\''<!-- landing-tracker:v1 -->'\'' as the first line of the comment body so later automations (2 through 7) can find and update this same comment in place. At this stage the 8-stage bar is '\''🔄 Review  →  ⬜ Merged  →  ⬜ In Prod  →  ⬜ Bug Bash  →  ⬜ Council Review  →  ⬜ Council Approved  →  ⬜ Flag On  →  ⬜ GA'\'', the checklist reflects whichever of the 6 items are already checked in the PR body (usually just items 1-3, since 4-6 are inherently post-merge), and the footer links to the Linear issue (no Slack thread yet — that gets added by Automation 3).\n6. End every human-facing GitHub or Linear body you create with: '\''_Automated by an OpenHands AI agent on behalf of the engineering team._'\''\n7. Report the Linear issue URL and PR URL in your final summary.",
    "trigger": {
      "type": "event",
      "source": "github",
      "on": ["pull_request.opened", "pull_request.edited"],
      "filter": "starts_with(pull_request.title, `(feat)`) && contains([`OpenHands/OpenHands`,`OpenHands/enterprise`,`OpenHands/agent-canvas`,`OpenHands/software-agent-sdk`,`All-Hands-AI/infra`,`OpenHands/saas-deploy`,`OpenHands/OpenHands-Cloud`], repository.full_name)"
    },
    "timeout": 300
  }'
```

Notes:
- The JMESPath `filter` hard-codes the same 7 repos as `repos.yml` as a fast
  first-pass filter (webhooks only deliver for repos where the automation's
  GitHub App/integration is installed anyway); the prompt re-checks against
  the live `repos.yml` so adding another repo only requires (a) installing the
  GitHub integration on it and (b) editing `repos.yml` — this filter
  expression should be updated at the same time for efficiency, but is not
  load-bearing for correctness since the prompt re-validates.
- Requires: `GITHUB_TOKEN` (repo read + PR comment write) and `LINEAR_API_KEY`
  registered as secrets on the automation's agent server.
- Repo names/orgs (`All-Hands-AI/infra`, `OpenHands/saas-deploy`, etc.) are
  now verified — see `repos.yml` and `PLAN.md`.
- `tracker-format.md` must be committed to `OpenHands/OpenHands` at
  `.github/landing-checklist/tracker-format.md` (alongside `repos.yml`)
  before this automation (and 2-6) can fetch it.
