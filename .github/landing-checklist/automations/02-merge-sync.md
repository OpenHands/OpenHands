# Automation 2: Landing — Merge Sync

**Trigger:** GitHub event, `pull_request.closed` where `pull_request.merged
== true`.
**Filter:** same repo allowlist + `(feat)` title as Automation 1.

```bash
curl -X POST "${OPENHANDS_HOST}/api/automation/v1/preset/prompt" \
  -H "Authorization: Bearer ${OPENHANDS_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Landing: Merge Sync",
    "prompt": "A GitHub pull request was just closed and merged. Payload includes repository.full_name, pull_request.number, pull_request.html_url, pull_request.merge_commit_sha, pull_request.title.\n\nFetch the current production repo allowlist from https://raw.githubusercontent.com/OpenHands/OpenHands/main/.github/landing-checklist/repos.yml. If repository.full_name is not in production_repos, stop and do nothing.\n\nFind the Linear issue in the '\''Feature Launches'\'' project whose description references pull_request.html_url. If none is found, do nothing and report that (this can happen if the PR title got the (feat) tag added only after opening, in which case Automation 1 may not have run — flag this for manual follow-up rather than silently creating a duplicate issue).\n\nIf found:\n1. Move its label from stage:review to stage:merged.\n2. Append a comment recording the merge commit SHA and merge timestamp.\n3. Leave the 6-item checklist as-is (Automation 3 will confirm the release item later).\n4. Update the PR'\''s existing Feature Landing Tracker comment (marker '\''<!-- landing-tracker:v1 -->'\'', created by Automation 1 — find it via GET /repos/{owner}/{repo}/issues/{number}/comments and PATCH it, do not create a new one) so the 8-stage bar reads '\''✅ Review  →  🔄 Merged  →  ⬜ In Prod  →  ⬜ Bug Bash  →  ⬜ Council Review  →  ⬜ Council Approved  →  ⬜ Flag On  →  ⬜ GA'\'' and the current-stage line notes the merge commit SHA. Follow the exact rendering in landing-checklist/tracker-format.md (fetch from OpenHands/OpenHands raw, main branch) so formatting stays consistent across every automation.\n5. End every human-facing GitHub or Linear body you create with: '\''_Automated by an OpenHands AI agent on behalf of the engineering team._'\''\n\nReport the Linear issue URL and the new stage in your summary.",
    "trigger": {
      "type": "event",
      "source": "github",
      "on": "pull_request.closed",
      "filter": "pull_request.merged == `true` && starts_with(pull_request.title, `(feat)`) && contains([`OpenHands/OpenHands`,`OpenHands/enterprise`,`OpenHands/agent-canvas`,`OpenHands/software-agent-sdk`,`All-Hands-AI/infra`,`OpenHands/saas-deploy`,`OpenHands/OpenHands-Cloud`], repository.full_name)"
    },
    "timeout": 300
  }'
```

Notes:
- The event filter repeats the seven-repo allowlist because webhook filters
  cannot fetch `repos.yml`. When adding a production repo, update this filter
  and Automation 1's filter alongside `repos.yml`.
- Requires `GITHUB_TOKEN` and `LINEAR_API_KEY` on the automation agent server.
