# Automation 3: Landing — Prod Tracker

**Trigger:** cron, daily (e.g. `0 14 * * 1-5`, 7am Pacific weekdays).

**Verified against the real repos** (using an authenticated `gh` session —
see PLAN.md for how). The production-detection logic below is no longer
speculative:

- `OpenHands/OpenHands`, `OpenHands/agent-canvas`, `OpenHands/software-agent-sdk`
  are public and release via git tags (e.g. `agent-canvas` uses semver tags
  like `v1.6.1`; `OpenHands` uses tags like `cloud-1.47.1`). Ancestry check
  against the latest tag is a safe default.
- `OpenHands/saas-deploy` is the **definitive prod-state source for the SaaS
  app**. It's ArgoCD-based: `gitops/production/applicationset.yaml` scans
  `app/*/environments/production/` and generates one auto-syncing
  `Application` per match — confirmed `automated`/`selfHeal`/`prune` sync
  policy, so merging to `main` reconciles to the live prod cluster with no
  manual `argocd app sync` step. For a given release (e.g. `openhands`),
  prod state lives in `app/openhands/environments/production/`:
  - `Chart.yaml`/`Chart.lock` pin the `openhands` Helm chart version (e.g.
    `0.31.0`) pulled from `oci://ghcr.io/openhands/helm-charts` — this
    chart is built and versioned by `OpenHands/OpenHands-Cloud`
    (`charts/openhands`, release-please-managed).
  - `version.yaml` pins per-image tags, e.g. `agent-canvas: tag: 1.6.1` —
    this is exactly how a `OpenHands/agent-canvas` release reaches SaaS
    prod, and the mechanism this automation should diff against for
    agent-canvas features.
  - `values.yaml` sets the actual `ENABLE_<FEATURE>` /
    `OH_WEB_CLIENT_FEATURE_FLAGS_ENABLE_<FEATURE>` env vars for prod
    (confirmed real examples: `ENABLE_BILLING`, `ENABLE_JIRA`, etc.) — so
    this file is also where "flag is on in prod" gets checked for
    Automations 5/6.
- `OpenHands/OpenHands-Cloud` is public and doubles as (a) the Helm chart
  source (`charts/openhands`, versioned via
  `.release-please-manifest.openhands.json`) and (b) the **embedded-cluster
  / Replicated installer source**
  (`replicated/openhands.yaml`, `replicated/config.yaml`,
  `replicated/embedded-cluster.yaml`). Confirmed the same
  `ENABLE_<FEATURE>` pattern is templated there too, e.g.
  `OH_WEB_CLIENT_FEATURE_FLAGS_ENABLE_JIRA_DC: 'repl{{ ConfigOptionEquals "jira_data_center_enabled" "1" }}'`
  — meaning a flag can be wired to a Replicated Config UI toggle rather than
  hardcoded, which is the actual mechanism behind the "available in
  self-hosted via Helm or embedded cluster installer" checklist item.
- `All-Hands-AI/infra` (note the different org) hosts core platform
  components under ArgoCD (`k8s/argocd/production-core`,
  `production-runtime` — cert-manager, traefik, oauth2-proxy, sysbox,
  workspace-recovery), not product feature code. See the note in
  `repos.yml` — confirm with the team whether this repo should really be
  in the same "feature landing" pipeline, or whether it needs a lighter
  variant of the checklist (it's unlikely a platform PR here needs an
  x/LinkedIn post, for instance).

```bash
curl -X POST "${OPENHANDS_HOST}/api/automation/v1/preset/prompt" \
  -H "Authorization: Bearer ${OPENHANDS_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Landing: Prod Tracker",
    "prompt": "Find every Linear issue in the '\''Feature Launches'\'' project with label stage:merged. For each one:\n\n1. Read the recorded repo + merge commit SHA from the issue description/comments (added by the '\''Landing: Merge Sync'\'' automation).\n2. Determine whether that commit has reached production, using the rule for that repo:\n   - OpenHands/OpenHands, OpenHands/software-agent-sdk: check whether the commit is an ancestor of the latest GitHub Release tag.\n   - OpenHands/enterprise (pilot): first confirm the merge commit is an ancestor of an OpenHands/enterprise semver release tag and that the corresponding ghcr.io/openhands/enterprise-server image was successfully tagged by .github/workflows/tag-image.yml. Then follow the release chain created by .github/workflows/bump-chart.yml: confirm OpenHands/OpenHands-Cloud charts/openhands/Chart.yaml .appVersion and charts/openhands/values.yaml image.tag were bumped to that Enterprise tag (or newer), and finally confirm OpenHands/saas-deploy app/openhands/environments/production/Chart.yaml + Chart.lock pin the resulting chart version. Only then treat the feature as in production; a released Enterprise image or chart PR by itself is not sufficient.\n   - OpenHands/agent-canvas: check whether the commit is an ancestor of the latest semver git tag (e.g. v1.6.1), THEN confirm that tag (or newer) is what OpenHands/saas-deploy'\''s app/openhands/environments/production/version.yaml pins for the agent-canvas image — a tagged agent-canvas release is not '\''in prod'\'' until saas-deploy'\''s production version.yaml is bumped to reference it.\n   - OpenHands/OpenHands-Cloud: check the enterprise-server / chart version via .release-please-manifest.openhands.json at the merge commit'\''s point in history, then confirm OpenHands/saas-deploy'\''s app/openhands/environments/production/Chart.yaml + Chart.lock pin that version (or newer) of oci://ghcr.io/openhands/helm-charts.\n   - OpenHands/saas-deploy: production state IS this repo'\''s app/*/environments/production/ directory (Chart.yaml, Chart.lock, version.yaml) on main — because gitops/production/applicationset.yaml auto-syncs on merge (automated + selfHeal + prune), the merge commit itself being on main'\''s history is sufficient; no further deploy step to check.\n   - All-Hands-AI/infra: this repo hosts core platform infra (cert-manager, traefik, etc.) under k8s/argocd/production-core and production-runtime, not per-feature app code. If a stage:merged ticket references this repo, check whether the merged commit'\''s changed files are referenced by any Application under k8s/argocd/production-*; if you cannot confidently determine prod status, say so explicitly in your comment rather than guessing — this repo'\''s '\''in prod'\'' semantics may differ enough from the other production repos that a human should weigh in.\n   For any repo where the expected file/structure has changed or is missing, do NOT guess — leave a comment on the Linear issue describing exactly what you found (file paths, structure) so a human can tell you the right thing to check, and skip moving the label for that issue this run.\n3. If the commit is confirmed in production:\n   a. Move the Linear issue'\''s label from stage:merged to stage:in-prod.\n   b. Add a comment noting the release version/tag/chart-version confirmed and approximate release date.\n   c. @-mention the issue assignee (the original PR author) that their feature is live in prod and that, per team process, they — as the bug-bash DRI — should schedule a bug bash with 3+ engineers within 3 business days.\n   c2. Verify checklist item 1 (docs) for real rather than trusting the PR-body checkbox: clone OpenHands/docs and search its merged PR history for one referencing this feature'\''s PR URL or Linear issue URL (check PR descriptions/titles, and git log --all --grep for the feature name as a fallback). If found, confirm a corresponding .mdx page exists (most likely under openhands/usage/, but check sdk/ or enterprise/ too depending on the repo the feature shipped in) and that its frontmatter currently has '\''hidden: true'\'' (expected at this stage — not GA yet). If both are true, tick checklist item 1 in the Linear issue description and in the PR'\''s tracker comment. If no matching docs PR is found at all, or a page exists without hidden: true this early, do NOT tick the item — leave it unchecked and add a comment on the Linear issue stating exactly what you found (or didn'\''t find), same as how ambiguous prod-detection cases are handled for other repos.\n   c3. Verify checklist item 2 from the original feature PR diff: identify at least one real E2E test or E2E scenario change that exercises this feature'\''s behavior. A checked PR-body box, unit test, snapshot, or unrelated E2E file is not sufficient. Record the test file and scenario as evidence; if coverage is ambiguous, leave item 2 unchecked and explain why.\n   c4. Verify checklist item 3 using the exact '\''Feature flag:'\'' value on the Linear ticket. Confirm the flag is wired through the implementation and through the appropriate self-hosted path, which may be a linked downstream PR rather than a file in the feature repo. For the OpenHands/enterprise pilot, verify the frontend/server flag wiring and matching Helm values/templates in OpenHands/OpenHands-Cloud; when embedded-cluster support applies, also verify the Replicated config under replicated/. Record file paths and PRs as evidence. If the flag is still TBD or either implementation or self-hosted wiring is missing, leave item 3 unchecked and explain the gap.\n   d. Update the original PR'\''s Feature Landing Tracker comment (marker '\''<!-- landing-tracker:v1 -->'\'', find via GITHUB_TOKEN and PATCH the existing comment): bar becomes '\''✅ Review  →  ✅ Merged  →  ✅ In Prod  →  🔄 Bug Bash  →  ⬜ Council Review  →  ⬜ Council Approved  →  ⬜ Flag On  →  ⬜ GA'\'', tick the release checklist item with the confirmed version/tag, and current-stage line names the DRI and the 3-business-day bug-bash deadline. Get the PR number/repo from the Linear issue description recorded by Automation 1. Fetch landing-checklist/tracker-format.md from OpenHands/OpenHands (raw, main) for exact formatting.\n   e. Post the FIRST Slack message for this feature to #tech-council, using the SLACK_BOT_TOKEN secret and chat.postMessage, following the '\''Slack Block Kit rendering'\'' (top-level, not threaded) section of tracker-format.md exactly — header with the feature title, the 8-stage bar as a context block, current-stage/DRI section, checklist section, and a footer context block linking the PR and Linear issue. Record the response'\''s channel + ts (thread starter timestamp) as a comment on the Linear issue (e.g. '\''slack-thread: <channel>/<ts>'\'') so Automations 4, 5, 6, and 7 can find it and reply in the same thread instead of posting new top-level messages.\n   f. End every human-facing GitHub, Linear, or Slack body you create with: '\''_Automated by an OpenHands AI agent on behalf of the engineering team._'\''\n4. If not yet in production, do nothing further for that issue this run (it will be re-checked tomorrow).\n\nSummarize how many issues were checked, how many moved to stage:in-prod, and how many could not be confirmed (with why).",
    "trigger": {"type": "cron", "schedule": "0 14 * * 1-5", "timezone": "America/Los_Angeles"},
    "timeout": 900,
    "repos": [
      {"url": "https://github.com/OpenHands/OpenHands", "ref": "main"},
      {"url": "https://github.com/OpenHands/enterprise", "ref": "main"},
      {"url": "https://github.com/OpenHands/agent-canvas", "ref": "main"},
      {"url": "https://github.com/OpenHands/software-agent-sdk", "ref": "main"},
      {"url": "https://github.com/OpenHands/saas-deploy", "ref": "main"},
      {"url": "https://github.com/OpenHands/OpenHands-Cloud", "ref": "main"},
      {"url": "https://github.com/All-Hands-AI/infra", "ref": "main"},
      {"url": "https://github.com/OpenHands/docs", "ref": "main"}
    ]
  }'
```

Because this clones `OpenHands/saas-deploy` and `All-Hands-AI/infra` (both
private), the automation's GitHub integration/token needs read access
granted to both before first deploy — confirm that's set up.

**Slack:** channel confirmed as `#tech-council`. This is the automation that
posts the *first* message for each feature (see `tracker-format.md` for why
Slack visibility starts here rather than at PR-open). Requires a
`SLACK_BOT_TOKEN` secret with `chat:write` scope, and the bot user invited
to `#tech-council` — set both up before deploying.

**Docs verification:** this is also the automation that closes the
honor-system gap on checklist item 1 — it confirms a real merged PR exists
in `OpenHands/docs` (not just a checked box) and that the page is currently
marked `hidden: true` (not yet public). See `docs-visibility.md` for the
full mechanism, including who later clears the `hidden` flag (Automation 7,
at `stage:flag-on`). `OpenHands/docs` is an additional integration repo
outside the seven production repos and needs read access.
