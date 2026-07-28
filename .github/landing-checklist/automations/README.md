# OpenHands Automations - Feature Landing Checklist

These automations implement ticket creation and the post-merge feature landing
lifecycle. Nothing is deployed by this workspace; these files are deployment
specifications and source artifacts for review.

Seven automations map to the Linear state machine in
`../linear/state-machine.md`:

| # | Name | Trigger | Implementation |
|---|---|---|---|
| 1 | Landing: Create Feature Ticket | GitHub PR opened/edited | Prompt preset; deterministic candidate |
| 2 | Landing: Merge Sync | GitHub PR merged | Prompt preset; deterministic candidate |
| 3 | Landing: Prod Tracker | Daily cron | Prompt preset; heterogeneous cross-repo release reasoning |
| 4 | Landing: Bug Bash Reminder | Daily cron | Prompt preset |
| 5 | Landing: Bug Bash Gate | Daily cron | Prompt preset; posts council reaction request |
| 6 | Landing: Council Reaction Gate | Every 5 minutes | Custom no-LLM Python script |
| 7 | Landing: Flag Cleanup Reminder | Weekly cron | Prompt preset |

Automation 6 is deliberately not agent-driven. It evaluates a fixed Slack
policy up to 288 times/day: two distinct human members of `#tech-council`
reacting `white_check_mark` approves, while any valid `no_entry_sign` blocks
until removed. The implementation and tests live in
`council-reaction-gate/`; deployment details are in
`06-council-reaction-gate.md`.

## Production repositories

`../repos.yml` is the source of truth. It currently contains seven production
repositories:

1. `OpenHands/OpenHands`
2. `OpenHands/enterprise` - pilot repository
3. `OpenHands/agent-canvas`
4. `OpenHands/software-agent-sdk`
5. `OpenHands/saas-deploy`
6. `OpenHands/OpenHands-Cloud`
7. `All-Hands-AI/infra`

`OpenHands/docs` is an additional integration repository, not a production
repo. Automation 3 needs read access to verify docs and Automation 7 needs PR
write access to reveal them after the flag is verified in production.

## Enterprise pilot release chain

A feature PR in `OpenHands/enterprise` is not considered in production merely
because it merged or received a release tag. Automation 3 follows the full
chain:

```text
OpenHands/enterprise merge commit
-> Enterprise semver release and enterprise-server image
-> OpenHands/OpenHands-Cloud chart image tag/appVersion bump
-> OpenHands/saas-deploy production chart pin
-> ArgoCD production sync semantics
```

The relevant Enterprise workflows are `.github/workflows/release.yml`,
`.github/workflows/tag-image.yml`, and `.github/workflows/bump-chart.yml`.
Frontend E2E coverage is exercised by `.github/workflows/fe-e2e-tests.yml`,
and feature flags currently live in `frontend/src/utils/feature-flags.ts`.

## Resolved decisions

- Linear project: `Feature Launches`, shared across all teams. Tickets use the
  PR author's team and fall back to `ALL` when no mapping exists.
- Bug-bash DRI: the original `(feat)` PR author.
- Bug-bash reminder: after 3 business days in production.
- Flag cleanup reminder: 3 months after the flag is verified on in production.
- GA requires both flag removal and a recorded public X/LinkedIn post URL.
- Council channel: `#tech-council`.
- Approval quorum: two distinct human channel members reacting
  `white_check_mark`.
- Blocking rule: any human channel member reacting `no_entry_sign` blocks
  approval until the reaction is removed.
- Council approval moves the ticket to `stage:council-approved`; it does not
  claim the flag is live.
- Docs remain `hidden: true` through council approval and are revealed only
  after independent production verification moves the ticket to
  `stage:flag-on`.
- Central artifacts live in `OpenHands/OpenHands` under
  `.github/landing-checklist/` and `.github/workflows/`.

## Remaining implementation gap

The production reconciler for this transition is not yet built:

```text
stage:council-approved -> stage:flag-on
```

For the Enterprise pilot it must verify the recorded feature flag in the
released Enterprise artifact and in `OpenHands/saas-deploy` production state.
It must record the evidence and timestamp, update the PR tracker and Slack
thread, and then trigger or perform the docs reveal. Council reactions alone
must never reveal docs or start the three-month cleanup clock.

## Deployment prerequisites

- Deploy event-driven Automations 1 and 2 on a publicly reachable OpenHands
  Automations host. This local workspace has no `RUNTIME_URL`, so GitHub cannot
  deliver webhooks here; local testing would need cron polling instead.

- Create every Linear label listed in `../linear/state-machine.md`, including
  `stage:council-approved`.
- Store `LINEAR_API_KEY`, `GITHUB_TOKEN`, and `SLACK_BOT_TOKEN` on the automation
  agent server.
- Give the Slack bot `chat:write`, `reactions:read`, `channels:read`,
  `users:read`, and `users:read.email`; add `groups:read` if `#tech-council`
  is private.
- Invite the bot to `#tech-council`.
- Give the GitHub integration read/write PR-comment access to all seven
  production repos, including private `OpenHands/enterprise`.
- Give the GitHub integration read access to `OpenHands/docs` for verification
  and PR write access for docs reveal.
- Install the reusable presubmit workflow in `OpenHands/enterprise`, run it
  once, and select its stable `landing-checklist` job from the branch-protection
  check picker for the pilot.
- Decide whether the PR author or marketing/comms owns the X/LinkedIn post;
  Automation 7 currently reminds the PR author.
- Resolve whether `All-Hands-AI/infra` should use the full product checklist or
  a scoped variant before broad rollout.

See `../tracker-format.md` for the shared GitHub/Slack rendering and
`../docs-visibility.md` for the hidden-doc lifecycle.
