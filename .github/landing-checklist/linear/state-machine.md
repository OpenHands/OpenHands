# Linear "Feature Landing" state machine

Why Linear and not the GitHub PR: the landing process runs for months after
the PR is merged and closed. We need one durable record per feature that
survives PR merge, so every automation reads/writes state on a single Linear
issue rather than trying to reconstruct state from GitHub each time.

## Setup (one-time, human)

1. Create a Linear **project** called `Feature Launches`, shared/owned
   across all teams (`CS`, `PRD`, `PLTF`, `FDE`, `OHE`, `OSS`, `ALL`) rather
   than scoped to one — any team can land a `(feat)` PR in a production
   repo, and its ticket lands in this one shared project. New issues are
   created under whichever team the PR author's Linear account belongs to
   (falls back to `ALL` if that can't be resolved), with the project set to
   `Feature Launches` regardless of team.
2. Create these **labels** (used as the state machine) as workspace labels, or
   create the full set for every participating team. Automation 6 prefers a
   matching team label and falls back to a unique workspace label:
   - `stage:review` — PR open, checklist in progress
   - `stage:merged` — PR merged, not yet confirmed in prod
   - `stage:in-prod` — confirmed shipped to production
   - `stage:bug-bash-pending` — reminder sent, awaiting scheduling
   - `stage:bug-bash-active` — bash scheduled, sub-issues open
   - `stage:council-review` — bug bash clear, awaiting Slack reactions
   - `stage:council-approved` — two valid council approvals, flag not yet verified on
   - `stage:flag-on` — approved and independently verified on for all users
   - `stage:ga` — flag removed, on by default, and public social-post evidence recorded
   - `landing:deferred` — non-stage label for a bug-bash issue explicitly moved
     into a named next cycle; without both this label and a target cycle, an
     open issue still blocks council review
3. Tech council approval happens in **`#tech-council`**. Automation 5 posts
   an approval request in the feature's existing Slack thread. Automation 6
   counts reactions deterministically: two distinct human channel members
   reacting `✅` approves; any channel member reacting `🚫` blocks until that
   reaction is removed. Bot/app reactions, reactions from users outside the
   channel, and reactions from the original `(feat)` PR author do not count.
   Automation 5 maps the PR author from their Linear email and fails closed if
   it cannot. The bot needs Slack scopes `chat:write`, `reactions:read`,
   `channels:read`, `users:read`, `users:read.email`, and (if the channel is
   private) `groups:read`.
4. Create an **issue template** "Feature Landing" with the checklist body
   (mirrors the PR template) so every ticket looks consistent — see
   `feature-landing-issue-template.md` in this folder.

## State transitions (who moves the label, and how)

| From | To | Trigger | Owner |
|---|---|---|---|
| (none) | `stage:review` | PR opened with `(feat)` title in an allowlisted repo | Automation 1 (event) |
| `stage:review` | `stage:merged` | PR merged | Automation 2 (event) |
| `stage:merged` | `stage:in-prod` | merge commit SHA found in production release/deploy | Automation 3 (cron) |
| `stage:in-prod` | `stage:bug-bash-pending` | 3 business days elapsed with no bug bash scheduled | Automation 4 (cron) |
| `stage:bug-bash-pending` | `stage:bug-bash-active` | child issues or structured `bug-bash-report` appears | Automation 4 (cron, next pass) |
| `stage:bug-bash-active` | `stage:council-review` | valid 3+ attendee report, all findings fixed or explicitly moved to a named next cycle, checklist items 1-5 have evidence, and approval request is posted in `#tech-council` | Automation 5 (cron) |
| `stage:council-review` | `stage:council-approved` | two distinct human `#tech-council` members react `✅`, with no `🚫` from a channel member | Automation 6 (deterministic reaction poller) |
| `stage:council-approved` | `stage:flag-on` | flag-enablement change independently confirmed in production | Production reconciler (not yet implemented) |
| `stage:flag-on` | `stage:ga` | 3 months elapsed, flag removed in code, and supported public X/LinkedIn post URL recorded | Automation 7 (cron) |

Each automation is idempotent: it re-derives "what state is this issue in"
from its label on every run, so a missed or duplicated run is harmless.
