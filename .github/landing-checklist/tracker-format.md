# Feature Landing Tracker - shared visual format

One canonical "where is this feature" picture, rendered identically in two
places so engineers never have to guess which surface is authoritative:

1. **GitHub PR** - a second sticky comment on the original `(feat)` PR,
   marker `<!-- landing-tracker:v1 -->` (separate from the GH Action's own
   `<!-- landing-checklist:v1 -->` pre-merge gate comment - see "Why two PR
   comments" below).
2. **Slack `#tech-council`** - one top-level message per feature, with every
   later update posted as a **threaded reply** to that same message.

Both are driven off the same source of truth: the Linear ticket's stage
label + checklist state. Nothing renders the tracker from anywhere else, so
the PR comment and the Slack thread can never show different stages.

## The 8-stage bar

```
[x] Review  ->  [x] Merged  ->  [x] In Prod  ->  [*] Bug Bash  ->  [ ] Council Review  ->  [ ] Council Approved  ->  [ ] Flag On  ->  [ ] GA
```

Legend: `[x]` done - `[*]` current stage - `[ ]` not yet reached. (When
rendering for real, use checkmark/circle-arrow/empty-box emoji as shown in
the "GitHub markdown rendering" and "Slack Block Kit rendering" examples
below - the bracket notation above is just for this plain-text summary
table.)

Maps 1:1 to the Linear labels in `linear/state-machine.md`, except
`stage:bug-bash-pending`/`stage:bug-bash-active` collapse into one visual
segment ("Bug Bash") - the tracker's freeform detail line underneath
carries the pending-vs-active distinction.

| Segment | Linear label(s) |
|---|---|
| Review | `stage:review` |
| Merged | `stage:merged` |
| In Prod | `stage:in-prod` |
| Bug Bash | `stage:bug-bash-pending`, `stage:bug-bash-active` |
| Council Review | `stage:council-review` |
| Council Approved | `stage:council-approved` |
| Flag On | `stage:flag-on` |
| GA | `stage:ga` |

## Full block content (both surfaces carry the same 4 parts)

1. The 8-stage bar above, with the current segment marked as current,
   everything before it marked done, everything after it marked pending.
2. A **current-stage detail line** - who's the DRI, what they're waiting
   on, and a due date if one applies (e.g. bug-bash nudge deadline, flag
   cleanup deadline).
3. The **6-item checklist**, each line showing done/pending, with inline
   detail where useful (e.g. release version, X/LinkedIn post link) -
   this is the same 6 items from `PULL_REQUEST_TEMPLATE_snippet.md`,
   always in this order:
   1. Docs added
   2. E2E test added
   3. Flag gated + exposed via Helm/embedded-installer
   4. Bug bash (3+ engineers) — includes a real Helm install test with the
      flag ON, plus a Replicated/embedded-cluster install test once that
      capability ships (see `repos.yml`'s `capabilities` block)
   5. Available in most recent release
   6. Included in an X/LinkedIn post
4. A **footer** cross-linking every surface: PR, Linear ticket, and (once
   it exists) the Slack thread permalink - so someone landing on any one
   of the three can jump to the other two. Every human-facing GitHub, Linear,
   or Slack message must also end with this disclosure:

   `_Automated by an OpenHands AI agent on behalf of the engineering team._`

### GitHub markdown rendering

Use the emoji below literally in the actual PR comment (GitHub renders
these fine, and renders `- [x]`/`- [ ]` as real checkboxes):

- done segment marker: a green checkmark emoji
- current segment marker: a blue circular-arrows (refresh) emoji
- pending segment marker: a white/empty square emoji

Example comment body:

```markdown
<!-- landing-tracker:v1 -->
### Feature Landing Tracker

[done] Review  ->  [done] Merged  ->  [done] In Prod  ->  [current] Bug Bash  ->  [pending] Council Review  ->  [pending] Council Approved  ->  [pending] Flag On  ->  [pending] GA

**Current stage:** Bug Bash (active) - DRI @<pr-author>, sub-issues open since 2026-06-12.

**Checklist**
- [x] Docs added for the feature on the website (hidden until flag-on - see docs-visibility.md)
- [x] E2E test added covering regression / up-to-spec behavior
- [x] Flag gated (`ENABLE_MY_FEATURE`) + exposed via Helm / embedded-installer
- [ ] Bug bash (3+ engineers) - in progress, 2 issues open (helm-test: pending, replicated-test: n/a-pending-support)
- [x] Available in most recent release (`v1.7.0`, shipped 2026-06-10)
- [ ] Included in an X / LinkedIn post

---
Linear: FEAT-42 (link) - Slack thread: (link)

_Automated by an OpenHands AI agent on behalf of the engineering team._
```

When actually posting this comment, replace `[done]`/`[current]`/`[pending]`
with the real emoji (checkmark / refresh-arrows / empty-box) and replace
the plain-text links with real markdown links - the bracket/plain-text
placeholders above exist only because this reference file needs to stay
plain-ASCII-safe when edited by different tools.

### Slack Block Kit rendering

Slack mrkdwn does **not** render `- [x]` as a checkbox - use the same
checkmark/empty-box emoji as bullet lines instead. Post via
`chat.postMessage` (top-level, from Automation 3) or with `thread_ts` set
to reply in the same thread (every later automation). Requires a
`SLACK_BOT_TOKEN` secret with `chat:write` scope, and the bot invited to
`#tech-council`.

```json
{
  "channel": "#tech-council",
  "blocks": [
    {
      "type": "header",
      "text": { "type": "plain_text", "text": "Feature: <Feature Title>", "emoji": true }
    },
    {
      "type": "context",
      "elements": [
        { "type": "mrkdwn", "text": "Review (done)  ->  Merged (done)  ->  In Prod (done)  ->  *Bug Bash (current)*  ->  Council Review  ->  Council Approved  ->  Flag On  ->  GA" }
      ]
    },
    {
      "type": "section",
      "text": { "type": "mrkdwn", "text": "*Current stage:* Bug Bash (active)\n*DRI:* <@U0123ABCD> (PR author)\n*Since:* 2026-06-12" }
    },
    {
      "type": "section",
      "text": { "type": "mrkdwn", "text": "*Checklist*\nDocs: done (hidden until flag-on)\nE2E test: done\nFlag + Helm/embedded-installer: done\nBug bash (3+ engineers): pending - 2 issues open (helm-test pending, replicated-test n/a-pending-support)\nAvailable in latest release: done (v1.7.0)\nX/LinkedIn post: pending" }
    },
    {
      "type": "context",
      "elements": [
        { "type": "mrkdwn", "text": "<https://github.com/OpenHands/enterprise/pull/1234|PR #1234> - <https://linear.app/.../FEAT-42|FEAT-42>\n_Automated by an OpenHands AI agent on behalf of the engineering team._" }
      ]
    }
  ]
}
```

Later automations replying in-thread should post a **lighter** update, not
the full block set again - just a `section` with what changed, e.g.:

```json
{
  "channel": "#tech-council",
  "thread_ts": "1750000000.123456",
  "blocks": [
    { "type": "section", "text": { "type": "mrkdwn", "text": "Bug bash complete (4 engineers, 2 issues found & fixed). Moving to *Council Review*. Council members: react ✅ to approve or 🚫 to block. Two distinct human approvals are required; reactions from the original PR author do not count, and any blocking reaction wins until removed. <https://linear.app/.../FEAT-42|FEAT-42>\n\n_Automated by an OpenHands AI agent on behalf of the engineering team._" } }
  ]
}
```

The Council Review reply (posted by Automation 5) is the one point in the
thread that should read as an explicit ask. Automation 5 records that reply's
exact `channel` and `ts` on Linear; Automation 6 counts reactions on that
message only. Two distinct human members of `#tech-council`, excluding the
original `(feat)` PR author, must react `✅`; any valid `🚫` from an eligible
member blocks approval until removed.

## Why two PR comments, not one

The GH Action (`landing-checklist-reusable.yml`) runs synchronously in CI on
every `pull_request` event and is a **required status check** - it must own
its own comment (`landing-checklist:v1`) so its pass/fail logic can never
race with an async automation. The tracker comment (`landing-tracker:v1`) is
a separate, non-blocking, informational comment created once by Automation 1
and updated in place by Automations 2 through 7 as the Linear stage changes.
Two markers, two owners, no race condition, no duplicate comments.

## Who updates what, when

| Trigger | Linear stage change | PR tracker comment (`landing-tracker:v1`) | Slack `#tech-council` |
|---|---|---|---|
| PR opened (`(feat)` title) | create ticket, `stage:review` | **Automation 1 creates** the tracker comment: bar at current=Review | - (not yet; see note below) |
| PR merged | -> `stage:merged` | Automation 2 updates: Review+Merged done, In Prod current | - |
| Confirmed in prod | -> `stage:in-prod` | Automation 3 updates: In Prod done, Bug Bash current | **Automation 3 posts the top-level thread-starter message** (first time Slack gets involved - see note below), then records the returned `channel`+`ts` as a Linear comment for later automations to thread off of |
| Bug-bash nudge / escalation | (no stage change on nudge; escalation stays `stage:bug-bash-pending`) | Automation 4 updates the DRI/due-date detail line only | Automation 4 replies in-thread with the nudge/escalation |
| Bug bash clear | -> `stage:council-review` | Automation 5 updates: Bug Bash done, Council Review current | Automation 5 posts the reaction request and records `council-approval-message: <channel>/<ts>` on Linear |
| Two valid `✅`, no valid `🚫` | -> `stage:council-approved` | Automation 6 updates: Council Review done, Council Approved current | Automation 6 replies that approval is recorded and production verification is pending |
| Flag verified enabled in production | -> `stage:flag-on` | Production reconciler updates: Council Approved done, Flag On current; Automation 7 reveals docs | Production reconciler replies with the production evidence |
| 3-month flag-cleanup nudge | (no stage change) | Automation 7 updates the cleanup-nudge detail line | Automation 7 replies in-thread nudging cleanup |
| Flag removed and social post URL recorded | -> `stage:ga` | Automation 7 updates: Flag On+GA done, marks tracker "done" and links social evidence | Automation 7 replies in-thread with a closing celebratory message |

**Why Slack visibility starts at Automation 3, not Automation 1:** posting
every opened `(feat)` PR to `#tech-council` on day one would make the
channel mostly noise about things that haven't shipped yet and may never
reach prod. Tech council's job starts in earnest once something is actually
live and heading toward a bug bash - that's also the point where a
human-readable Slack thread becomes genuinely useful to track. Before that,
the PR itself (via the tracker comment) is the venue. If you'd rather have
full visibility from PR-open, this is a one-line change (move the
thread-start into Automation 1) - flag it if you want that instead.

**Known gap - `stage:council-approved` -> `stage:flag-on`:** the Slack
reaction gate is implemented, but the production reconciler is not. For the
`OpenHands/enterprise` pilot it must verify the recorded `ENABLE_<FEATURE>`
change in the released Enterprise image and the production configuration in
`OpenHands/saas-deploy`, then record that evidence before advancing. Council
approval alone must never reveal docs or start the three-month cleanup clock.

**Docs visibility during the journey:** the checklist's docs item (#1) has
its own hide/reveal lifecycle layered on top of this tracker - the page
exists from merge time onward but stays undiscoverable (`hidden: true` in
Mintlify frontmatter) until `stage:flag-on`. See `docs-visibility.md` for
the full mechanism and which automation does what.
