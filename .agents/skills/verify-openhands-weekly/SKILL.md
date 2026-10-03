---
name: verify-openhands-weekly
description: >
  This skill should be used for a "weekly OpenHands feature audit", "verify main
  since last week", or "maintain the OpenHands feature map". Reconciles main's
  changes with merged PR intent, drives mapped user features, compares baseline
  evidence, and proposes only proven verification-map or harness corrections.
---

# Weekly OpenHands verification

Answer three separate questions: **Is the feature present? Does it work and look
right? Is the change from last week documented and intended?** A merged PR alone
does not prove the last two. A working new behavior with missing intent evidence
is still an unresolved finding.

Use [verify-openhands](../verify-openhands/SKILL.md) for runtime operation and its
[feature map](../verify-openhands/features/README.md) as the coverage inventory.
This skill is a reusable weekly procedure, not a scheduler. Do not create an
automation, enable a recurring task, merge code, or post externally unless asked.

## 1. Freeze the comparison and authority

Record UTC run time, repository, primary surface/backends, authorized test accounts,
allowed model/spend budget, and permitted side effects. Without a working LLM or
Cloud account, continue credential-free coverage and explicitly mark dependent
checks blocked; do not spend money or approve device codes speculatively.

Fetch `OpenHands/OpenHands` main and freeze its full SHA as TARGET. Check
`git rev-parse --is-shallow-repository`; deepen history until the previous baseline
(or bootstrap cutoff and its preceding history) is available. Do not interpret an
empty log from a shallow checkout as “no changes this week.” Prefer the previous
completed audit's recorded TARGET as BASE, together with its immutable
report/evidence location. Check ancestry:

```sh
git merge-base --is-ancestor "$BASE" "$TARGET"
git log --first-parent --format='%H %cI %s' "$BASE..$TARGET"
git diff --name-status "$BASE" "$TARGET"
```

If there is no prior audit, record a first-run bootstrap and choose the last
first-parent commit at/before the agreed UTC cutoff (normally seven days before
this run): `git rev-list --first-parent -1 --before="$CUTOFF" "$TARGET"`.
Resolve and record both SHAs before testing. A date-derived source baseline is not
last week's live evidence. If the baseline is missing, non-ancestral, or cannot
run, report the exact comparison gap rather than declaring visual parity.

Use separate worktrees and state. Never reset a dirty checkout or rebuild the
baseline from today's source. Do not auto-promote a failed/partial run or overwrite
last week's screenshots; record it separately and require an explicit baseline
acceptance decision. Keep the UTC reporting window distinct from the commit range
when the previous successful audit is older than seven days.

## 2. Build the PR-intent ledger

Enumerate commits in BASE..TARGET and resolve their associated PRs through GitHub's
API. For example, `gh api --paginate repos/OpenHands/OpenHands/commits/$SHA/pulls`
handles associations that a `(#123)` commit-message regex misses. Read each merged
PR's body, linked acceptance criteria, relevant discussion/design decisions and
changed files (`gh api --paginate repos/OpenHands/OpenHands/pulls/$PR/files`).
Filter for changes actually present in TARGET and the intended base branch.
Use recent merged-PR search only as a cross-check, not the source of truth: squash,
rebase, direct commits, reverts, and older merges can defeat date-only searches.

For every change, record:

| Commit / PR | Changed paths | Feature IDs / entry points | Intended result + exact PR evidence | Runtime proof needed |
|---|---|---|---|---|

Include direct commits and unassociated/unavailable PRs explicitly as unknown
intent. Record retrieval time and exact source links/quotes; do not treat a title,
green CI, or mere mention of the feature as a behavioral specification. Inspect
reverts and superseding PRs to determine the net intended behavior at TARGET.
If an API is truncated or unavailable, retain the incomplete-evidence flag.

Keep **intent** (`documented`, `undocumented`, `contradictory`) independent of
**runtime** (`pass`, `fail`, `blocked`, `not-run`). A visual change can work yet be
undocumented. Do not edit PR descriptions—especially the human-only `HUMAN:`
section—to retroactively justify it. Ask the maintainer to resolve ambiguous intent.

## 3. Reconcile the feature map with source

Read the index and all feature files. Check missing, duplicate and dead links/IDs.
Assign read-only source readers by feature if delegation is available; otherwise
work serially. Return source anchors, likely drift/new surfaces, and a concrete
live recipe. Children must not edit files or drive the shared browser.

Map **every changed path** to a feature or an explicit non-user-facing rationale.
For shared CSS, primitives, API clients, config, and dependencies, expand to their
consumers rather than sampling the one convenient screen. Check `src/routes.ts`,
menus, launcher flags and new tests for surfaces absent from the map. Keep route
presence distinct from reachability and behavior; modals/panels are features too.

Retain stable IDs. Add newly discovered features; mark deliberate removals with
the authorizing PR and retirement check instead of silently deleting their rows.
Treat source disagreement as a hypothesis, not runtime proof. Combine overlapping
recipes into a few coherent disposable workflows without erasing entry-point coverage.

## 4. Drive main and compare

Follow the runtime skill, doctor first, with one coordinator owning the browser.
Visit every mapped feature family and record each sub-feature/entry point's result;
exercise all changed behavior and its shared consumers. A quick smoke pass does
not complete the weekly audit. Explicitly record untouched checks as not-run.

Use the same browser, viewport, font installation, theme, fixture content, state,
and backend versions for baseline/target comparisons. Record version changes when
that is impossible, and distinguish Canvas, SDK, automation and environment effects.
Recreate disposable state; never share a writable database across revisions.

Read screenshots side by side and inspect numerical overflow/geometry and console
errors. Use pixel diffs when useful, but review noise from fonts, timestamps,
network data and rendering. A nonzero pixel diff is not automatically a bug; a
zero diff does not establish functional correctness. Do not alter CSS, hide the
failing element, or replace baseline screenshots to make the comparison pass.

For an apparent regression, execute the same minimal recipe on BASE and TARGET.
Only claim introduced-in-range when BASE passes and TARGET fails under comparable
conditions. Otherwise classify as reproduced-on-both, environment/harness gap,
intent gap, or observed-origin-unconfirmed. Unchanged source is not sufficient
proof that a runtime defect predates the range.

## 5. Triage and maintain

- **Map drift:** correct an instruction only with current source and live evidence.
- **Harness gap:** fix within the verification skill; re-run the affected recipe.
- **Product defect:** preserve evidence and report separately. Check existing issues
  before filing; never modify product code during this maintenance run.
- **Undocumented/contradictory change:** cite the PR and observed difference; request
  clarification, not a rewritten expectation that silently approves the change.
- **Blocked capability:** record attempted route/action and the concrete missing
  prerequisite. An unsupported local endpoint is neither Cloud success nor proof
  of a product bug. Do not add mocks to turn the cell green.

Default edit scope is the verification skills and their feature maps only. Treat
known issues as repro candidates: check current status, retry, and link fresh
results. Do not repeatedly file the same issue or silently waive its failed check.

## 6. Report and hand off

Use [the report contract](../verify-openhands/references/report.md). Lead with the
verdict, feature/check coverage counts, intent gaps, defects, and blockers. Link
reviewed evidence and the PR-intent ledger. Keep three outcomes distinct:

- **Clean in the declared scope:** all required checks executed, expected behavior
  and intent reconciled, no unexplained failures or gaps. No maintenance PR.
- **Changed:** proven map/harness corrections are ready. If authorized, open at most
  one maintenance PR from current main, separate from product bug fixes.
- **Partial/blocked:** explain what could not be verified; do not call the run clean
  or advance the accepted baseline. Safe completed findings still belong in the report.

Include the separate product verdict even when maintenance corrections ship.
Follow the repository's PR template/review rules; add AI attribution to external
reports, issues and PRs. Leave `HUMAN:` content to the human contributor.
Finish with owned-instance cleanup, retained evidence paths, and the precise next
inputs needed to complete any blocked checks.
