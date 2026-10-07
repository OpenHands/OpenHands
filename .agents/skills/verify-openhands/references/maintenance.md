# Maintaining the map (periodic or weekly)

A feature map rots the moment the app changes. This pass keeps it honest and
answers three separate questions about the interval since the last pass:
**Is each feature present? Does it work and look right? Was each change
documented and intended?** A merged PR proves none of the last two. Between
full passes, [daily.md](daily.md) runs the delta pass each day against the
same `Maintenance baseline` line of the map index, and moves it the same way.

The unit of rigor is the feature: every feature file gets source coverage and
live coverage, without re-proving every sentence. This is a procedure, not a
scheduler: do not create automations, merge, or post externally unless asked.

## Outcomes

Pick one and say which:

- **clean**: every feature got source and live coverage, intent is reconciled,
  nothing worth shipping. No branch, no PR.
- **changed**: one PR of proven map/CLI corrections, separate from product fixes.
- **blocked/partial**: coverage could not finish. Say exactly what and why; do
  not call it clean or advance the accepted baseline.

Edit scope is this skill's directory only (SKILL.md, references/, scripts/).
Never edit product code during a maintenance pass.

## 1. Freeze the comparison

Record the UTC run time, budget (model, spend), accounts available and permitted
side effects. Fetch `main` and freeze its full SHA as TARGET. Check
`git rev-parse --is-shallow-repository`; deepen until BASE is present (an empty
log from a shallow clone is not "no changes"). BASE is the previous completed
pass's recorded TARGET, kept on the `Maintenance baseline` line of
[the map index](feature-map/README.md); on a first run, the last first-parent
commit before the agreed cutoff
(`git rev-list --first-parent -1 --before="$CUTOFF" "$TARGET"`). A pass that
changes the map updates that line to its TARGET in its PR, so merging the PR
accepts the new baseline.

A daily pass may be a **delta pass**: source and live coverage only for the
features whose paths changed in BASE..TARGET (§3) and for rows that link an
issue closed since BASE. It says "delta" in its report. A full pass is still
needed now and then (weekly, or before a release), because Agent Server and
automation releases change behavior without touching this repository.

```sh
git merge-base --is-ancestor "$BASE" "$TARGET"
git log --first-parent --format='%H %cI %s' "$BASE..$TARGET"
git diff --name-status "$BASE" "$TARGET"
```

Use separate worktrees for BASE and TARGET and `control-openhands launch --new`
for each, so state and ports never mix. Never rebuild the baseline from today's
source or overwrite last pass's evidence.

## 2. Build the PR-intent ledger

Resolve every commit in BASE..TARGET to its PR through the API
(`repos/OpenHands/OpenHands/commits/<sha>/pulls`; commit-message `(#123)`
regexes miss rebases and direct pushes). Read each PR's description, linked
issue acceptance criteria, relevant review discussion and file list.

| Commit / PR | Changed paths | Feature IDs / entry points | Intended result + quoted evidence | Runtime proof needed |
|---|---|---|---|---|

Record direct commits and unresolvable PRs as unknown intent. Keep intent
(`documented`, `undocumented`, `contradictory`) independent of runtime
(`pass`, `fail`, `blocked`, `not-run`). A working change can still be
undocumented. Never edit a PR description (the `HUMAN:` section is human-only)
to justify a change after the fact.

## 3. Index hygiene and source wave

Run `control-openhands map check`, `map coverage` and `map testids`; fix
missing, duplicate and dead entries, `Source:` paths that are gone and test
ids no literal in `src/` accounts for (`map check --fix-counts` refreshes the
index counts). Then give one read-only reader per
feature file (parallel if delegation is available). Each reads current source for that feature and returns:
summary, source entry points, likely drift with citations (or none), new
surfaces missing from the map, and one live recipe. Readers never edit files or
drive the browser.

Re-check stale harness claims whenever the CLI gained verbs: grep the map for
`blocked`, `not run`, `harness gap`, `no verb` and `cannot be driven`, and
re-drive any that a current verb can now reach (compare with
`control-openhands --help` and `help browser`).

Map **every changed path** in BASE..TARGET to a feature ID or an explicit
non-user-facing reason: `control-openhands map affected --base "$BASE" --target
"$TARGET"` does the first cut from the `Source:` lines. Its `shared` paths (components, CSS, API clients,
hooks, stores, settings schemas, dependency bumps) expand to their consumers
rather than sampling one convenient screen; its `unmapped` paths under `src/`
are map gaps. A new surface needs a concrete source path before it is called
missing; then add it per [mapping.md](mapping.md).

## 4. Live pass

Required even when source looks clean. One coordinator owns each launched run
and drives it serially; parallel live work needs one `launch --new` per worker.
Exercise every feature at least once and every changed behavior on all its entry
points, at desktop and phone viewports for UI changes. Hold three invariants:

1. Never drive an instance not doctored since its last surprise
   (`control-openhands doctor`; reload or relaunch when the UI is wedged).
2. Evidence captured so far survives every cleanup (check it after `stop`).
3. Nothing a drive started outlives its usefulness: fixtures deleted, schedules
   disabled, runs stopped.

A doctor failure caused by skill drift is drift: fix it and retry once. A
feature that cannot be reached is `blocked` only with the concrete prerequisite
and the route attempted; if the map omits that prerequisite, that is drift too.
Re-drive any CLI fix before shipping it.

For an apparent regression, run the same minimal recipe on BASE and TARGET.
Claim introduced-in-range only when BASE passes and TARGET fails under
comparable conditions (same browser, viewport, fixture state, backend pins).
Otherwise classify as reproduced-on-both, environment/harness gap, intent gap,
or origin-unconfirmed. Use screenshots side by side; a pixel difference is not
automatically a bug and a zero difference does not prove behavior.

## 5. Triage

- **Map drift**: wrong or missing user-POV description. Fix the entry with
  current source and live evidence.
- **Harness gap**: working behavior the CLI cannot drive. Extend the CLI, keep it
  executable and documented in `--help`, re-drive.
- **Product defect**: keep the evidence, search existing issues, then file in the
  owning repository (Canvas here; Agent Server/SDK in
  `OpenHands/software-agent-sdk`; scheduling/dispatch in `OpenHands/automation`),
  following that repository's issue template. Keep it out of the maintenance PR.
- **Undocumented or contradictory change**: cite the PR and the observed
  difference; ask the maintainer instead of rewriting the expectation.
- **Known issues** linked in Gotchas are repro candidates: retest, link fresh
  results, never file duplicates or silently waive the check.

## 6. Report and hand off

Use [the report contract](report.md): verdict first, then family and check
counts, intent gaps, defects with issue links, blocked checks with their
prerequisites (`control-openhands evidence report --baseline <previous run>`
renders the counts, the blocked rows' notes and what changed since the last
accepted ledger). For **changed**, open at most one PR from current `main`, re-read
every changed file first, and follow the repository's PR template. Finish with
`control-openhands stop`, retained evidence paths, and the exact inputs needed to
unblock what was blocked. Name the proposed next baseline; never replace an
accepted baseline automatically.
