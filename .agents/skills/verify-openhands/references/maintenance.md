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
(`pass`, `fail`, `blocked`, `not-run`, and `xfail`/`xpass` for known bugs). A working change can still be
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

Reconcile before driving. Every feature file has a returned summary; re-run a
reader that came back empty or failed instead of treating its feature as clean.
Merge the readers' recipes into as few app states as practical (one fixture
set, one LLM setup) so the live pass does not rebuild the same state per
family. Spot-check each cited drift against source before you trust it; do not
repeat the source analysis for claims a reader found clean (their live coverage
in §4 still applies).

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

1. Never drive an instance not doctored since its last surprise. Run
   `control-openhands doctor` before the first drive, on each `launch --new`,
   and again after any failed drive. Doctor cannot see a wedged UI on a healthy
   stack: `browser reload`, return to a known state or relaunch rather than
   driving on and hoping.
2. Evidence captured so far survives every cleanup: check it at its recorded
   path after `stop`, do not assume it.
3. Nothing a drive started outlives its usefulness: fixtures deleted, schedules
   disabled, runs stopped. Clean a failed drive's leftovers too, whether the
   run is stuck or exited. Clean only that drive's own leftovers: never touch
   another worker's run, and leave up a run that the coordinator still uses.

A doctor failure caused by skill drift is drift: fix it in this skill's
directory, restart only what the fix invalidated, and retry once; a second
failure makes the pass blocked. A feature that cannot be reached is `blocked`
only with the concrete prerequisite and the route attempted; if the map omits
that prerequisite, that is drift too. Re-drive any CLI fix before shipping it.
Run the final `control-openhands stop` after the last drive of the pass,
including those re-drives.

For an apparent regression, run the same minimal recipe on BASE and TARGET.
Claim introduced-in-range only when BASE passes and TARGET fails under
comparable conditions (same browser, viewport, fixture state, backend pins).
Otherwise classify as reproduced-on-both, environment/harness gap, intent gap,
or origin-unconfirmed. Once the regression is a known bug with a `known-bug`
block, the same rule is executable: `bug run <ID> --attempts 2` on each run and
`bug proof` (introduced-in-range is the proof read the other way: BASE `xpass`
twice, TARGET `xfail` twice).

To attribute a **fix** (a PR that closes a recorded bug, or an issue closed
since BASE), reproduce twice on the baseline and confirm twice on the fix,
with the same recipe, viewport, entry point and backend pins:
`control-openhands bug run <ID> --attempts 2 --label baseline --run "$BASE_RUN"`,
the same with `--label fix --run "$FIX_RUN"`, then
`control-openhands bug proof <ID> --baseline "$BASE_RUN" --baseline-label baseline --fix "$FIX_RUN" --fix-label fix`.
Each attempt records its revision, backend versions, viewport, recipe hash,
failing line and assertion in `evidence/known-bugs/attempts.jsonl`. Only
`verified` is a fix. A baseline that reproduces fewer than two times out of
two, or attempts under different inputs, are `inconclusive`: say so and do not
remove the note. A setup failure, a stopped backend, a timeout or another
symptom is a `fail`, never a reproduction. Use screenshots side by side; a pixel difference is not
automatically a bug and a zero difference does not prove behavior.

## 5. Triage

- **Map drift**: wrong or missing user-POV description. Fix the entry with
  current source and live evidence.
- **Harness gap**: working behavior the CLI cannot drive. Extend the CLI, keep it
  executable and documented in `--help`, re-drive.
- **Product defect**: keep the evidence, search existing issues, then file in the
  owning repository (Canvas here; Agent Server/SDK in
  `OpenHands/software-agent-sdk`; scheduling/dispatch in `OpenHands/automation`),
  following that repository's issue template. The fix stays out of the
  maintenance PR, but the defect goes in: the same PR adds a `Known failure
  (reproduced <date>, #<issue>)` note to the bullet that reproduces it, next to
  the expected result, or a `Known issue #<issue>` Gotchas entry when no recipe
  can drive it.
- **Bugs filed outside the pass**: open `bug` issues without a supported
  map or ledger outcome are repro candidates too, regardless of creation date.
  Include newly opened, reopened and newly bug-labeled reports. Reproduce each and add its
  note; when one does not reproduce, comment on the issue with the run instead.
  `ready-for-dev` prioritizes live work but does not exclude older actionable
  reports. [bug-backfill.md](bug-backfill.md) defines the inventory outcomes and
  their optional GitHub labels; a label must have a map note or report behind it.
- **Undocumented or contradictory change**: cite the PR and the observed
  difference; ask the maintainer instead of rewriting the expectation.
- **Known issues** linked in Gotchas are repro candidates: retest, link fresh
  results, never file duplicates or silently waive the check. Run every
  `known-bug` block (`control-openhands bug list`, then `bug run <ID>`): an
  `xfail` keeps the note, with its date refreshed. An `xpass` is a map review,
  not a pass: find what fixed it (a merged PR, a backend release), prove it
  with `bug proof` where a baseline is at hand, then drop the note and keep the
  block's commands as the bullet's normal steps. A `fail` is a broken recipe or
  a new symptom: triage it like any other failure.

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
