# The daily pass

The weekly pass in [maintenance.md](maintenance.md) proves every feature live.
A daily pass cannot: 27 families and 700-odd sub-features take longer than one
agent's day, and most of the map did not change since yesterday. The daily
pass answers a narrower question with the same rigor: **did anything that
merged since the last pass change what a user sees, and does the rest still
hold where it is cheapest to look?** It is the delta pass that
[maintenance.md](maintenance.md) allows, and it moves the same baseline.

This is a procedure, not a scheduler: run it from whatever runs you daily, do
not create automations or post externally unless asked, and hand what you
found to a person.

## Inputs and hand-off

A scheduled pass starts in a fresh session with no memory of yesterday, so
everything it needs from the previous pass lives in the repository or on an
open PR:

- `BASE` is the `Maintenance baseline: main@<sha> (<date>)` line of
  [the map index](feature-map/README.md), the previous pass's `TARGET`.
  `control-openhands map baseline` reads it and says how far `HEAD` is from
  it; `map affected` starts there when no `--base` is given. Fetch `main`
  and freeze its full SHA as `TARGET`; deepen a shallow clone until `BASE` is
  present.
- A pass proposes `TARGET` as the next baseline in its PR
  (`control-openhands map baseline --set "$TARGET"` moves the line; the SHA
  must be on `main`) only when tier 2 finished: every family `map affected`
  listed was driven, every commit in `BASE..TARGET` was resolved to its PR,
  and the closed-issue rows below were re-driven. Merging that PR accepts the
  baseline. A pass that ran out of time or was blocked in tier 2 leaves the
  line where it was, even when its PR carries other map or CLI fixes, and says
  so in its report (maintenance.md's blocked/partial outcome); a pass that
  changes nothing leaves it too, so the next pass covers a longer range. If an
  earlier pass's PR is still open, the next pass continues on that branch and
  takes `BASE` from the line there.
- The rotation position (which families get their full recipe today) is
  derived from the date, below, so no state carries it.
- Yesterday's `evidence/ledger.jsonl` is an optional input: when the previous
  run's ledger is at hand (a CI artifact, a shared directory), pass it to
  `evidence report --baseline`; without it the report has no "Changes since
  baseline" section and the verdict rests on today's fail and blocked rows.

Record the UTC time, model and spend budget, and the accounts and keys
available, as the weekly pass does. A pass that runs out of time says so and
names what it did not reach; it never rolls the remainder into tomorrow
silently.

## Tiers, in order

Run the tiers in this order and stop when the budget runs out; each tier is
cheaper than the next and catches a different kind of rot.

1. **Static (minutes, no stack).** `control-openhands map check`,
   `map coverage` and `map testids`. A `Source:` path that is gone, or a test
   id the map drives that no literal in `src/` accounts for, is drift found
   before any browser opens. Fix it in the map
   with a live drive later today, or report it. Zero unresolved test ids is
   the normal state; treat a new one as a rename until a drive says otherwise.
2. **Changed (the core).** `control-openhands map affected --target $TARGET`
   (`--base` defaults to the recorded baseline). Launch, doctor, then drive
   **every** sub-feature the
   listed families map for the changed paths, on each entry point the family
   lists, at desktop and phone viewports for UI changes. Widen `shared` paths
   to their consumers (an API client or store change touches every page that
   reads it; pick the families whose `Source:` lines name those consumers, not
   one convenient screen). An `unmapped` path under `src/` is a map gap: a new
   surface to map per [mapping.md](mapping.md), or a `Source:` line to extend.
   Resolve each merged commit in `BASE..TARGET` to its PR and keep intent
   (`documented`, `undocumented`, `contradictory`) separate from runtime
   results, as [maintenance.md](maintenance.md) step 2 describes. The delta
   pass has a second half: every bullet whose Gotchas or Known failure note
   links an issue that closed since `BASE` (check each linked issue's state)
   is re-driven too, its row recorded and its note updated. A fix that landed
   in the Agent Server or the automation service shows up only this way.
3. **Smoke (every family, cheaply).** For each family not already driven in
   tier 2: open its first entry point, run the family's `errors --app-only`
   sweep, and drive the one bullet that proves the page's main state (a list
   renders, a form opens). One `evidence add` row per family, under the
   sub-feature ID that bullet names. This catches a page that stopped loading
   without re-proving every row.
4. **Rotation (depth over the week).** Drive the full recipe of about a
   seventh of the families, four a day, so every family gets a full live pass
   once a week between weekly maintenance passes. The day's slice comes from
   the date, not from yesterday's notes: with `d` the UTC day of the year,
   today's families are the ones at positions `(d mod 7) * 4` to
   `(d mod 7) * 4 + 3` of the index table, in ID order (position 24 onwards
   holds F25 to F27). Model-backed bullets run only inside the LLM budget;
   without a key they are `blocked` with the missing prerequisite.

## Reading the result

Render `control-openhands evidence report`, with `--baseline <yesterday's
ledger>` when that ledger is at hand. Fail and blocked rows come first; the
"Changes since baseline" section, when present, is the daily verdict:

- **Newly failing** on a family that tier 2 drove for a changed path: an
  introduced-in-range candidate. Re-drive the same minimal recipe on `BASE`
  (a second `launch --new` on a worktree at `BASE`) before calling it a
  regression; otherwise classify it as reproduced-on-both, environment or
  harness gap, or origin-unconfirmed (maintenance.md step 4).
- **Newly failing** elsewhere, or **newly passing** without a PR that explains
  it: an undocumented change; cite the commit and ask.
- **Not checked this run** rows are the rotation moving on, not regressions;
  the family table shows which families got depth today.
- Without yesterday's ledger, read today's fail rows the same way: a fail on a
  family tier 2 drove for a changed path is a candidate to re-drive on
  `BASE`; elsewhere it is a finding to triage, with the family's last known
  state taken from the map (a bullet's Known failure note, a linked issue).

Triage as the skill does: map drift (fix, with live evidence), harness gap
(extend the CLI, re-drive), product defect (keep the evidence, search issues,
file in the owning repository, keep it out of the map PR), blocked (name the
prerequisite and the route). At most one PR of proven map and CLI corrections
per pass, from current `main` (or from the previous pass's branch while its
PR is open), carrying the moved baseline line and saying "delta" in its
report; never edit product code in a pass.

## Several agents

Shard by family, never by bullet: each agent gets whole families, its own
`launch --new`, its own `OH_VERIFY_RUN`, and writes only its own ledger. One
coordinator merges the ledgers' reports and owns the hand-off. A result that
names no run, revision or entry point does not count; a gap is not a pass.

## What the daily pass is not

- Not the full pass: it does not re-prove every bullet, and the baseline it
  moves says only that the day's changes were covered, which is why an
  unfinished tier 2 never moves it. When tier 2 finds that
  most of the map is affected (a shared component or style change), say so
  and run the full procedure instead. A full pass is still needed weekly or
  before a release, because Agent Server and automation releases change
  behavior without touching this repository.
- Not CI: a green Playwright run or a passing `map check` is an input to a
  verdict, never the verdict.
- Not a bug-fix lane: product defects are filed, not fixed, inside the pass.
