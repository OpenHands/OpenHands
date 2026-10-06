# The daily pass

The weekly pass in [maintenance.md](maintenance.md) proves every feature live.
A daily pass cannot: 27 families and 700-odd sub-features take longer than one
agent's day, and most of the map did not change since yesterday. The daily
pass answers a narrower question with the same rigor: **did anything that
merged since the last pass change what a user sees, and does the rest still
hold where it is cheapest to look?** It never advances the weekly baseline.

This is a procedure, not a scheduler: run it from whatever runs you daily, do
not create automations or post externally unless asked, and hand what you
found to a person.

## Inputs and hand-off

Each pass starts from the previous one. Keep, per pass, outside the repository:

- `TARGET`: the full `main` SHA the pass drove (today's `BASE` is yesterday's
  `TARGET`; on the first pass, the last completed weekly pass's `TARGET`).
- The run's `evidence/ledger.jsonl` (today's `--baseline`).
- The rotation position (which families got their turn, below).

Record the UTC time, model and spend budget, and the accounts and keys
available, as the weekly pass does. A pass that runs out of time says so and
names what it did not reach; it never rolls the remainder into tomorrow
silently.

## Tiers, in order

Run the tiers in this order and stop when the budget runs out; each tier is
cheaper than the next and catches a different kind of rot.

1. **Static (minutes, no stack).** `control-openhands map check`,
   `map coverage` and `map testids`. A `Source:` path that is gone, an `E2E:`
   spec that moved, or a test id the map drives that no literal in `src/`
   accounts for is drift found before any browser opens. Fix it in the map
   with a live drive later today, or report it. Zero unresolved test ids is
   the normal state; treat a new one as a rename until a drive says otherwise.
2. **Changed (the core).** `control-openhands map affected --base $BASE
   --target $TARGET`. Launch, doctor, then drive **every** sub-feature the
   listed families map for the changed paths, on each entry point the family
   lists, at desktop and phone viewports for UI changes. Widen `shared` paths
   to their consumers (an API client or store change touches every page that
   reads it; pick the families whose `Source:` lines name those consumers, not
   one convenient screen). An `unmapped` path under `src/` is a map gap: a new
   surface to map per [mapping.md](mapping.md), or a `Source:` line to extend.
   Resolve each merged commit in `BASE..TARGET` to its PR and keep intent
   (`documented`, `undocumented`, `contradictory`) separate from runtime
   results, as [maintenance.md](maintenance.md) step 2 describes.
3. **Smoke (every family, cheaply).** For each family not already driven in
   tier 2: open its first entry point, run the family's `errors --app-only`
   sweep, and drive the one bullet that proves the page's main state (a list
   renders, a form opens). One `evidence add` row per family, under the
   sub-feature ID that bullet names. This catches a page that stopped loading
   without re-proving every row.
4. **Rotation (depth over the week).** Drive the full recipe of about a
   seventh of the families (four each day, in ID order, continuing from
   yesterday's position), so every family gets a full live pass once a week
   between weekly maintenance passes. Model-backed bullets run only inside the
   LLM budget; without a key they are `blocked` with the missing prerequisite.

The `E2E:` specs that `map affected` lists are a cheap signal before tier 2
(run them with the suite's own command when the machine can; their green is
an input, never a pass in the ledger), and a selector reference while driving.

## Reading the result

Render `control-openhands evidence report --baseline <yesterday's ledger>`.
The "Changes since baseline" section is the daily verdict:

- **Newly failing** on a family that tier 2 drove for a changed path: an
  introduced-in-range candidate. Re-drive the same minimal recipe on `BASE`
  (a second `launch --new` on a worktree at `BASE`) before calling it a
  regression; otherwise classify it as reproduced-on-both, environment or
  harness gap, or origin-unconfirmed (maintenance.md step 4).
- **Newly failing** elsewhere, or **newly passing** without a PR that explains
  it: an undocumented change; cite the commit and ask.
- **Not checked this run** rows are the rotation moving on, not regressions;
  the family table shows which families got depth today.

Triage as the skill does: map drift (fix, with live evidence), harness gap
(extend the CLI, re-drive), product defect (keep the evidence, search issues,
file in the owning repository, keep it out of the map PR), blocked (name the
prerequisite and the route). At most one PR of proven map and CLI corrections
per pass, from current `main`; never edit product code in a pass.

## Several agents

Shard by family, never by bullet: each agent gets whole families, its own
`launch --new`, its own `OH_VERIFY_RUN`, and writes only its own ledger. One
coordinator merges the ledgers' reports and owns the hand-off. A result that
names no run, revision or entry point does not count; a gap is not a pass.

## What the daily pass is not

- Not the weekly pass: it does not re-prove every bullet or move the accepted
  baseline. When tier 2 finds that most of the map is affected (a shared
  component or style change), say so and run the weekly procedure instead.
- Not CI: a green Playwright run or a passing `map check` is an input to a
  verdict, never the verdict.
- Not a bug-fix lane: product defects are filed, not fixed, inside the pass.
