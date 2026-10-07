---
name: verify-openhands
description: >
  This skill should be used to "verify OpenHands features", "test the Canvas UI
  like a user", "drive Agent Canvas", "check a UI change in the real app",
  "create or update the feature map", "run the daily pass" or "run the weekly
  feature audit". Ships control-openhands (launch, doctor, browser, LLM
  profiles, conversations, evidence, map tooling, cleanup) and the maintained
  map of every user-facing feature.
triggers:
- /verify-openhands
- feature map
- verify feature
---

# Verify OpenHands through the real app

Prove what a user sees, not that a route rendered or CI passed. Three parts work
together:

1. **`control-openhands`** ([scripts/](scripts/)) is the lever. It launches this
   checkout as an isolated real stack (Agent Server, automation, static frontend,
   ingress), keeps one browser alive between commands, and turns every user step
   into a command you can rerun.
2. **The feature map** ([references/feature-map/](references/feature-map/README.md))
   lists every user-facing behavior with stable IDs, user entry points, exact
   `control-openhands` recipes, observable results and gotchas.
3. **Evidence**: screenshots, ARIA snapshots and a pass/fail/blocked/not-run
   ledger that survive cleanup.

Neither this skill nor the map authorizes external writes, paid models beyond
the budget you were given, or product fixes the user did not ask for.

## The CLI comes first

Put it on `PATH` and read its help before anything else:

```sh
export PATH="$PWD/.agents/skills/verify-openhands/scripts:$PATH"
control-openhands --help                # then: control-openhands <command> --help
```

Requirements: Node >=24 (the launcher's engine), npm dependencies installed
(`npm ci --ignore-scripts`), `uv`/`uvx`, and a Chromium Playwright can launch
(set `CONTROL_OPENHANDS_BROWSER=/path/to/chrome` when the pinned browser is not
installed; `launch` then reports the running stack and `browser start` picks it
up). Every command prints one JSON object; exit 0 ok, 1 action failed, 2 usage,
3 environment.

If a user path cannot be driven with the CLI, that is a **harness gap**: extend
`scripts/control-openhands.mjs` (keep it executable, document the verb in
`--help` and below), prove the new verb live, then write the recipe. Never work
around a gap with an untracked one-off script the next agent cannot rerun.
Other agents on the machine may be running the same script: edit a copy, run
`node --check` and the tests on it, then move it into place in one step.

## Launch → doctor → drive → evidence → cleanup

```sh
export OH_VERIFY_RUN=$(control-openhands launch --new --print-run)   # builds if needed; isolated run
control-openhands doctor                       # read-only; must be ok before driving
control-openhands llm preset deepseek          # deepseek-flash (active) + deepseek-pro; key from $DEEPSEEK_API_KEY or --api-key-file
control-openhands onboard --skip               # consent + onboarding (walk it instead when F01 is under test)
control-openhands browser goto /settings/secrets
control-openhands browser testids              # discover handles on the current page
control-openhands browser click 'testid=add-secret-button'
control-openhands browser screenshot --feature F14.create --name form
control-openhands evidence add --feature F14.create --result pass --entry "Settings > Secrets > Add" \
  --expected "add form" --actual "add form" --artifact evidence/F14.create/form.png
control-openhands stop                         # stops only this run; evidence stays
```

- **Which run.** Commands use `--run DIR` (anywhere on the line), else `$OH_VERIFY_RUN`, else the only live
  run. With several live runs they refuse to guess, so export `OH_VERIFY_RUN` in
  every shell command when other agents share the machine: an agent that drives
  someone else's run corrupts both evidence ledgers. For the same reason never
  `pkill -f` or `killall` by pattern; `control-openhands stop` ends only your run.
- **Launch** refuses to start when less than about 2 GB of memory is free: each
  run holds an Agent Server, automation, a static frontend and Chromium (about
  1.5 GB together). Stop runs you are done with; several agents on one machine
  should each `launch --new`, export their own `OH_VERIFY_RUN`, and never touch
  another agent's run.
- **Launch** starts `bin/agent-canvas.mjs` from this checkout with a private
  `HOME`, state, session key and free port block, so it never touches a user's
  `~/.openhands` or another run. `launch --new` starts a second independent run
  (for a baseline, or to drive two backends); `--public` exercises the API-key
  login screen (`control-openhands login`). It still binds to 127.0.0.1; it only
  stops injecting the session key into the page. `--sdk-version`, `--sdk-ref` or
  `--sdk-path` (and the `--automation-*` equivalents) choose other backends;
  record them. Version variables exported in your shell are not forwarded.
  The block includes the VS Code editor port (`ports.vscode`) only when the
  checkout's launcher sets VS Code up (the port is reserved even if the
  agent-server has no editor binary and nothing listens there). Where that is opt-in (#17660), `--vscode`
  turns it on; where the editor is bundled (before #17660, or since #18048), it
  is always included and `--vscode` changes nothing.
- **Doctor** checks the launcher's process group, ports, served build revision,
  unauthenticated rejection, authenticated settings, Agent Server pin and the
  UI's minimum Agent Server version, automation health and a throwaway-tab UI
  probe. Run it first, after every surprising
  failure, and before blaming the product. A wedged UI on a healthy stack:
  capture evidence, `browser reload` or `goto /`, retry once.
- **Drive** through the real UI with `control-openhands browser ...`. Selectors are
  `testid=`, `role=button[name="Save"]`, `label=`, `text=`, chained with ` >> `.
  Use `browser testids` and `browser snapshot` to find handles; prefer scoped
  test IDs and accessible names over CSS. Failures return a hint and a
  screenshot path; read the screenshot before retrying. A click returns the URL
  from *before* any client-side navigation: use `click ... --expect-url '<regex>'`
  (or `browser wait-url`) after every navigating click, because many routes
  redirect (`/settings` → `/settings/agents`, `/customize` → `/mcp`).
- **Arrange, don't fake.** `llm`, `fixture` and `api ... --write` exist to set up
  preconditions (a configured profile, a git repo, a dummy secret). They never
  count as proof that the UI path works; the map says which steps are UI proof.
  Never intercept routes or add mock LLM responses to make a live check pass.
- **Essential pathways** are single commands so recipes can start from a known
  state: `onboard`, `llm preset|set`, `conversation start --prompt ... --wait`
  (add `--workspace qa-repo` to run it in a `fixture git-repo`),
  `conversation events <id>`, `workspace open`.
- **State control** reaches states a happy path never shows, without mocks:
  `browser reset` (fresh browser profile: first run again), `service stop
  automation|agent-server` (backend-down UI), `restart` (same state after a
  backend restart: persistence and reconnection) and `restart --rotate-key`
  (stale session key). `restart` keeps the browser on its page. While a service
  is stopped, a reload replaces the page with the backend-unavailable screen, so
  drive backend-down states on the page that was already loaded. `browser
  network`, `browser toasts` and `browser media` observe requests by origin,
  toasts and sound without changing anything. Check `--help` before calling a
  state unreachable: most "can't be driven" claims predate a verb.
- **Evidence** goes under `<run>/evidence/<feature-id>/` and the append-only
  ledger `<run>/evidence/ledger.jsonl`. Nothing is overwritten: a repeated
  screenshot name is saved as `<name>-2.png`, so cite the path the command
  prints; `evidence report` renders the table from
  [the report contract](references/report.md), fail and blocked rows first with
  their `--note` (a blocked row names its missing prerequisite there), and
  `--baseline <yesterday's run>` lists what changed since that ledger. Keys, logs, browser profile and
  downloads stay in `<run>/private/`. Evidence is not automatically public:
  review every image before publishing it. The CLI masks password fields in
  `snapshot`, `value` and `testids`; a screenshot of a visible key field is
  still a leak.
- **Cleanup** with `control-openhands stop` (add `--purge-private` to delete keys
  and state once you have checked the evidence). It only signals the process
  group it launched and verifies the ports closed. Delete run-owned fixtures
  through the UI when deletion is the path under test.

## LLM budget

Features that start a real agent with automation or debugging skills (for
example "Debug with OpenHands") can act on other fixtures by name: stop those
conversations when their recipe is done.

Use `deepseek-flash` for everything that needs a model; switch to `deepseek-pro`
only for checks that need a second profile or a stronger model. Keep prompts
small and confined to the run workspace. Without a key, run every credential-free
recipe and record model-dependent ones as `blocked` with the missing
prerequisite; never substitute a mock and call it a pass.

## Choose the job

- **Verify a change or a PR**: `control-openhands map affected --base <ref>`
  (or `--paths` with the PR's file list) maps the changed paths to the
  families whose `Source:` lines own them, and separates shared code to widen,
  `src/` paths no family owns (a map gap) and non-user-facing paths. Drive every entry point those features list at
  desktop and phone viewports, and report with
  [the report contract](references/report.md).
- **Run the daily pass**: follow [references/daily.md](references/daily.md):
  static checks (`map check`, `map coverage`, `map testids`), the changed
  families since the `Maintenance baseline` line of the map index
  (`control-openhands map baseline`; `map affected` starts there by default),
  a smoke row per family, and a date-derived rotation that gives every family
  a full live pass once a week. A pass that finished its changed families
  proposes `TARGET` as the next baseline with `map baseline --set "$TARGET"`
  in its PR; compare ledgers with `evidence report --baseline` when
  yesterday's is at hand.
- **Create or extend the map**: follow [references/mapping.md](references/mapping.md).
  It teaches how to discover features, write entries against the CLI and prove
  each one live. `control-openhands map coverage` measures what is still unmapped.
- **Maintain the map (periodic or weekly)**: follow
  [references/maintenance.md](references/maintenance.md): index hygiene, a source
  wave, one live pass over every feature, PR-intent reconciliation for the week's
  changes, and at most one PR of proven corrections.

## Triage what you find

- **Map drift**: the map describes something the app no longer does by design.
  Fix the entry with current source and live evidence.
- **Harness gap**: the app works but the CLI cannot drive it. Fix the CLI, re-drive.
- **Product bug**: the app is broken. Keep the evidence, search existing issues,
  report it separately (right repository: Canvas UI here, Agent Server/SDK in
  `OpenHands/software-agent-sdk`, scheduling/dispatch in `OpenHands/automation`).
  Never rewrite an expected result to bless broken behavior.
- **Blocked**: name the missing prerequisite (account, entitlement, OS, binary)
  and the route you attempted. Unreachable is never a pass.

See [references/adaptation.md](references/adaptation.md) for where these ideas
come from and what was deliberately left out.
