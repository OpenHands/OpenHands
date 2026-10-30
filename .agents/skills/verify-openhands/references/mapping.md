# Creating and extending the feature map

The map is written for the next agent, read cold and mid-task by someone who has
never seen the app. Every entry answers, from the user's point of view: what the
feature is, how a user reaches it, how to drive it with `control-openhands`, and
what observable end state proves it works. A recipe that was never executed is a
draft, not an entry.

## 0. Build or check the lever first

Before mapping anything, prove the CLI can carry one feature end to end:
`launch`, `doctor`, drive one mapped recipe, `browser screenshot`, `evidence add`,
`stop`, and confirm the evidence file still exists after `stop`. If any step
fails, fix the CLI or report the environment blocker before writing entries:
a map written against a broken harness teaches wrong steps.

While mapping, every time you reach for an ad-hoc script, a raw Playwright
snippet or a manual curl to drive a user path, stop and add a verb or flag if missing to
`scripts/control-openhands.mjs` instead (documented in its `--help`). The CLI is
what makes the map rerunnable. But if it exists already, use it instead.

## 1. Inventory user-facing surfaces

Work from what a user can touch, and require a concrete source path for each
candidate. Sources, roughly in order of yield:

1. **Routes**: `control-openhands map routes` lists `src/routes.ts`. Every route
   is reachable by URL; most are also reachable from navigation. Both entry
   points count separately.
2. **Navigation and chrome**: on each page run `control-openhands browser testids`
   and `browser snapshot`. Record the sidebar, command menu (`command-menu-trigger`,
   `Control+k`/`Meta+k`), backend selector, settings navigation, onboarding
   checklist, and every button that opens a dialog, drawer, popover or menu.
3. **Per-item menus and dialogs**: row "more" menus, context menus
   (`src/components/features/context-menu`), confirmation modals, edit forms.
   Modals and panels are features even without a route.
4. **The conversation surface**: composer controls (attachments, profile and agent
   pickers, slash/skill commands, dictation, stop/play), message rendering
   (markdown, code, images, tool visualizers), the conversation panel tabs, git
   controls, and anything the agent can drive through client tools
   (`canvas_ui_control`).
5. **Keyboard shortcuts**: grep `keydown`, `useHotkeys`, `metaKey`, `ctrlKey` in
   `src/`. A shortcut is an entry point. When it reads `event.repeat` (a held
   key must not fire twice) or needs a modifier held across a click, drive it
   with `browser key down <Key> [--repeat N]` and `browser key up <Key>`
   (`browser press` sends one keydown and keyup).
6. **Backends and modes**: local vs Cloud backend, public vs local auth, ACP
   agents, partial stacks (`agent-canvas --help`), Electron, Docker, and the
   embeddable library (`src/index.ts` / `build:lib`).
7. **Docs and tests**: `README.md`, `docs/*.md` describe
   intended behavior. The Playwright suites under `tests/e2e/` (mock-LLM, live,
   bind-policy, live-acp) assert behaviors a user sees too: read them for
   selectors and for behaviors the map lacks, then map those behaviors as
   sub-features with their own recipes. The map never cites a spec: the
   suites run differently and prove a commit, not this run.
8. **Strings**: `src/i18n/translation.json` keys reveal user-visible states
   (empty, error, disabled, confirmation) that a happy path never shows.

`control-openhands map coverage` lists routes and `src/components/features/*`
directories no entry references yet. Drive it to zero, or list each exclusion
with its reason in the index's "Not mapped" section.

## 2. Group into families

One file per user-facing job (for example "Secrets", "LLM profiles",
"Conversation composer"), not per component. Aim for files an agent can verify
in one sitting: usually 4 to 15 sub-features, up to about 30 for a dense page;
split beyond that (for example dashboard list vs. per-item actions). Name files
`Fnn-short-slug.md` and give the family the ID `Fnn`. Keep IDs stable forever:
when a label or route moves, update the text, not the ID; when a feature is
removed, mark it `retired` with the authorizing PR instead of deleting the row.

A **sub-feature** is one observable behavior with its own pass/fail, such as
"saving persists after reload" or "empty state explains what to do". IDs look
like `F14.secret-create`: lowercase, hyphenated, unique across the map. Settle
the IDs before taking evidence, and write the `## Sub-features` list into the
file first: screenshots and ledger rows are filed under them (`evidence add`
and `map check` reject unknown IDs), and renaming later means re-shooting.

An entry point shared by several families (a launcher card, a settings link)
is mapped once, by the family that owns the destination; the others reference
that ID instead of re-mapping it.

## 3. Write each entry

Describe current behavior and how to verify it. Once a fix is proven, keep
the corrected expectations and normal recipe steps, and remove the fix's
PR or closed-issue references, verification dates or device lists, old UI
wording, and descriptions of checkouts without the fix. Keep that history
in the maintenance PR and evidence ledger. Retain links for known open
bugs, authorizing PRs for retired IDs, and version or environment
distinctions that still affect a supported run.

Each file starts with an H1 title, one paragraph describing the user-visible
behavior, and one `Source:` line with the main implementation paths (for drift
checks and `map affected`; keep other implementation detail out). Every path
must exist (`map check` verifies it); a directory ends with `/`, names in
parentheses after a directory are relative to it, and `name-*.tsx` matches by
prefix. For a family spread over many files, list the route modules and
top-level component directories, not every file. Then exactly four H2
sections, in this order:

1. `## Sub-features`: one bullet per ID: `` - `F14.create`: add a dummy secret; it persists after reload. ``
2. `## How to get to it (user POV)`: every entry point, in user language: the
   sidebar link, the direct URL, the command-menu entry, the keyboard shortcut,
   the row menu. Note which entry point each recipe exercises.
3. `## Driving it with control-openhands`: starts with `Preconditions:` (run
   state, profile, fixtures, viewport) as bullets, then labeled bullets that pair
   one user action with exact commands and the observable result:

   ```markdown
   - **Create (`F14.create`).** Add a secret from the list page.
     Run `control-openhands browser click 'testid=add-secret-button'`,
     `control-openhands browser fill 'testid=add-secret-form >> testid=name-input' QA_TMP`,
     `control-openhands browser fill 'testid=add-secret-form >> testid=value-input' dummy-value`,
     `control-openhands browser click 'testid=add-secret-form >> testid=submit-button'`,
     then `control-openhands browser reload` and
     `control-openhands browser count 'testid=secret-item >> has-text=QA_TMP'`.
     The count is `1` after reload.
   ```

   Include the read-only second view for every mutation (reload, reopen, or a
   GET through `control-openhands api GET ...`), the empty/error/disabled states
   that matter, and a `browser screenshot --feature <ID> --name <label>` where the
   proof is visual. Include phone-viewport checks (`browser viewport phone`) for
   layouts that change on small screens.
4. `## Gotchas`: traps that waste or invalidate a run: debounces, focus traps,
   items that only appear after the agent runs, features hidden behind a
   prerequisite, and known open issues (link them; they are repro candidates, not
   exemptions). The map should know every open bug. Record one on the bullet
   that reproduces it, next to the expected result, as `Known failure
   (reproduced <date>, #<issue>): <what happens>`; list it here as
   `Known issue #<issue>: ...` only when no recipe can drive it. The completed
   [existing-bug backfill](backfills/2026-10-12-open-bugs.md) accounts for its frozen
   inventory; the [main refresh](backfills/2026-10-16-main-refresh.md) records later
   reconciliation. [bug-backfill.md](bug-backfill.md) defines ongoing intake and outcome
   labels. Absence of `ready-for-dev` is not an exclusion.
   Use full repository-qualified links for SDK and automation issues, so their
   numbers cannot be mistaken for Canvas issue numbers.

   Make the note executable when a recipe can drive it: indent a fenced block
   under it, opened with `` ```sh known-bug <ID> [name] `` (the name tells
   entry points or viewports apart, such as `desktop` and `phone`). Its
   commands set their own viewport and starting page and assert the *correct*
   behavior; exactly one command, the one that fails because of the bug, is a
   `control-openhands assert --field <path> --equals <correct> --bug-shows
   <what the bug shows> -- <command>` ending in `# bug` (a long one may
   continue over lines ending in `\`). When the bug's value varies, or
   needs more than one field to pin down, name it with `--bug 'PATH OP
   [VALUE] && ...'` instead (see the known-bug contract below). Arrange steps go
   before it and must be able to fail on their own (assert the starting value
   too), so a broken setup is never read as the bug. Keep a `browser
   screenshot` before the assertion where the symptom is visual.
   `control-openhands bug run <ID>` then reports `xfail` while the bug lasts,
   `xpass` once it stops, and `fail` for anything else; `map check` rejects a
   block without exactly one such command. F25's `F25.selector-dropdown` note is
   the worked example. When a fix is proven, the PR that drops the note turns
   the block's commands into the bullet's normal steps, so the same assertion
   guards the corrected behavior.

Treat commands as literal: quote selectors, keep names unique to the run (prefix
fixtures with `QA_`), and restore shared state after mutations (delete the secret,
deactivate the profile) unless the next recipe depends on it, in which case say so.
Values that differ per run (conversation and automation ids) are written as
`<id>` placeholders next to the command that prints them.

The driving bullets must run top to bottom on one stack, because the next agent
will run them that way:

- Put every arrange step in `Preconditions:`, in execution order. A
  precondition with a short life (a conversation that is running, a toast)
  says when to create it: "right before the Stop bullet".
- A bullet never depends on state that a later bullet creates, and never
  deletes a fixture that a later bullet needs.
- Write environment-dependent results as conditionals ("if the registry is
  reachable, `Up to date`; otherwise ...") and check them with `browser
  network`, not from memory.
- A bullet that leaves an unsaved edit ends with `browser reload`, and a
  bullet that changes a setting a later editor inherits (an LLM temperature,
  a theme) restores it or says so: later bullets inherit both.
- After picking an option and saving, read the stored value back; a failed
  option click otherwise goes unnoticed until a later step.
- Assert deterministic text (what the UI sent, a file the agent wrote, a tool
  observation), not the model's wording.
- Secret and credential checks assert the tool's output, not the model's
  reply: models refuse to echo secrets. Compare inside the command
  (`test "$QA_SECRET" = expected && echo match`) and read the observation
  with `conversation events <id> --kinds ObservationEvent`.

## 4. Prove every recipe live

Execute every command you wrote, in order, on a fresh `launch` (or a run you
have doctored since its last surprise). Before you hand the file over, run the
whole file once more from the top, exactly as written; a multi-minute,
model-backed step whose commands did not change since it last passed may be
skipped in that second pass if you say so in the report, and a family that
stops or restarts services may use a fresh stack for it.

Families about the launcher or the outside world (partial stacks, Docker,
desktop, a git remote) need some plain shell commands. Keep them few, give
every hand-started process its own process group and stop it by that group
(never by name pattern), and prefer adding a CLI verb once a command repeats.
Builds that rewrite tracked or generated files (`npm run build:lib`) run in a
copy of the checkout, not in the one other runs serve. For each sub-feature record
`control-openhands evidence add --feature <ID> --result pass|fail|blocked|not-run`
with the entry point, expected and actual result and artifacts.

- A recipe that fails because the instructions are wrong: fix the instructions
  and re-drive (map drift).
- A recipe the CLI cannot express: check `control-openhands --help` and
  `control-openhands help browser` first (backdrops, modifier clicks, stopped
  services, stale keys, first run and sound all have verbs), then extend the
  CLI and re-drive (harness gap).
- A recipe that fails because the app is broken: keep the expected result,
  record `fail` with evidence and report the product bug separately.
- A recipe that needs something the run cannot have (a Cloud account, a
  GitHub token, a Docker daemon, macOS): mark it `blocked`, name the prerequisite
  in `Preconditions:` and keep the recipe as far as it can be written.

`blocked` is the most common wrong verdict. Before you write it, try the
cheaper routes that earlier mappers missed:

- Another family may already create the precondition through the UI (F22's
  Daily news digest makes a script automation with no account). Search the
  other files for it.
- A bundled fixture often exists upstream too: a public, read-only Git source
  such as `https://github.com/OpenHands/OpenHands/tree/main/src/fixtures/...`
  works through the Agent Server's network access.
- When a path is "blocked because every X needs Y", list every X by its real
  attributes first. For example, sort the template cards by setup form and
  integration type: cards whose integrations all need external setup skip the
  install queue entirely.
- Check `<verb> --help` again; a `fixture` or `browser` verb may exist now.

## 5. Check the map

```sh
control-openhands map check --file Fnn-name.md   # one entry while others are being written
control-openhands map check      # whole map: structure, unique IDs, index links, valid commands
control-openhands map coverage   # unmapped routes and feature component dirs
control-openhands map ids        # every ID with its file, for the index tables
```

Whoever owns the index ([feature-map/README.md](feature-map/README.md)) adds the
new file to the families table with its entry points, prerequisites and ID
count, and lists "Not mapped" reasons. `control-openhands map check` fails when
a count is stale; `map check --fix-counts` rewrites the counts and the total. When several agents map families in
parallel, each edits only its own file and checks it with `--file`; one
coordinator updates the index afterwards.
Then hand the evidence ledger (`control-openhands evidence report`) to whoever
asked, with fail/blocked rows first.

## The known-bug contract

Keep these rules in step with the sibling verification skills
(verify-agent-server in OpenHands/software-agent-sdk, verify-automation in
OpenHands/automation), so a proof means the same thing in each. The code stays
separate; only the syntax for naming the bug differs, noted below.

- The recipe asserts the correct behavior. The command that fails because of
  the bug ends its line with `# bug` and names the bug's own symptom: the
  value or error it produces, not just any wrong value. Here that command is
  always a `control-openhands assert`, which carries the symptom as
  `--bug-shows V` or `--bug PREDICATE`; the sibling skills write
  `# bug: PREDICATE` on the marked line instead.
- Three outcomes. `xfail`: the marked command failed and its symptom held (the
  documented bug reproduced). `xpass`: every command passed, so the correct
  behavior was observed; `bug run` exits 1 so the note gets reviewed and the
  marker removed. `fail`: anything else, including a different value at the
  marked command, an arrange step that breaks first, a timeout, an unhealthy
  stack before or after, a failure after the marked command, or a failing line
  the runner could not locate.
- The predicate is `PATH OP [VALUE]` clauses joined by `&&`, matched against
  the JSON the asserted command printed: `--bug 'status eq 500 &&
  body.error contains "has no attribute"'`. OP is `eq`, `ne`, `contains`,
  `not-contains`, `matches` (a regex), `exists` or `missing`. `--bug-shows V`
  is `--bug '<field> eq V'`. Prefer `contains` or `matches` on the stable part
  of an error message over pinning its whole text.
- `xpass` must mean the correct value was observed, so the marked command
  asserts positively. Negative shapes (`! cmd`, `if cmd; then false; fi`,
  `test !`) cannot be marked: the marked command must be an `assert`, and
  `map check` rejects anything else.
- `map check` rejects a missing or malformed predicate.
- The observation is the marked command's own. Every assertion is logged with
  the recipe line that ran it and the first failure is kept, so a later step
  or an `EXIT` trap's cleanup cannot replace it. A marked command continued
  over several lines counts as one, from its first line to the `# bug` line.
- A fix is proven with the baseline reproducing (`xfail`) twice and the fix
  passing (`xpass`) twice, with the same recipe, viewport and backend:
  `control-openhands bug proof` computes that verdict (`verified`,
  `not-fixed` or `inconclusive`).
