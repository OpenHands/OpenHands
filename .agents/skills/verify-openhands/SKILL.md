---
name: verify-openhands
description: >
  This skill should be used to "verify OpenHands features", "drive Agent Canvas",
  "check a Canvas UI change", or collect real desktop/mobile evidence in this
  repository. Provides the feature map, isolated launch, doctor, browser recipes,
  and cleanup. For a week-over-week audit, use verify-openhands-weekly as well.
---

# Verify OpenHands through the real app

Prove the user-visible result, not merely that a route loaded or CI passed.
This is a repository-local companion to the public `qa-changes` skill, not a
replacement for it. Use `qa-changes` for the verdict on an individual PR and this
map for the actual Canvas paths and traps. Neither skill authorizes external
writes, paid model calls, or product fixes beyond the user's request.

## Select the surface and scope

Read [the feature index](features/README.md), then the relevant feature files.
Record the exact checkout SHA and selected entry points before starting. Use
[the weekly skill](../verify-openhands-weekly/SKILL.md) when comparing main with
last week's evidence or checking merged PR intent.

The primary surface is the standalone web app with a real local Agent Server and
real automation service. Cloud, ACP agents, browser tools, third-party services,
Electron, Docker, and embedded-library behavior have separate prerequisites.
A passing local route is not a pass for another backend or platform.

## Launch

Follow [the runtime recipe](references/runtime.md) from the checkout being
verified. It uses the existing production CLI, pinned dependencies from
`config/defaults.json`, fresh state, and authenticated public mode to avoid
publishing a launcher-injected session key. Do not install the latest published
Canvas and call that a test of this checkout.

Use an isolated worktree for each revision, a fresh browser context per backend,
and distinct ports/state for baseline and target. Drive one instance serially.
Parallel source readers are useful; concurrent agents clicking the same page are
not. Do not reuse a user's running session or copy their profiles/state.

## Doctor

Before driving, and after a surprising failure, use the runtime recipe to check:

1. The expected launcher and owned child processes are alive; their actual ports
   match this run's log, not an unrelated listener.
2. The build and checkout match the recorded target SHA.
3. Unauthenticated API requests are rejected, and the private session key reaches
   the expected Agent Server and automation versions.
4. The UI reaches that backend without an auth loop or loading overlay.

A successful HTML response alone is not readiness. If an API returns the SPA,
JSON parsing fails, or a backend is unavailable, stop driving that surface. Record
an environment blocker. If the process is healthy but the UI is wedged, preserve
failure evidence, reset to a known state, and retry once. Never silently switch
to mocked responses or remove a failing assertion to manufacture a pass.

## Drive

Run the credential-free starter recipe after doctor:

```sh
node .agents/skills/verify-openhands/scripts/smoke.mjs
```

It uses the `QA_RUN` / `QA_BASE_URL` exported by the runtime recipe and optionally
`QA_BROWSER_EXECUTABLE`. It authenticates through the UI, creates a dummy secret,
reloads to prove persistence, deletes it through the UI, and captures desktop and
mobile GitHub MCP credential fields without installing the integration. Its
passing result covers only the named checks, not all features in the map.

For further driving, use installed Playwright (`@playwright/test`) or the available
browser tools. Follow the feature's actual user entry point; prefer scoped ARIA
roles, labels, and `data-testid` handles over coordinates or text scraped from
hidden drawers. Reinspect current source when a handle drifts. The existing
`tests/e2e/mock-llm/` specs are useful selector references, **not permission to copy
their API interception or mock LLM into a live proof**.

For each selected sub-feature:

- Capture the initial state and action, then assert the named observable result.
- For mutations, verify through a second read or reload, not just a toast.
- Check empty, populated, loading/error, disabled, and confirmation states where
  relevant to the change. Treat each changed entry point separately.
- Exercise desktop 1440×1000 and phone 390×844 for UI changes; add 320px width for
  responsive fixes. Inspect screenshots, dialogs' bounding boxes, overflow,
  keyboard/focus/close behavior, and browser exceptions. A visible destination
  panel does not cancel a lifecycle exception during the transition.
- Scope assertions and request authentication to the intended backend. Never put
  a backend key in global headers sent to third-party origins.

Use harmless dummy values for appearance-only credential checks. Keep native
password masking: do not cover the input background/border with a screenshot
mask. If a real credential is necessary for an authorized integration, keep it
concealed and inspect the entire frame before publication. Plaintext secret forms,
exports, system prompts, tool definitions, traces, and storage can still leak data.

## Evidence

Keep raw logs, browser state, API payloads, and traces under `$QA_RUN/private`.
Keep candidate screenshots and the feature/action/result ledger under
`$QA_RUN/evidence`, which is **not automatically public**. Review before publishing
only the selected safe artifacts to an authorized destination. Never serve the
run root or copy a secret-bearing file into a served or committed directory.

Use [the report contract](references/report.md). Every claim names a feature ID,
entry point, revision, backend/capabilities, viewport, command/action, expected
result, actual result, and evidence path. Preserve failures and coverage gaps.
Record console exception class/count separately from successful visual rendering.
CI results and old screenshots are supporting context, not this run's proof.

## Cleanup

Delete only fixtures created by this run, through the UI where that is the path
under test. Disable schedules and remove test integrations before ending a live
session. Stop only the launcher/process tree this run owns, using the runtime
recipe; verify its ports close. Do not kill by a generic process name.

Delete temporary private state only after teardown and after checking evidence
exists. Keep reviewed proof artifacts and the run ledger after cleanup, including
on failed iterations. A run whose cleanup deletes its proof is not complete.

## Improve the map, not the product under test

Document a broken selector as a harness gap and a wrong recipe as map drift.
Re-drive corrected instructions before claiming they work. Report product bugs
separately; never rewrite the expected result to bless broken behavior. Historical
issues in the map are repro candidates, not permanent exemptions or claims that
the issue is still open.

See [adaptation notes](references/adaptation.md) for pstack provenance and the
OpenHands-specific decisions behind this workflow.
