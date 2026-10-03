# F01 — Entry, onboarding and backend selection

Reach the app, select an authenticated backend and workspace, and start from the
right context. Source anchors: [routes](../../../../src/routes.ts),
[backend forms](../../../../src/components/features/backends/backend-form-modal.tsx),
[onboarding](../../../../src/components/features/onboarding),
[backend spec](../../../../specs/backend-management.md).

## Sub-features

- `F01.auth`: first-run public-key entry, wrong-key rejection and returning-user access.
- `F01.onboarding`: backend → agent/provider → hello, with skip/cancel and consent.
- `F01.backends`: add/edit/remove/switch owned local backends; Cloud is F11.
- `F01.workspace`: folder picker, new/existing workspace and conversation grouping.
- `F01.navigation`: home, conversation list, sidebar collapse, command search and direct links.

## How to get to it (user POV)

Open `/` or `/conversations`; use the bottom backend selector, New Chat, the
workspace picker and Search commands. Reopen each supported link in a fresh tab;
do not infer direct-link behavior from client-side navigation alone.

## Driving it with Playwright

Preconditions: fresh run/browser for first-run checks; two owned servers with distinct
names/state for switching; an authorized provider for completing the hello step.

- **Auth:** the [smoke helper](../scripts/smoke.mjs) skips onboarding, fills
  `api-key-entry-name` / `api-key-entry-api-key`, and submits `api-key-entry-submit`.
  Separately try a dummy wrong key: remain on the entry screen with an inline error.
  Use a private returning context to prove reload preserves authorized access.
- **Onboard:** follow `onboarding-step-check-backend`, `onboarding-step-choose-agent`,
  `onboarding-step-setup-llm`, then `onboarding-step-say-hello`; assert a real reply,
  not just dismissal. Choose telemetry consent explicitly and verify it persists.
- **Backends:** open `backend-selector` → `add-backend-menu-item` →
  `add-backend-option-agent-server`; fill `add-backend-name`, `add-backend-host`,
  `add-backend-api-key`, submit `add-backend-submit`. Switch and verify conversation
  and settings data belong to the selected server; exercise edit/remove/cancel too.
- **Workspace/navigation:** use `open-workspace-button` → `workspace-dropdown` /
  `add-workspaces-button`; select a disposable folder using `folder-browser-use`.
  Verify its path, launch a conversation, then find it under that workspace. Use
  command search to reach a settings page and inspect narrow-screen navigation.

## Gotchas

Skipping onboarding is not onboarding completion. An offscreen drawer can have
mounted controls; test intended open state, not DOM presence alone. Never share a
browser's backend registry between baseline and target or switch into a real user's
backend. Auth screenshots/storage may expose keys; capture after successful entry.
