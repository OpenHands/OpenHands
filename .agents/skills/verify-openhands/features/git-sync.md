# F10 — Automation Git Sync

Connect automation definitions to a repository and inspect the real synchronization
result. Source: [Git Sync route](../../../../src/routes/automation-git-sync.tsx),
[automation UI](../../../../src/components/features/automations).

## Sub-features

- `F10.form`: unconfigured/disabled settings, prerequisite guidance and validation.
- `F10.sync`: authorized remote configuration, real sync status and persisted definitions.
- `F10.failure`: useful authentication/conflict/unreachable feedback without silent loss.

## How to get to it (user POV)

Use the automation dashboard's Git Sync entry or `/automations/git-sync` directly.
Inspect the form before connecting a repository.

## Driving it with Playwright

Preconditions: a healthy automation backend. F10.sync/failure require explicit
permission for a disposable remote/branch and credential scope; do not assume a
weekly UI audit authorizes remote writes or use the product repository as a fixture.

- **Form:** visit the route, inspect disabled/unconfigured controls, enter harmless
  invalid local form values and verify validation. Capture desktop/mobile without
  credentials. Record this as form-only coverage, not sync success.
- **Sync:** for an approved fixture, read the current form's direction/branch behavior
  and configure it through the UI. Capture the initiating action and resulting
  status; inspect the authorized remote branch and resulting automation definitions
  read-only to prove which data actually moved. Reload and verify persisted settings.
- **Failure/cleanup:** use a controlled unreachable fixture or a revoked test-only
  credential, not damage to a real repository. Check actionable failure status and
  retained definitions. Disable the run-owned connection and remove only explicitly
  approved fixture resources; preserve safe proof of the resulting remote state.

## Gotchas

Do not trust a “sync” or “dry-run” label to imply read-only behavior. Inspect the
actual direction and remote-ref effects before invoking it. Missing credentials
are a blocker for sync success, not justification to fabricate a remote response.
