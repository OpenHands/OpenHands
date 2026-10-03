# F05 — Runtime settings and secrets

Persist configuration and manage secrets without confusing form rendering with
successful storage. Source: [settings routes](../../../../src/routes.ts),
[secret form](../../../../src/components/features/settings/secrets-settings/secret-form.tsx),
[secret list](../../../../src/routes/secrets-settings.tsx).

## Sub-features

- `F05.settings`: settings landing, Application, Condenser, Agent Context and Verification.
- `F05.modes`: available Basic/Advanced/All views, disabled/save states and help text.
- `F05.secret-create`: add a dummy secret and prove it persists after reload.
- `F05.secret-edit`: edit description/name; blank value preserves the existing secret.
- `F05.secret-delete`: cancel deletion, then confirm and verify removal after reload.
- `F05.secret-forward`: a newly created conversation receives only the intended secret.

## How to get to it (user POV)

Use Settings navigation and direct links `/settings`, `/settings/app`,
`/settings/condenser`, `/settings/agent-context`, `/settings/verification` and
`/settings/secrets`. Test route entry as well as transitions between settings pages.

## Driving it with Playwright

Preconditions: isolated settings/backend; dummy secret values only. Secret-forwarding
requires a real authorized conversation; never ask an agent to print a real key.

- **Settings:** visit each route, expand available field modes and inspect labels,
  help and save state. Change one safe setting, save, reload and verify the stored
  choice, then restore it. Check long values and phone-width scrolling. Record
  which fields were mutated rather than claiming all configuration permutations.
- **Create/delete:** run [the smoke helper](../scripts/smoke.mjs). It uses
  `add-secret-button`, scoped `add-secret-form` → `name-input` / `value-input` /
  `description-input` → `submit-button`; checks the named `secret-item` after reload;
  then `delete-secret-button` → `confirmation-modal` → `confirm-button` and reload.
- **Edit:** scope `edit-secret-button` to the run-owned row; change its description
  but leave `value-input` blank. Save, reopen and verify the description. Prove
  retained value only in an isolated consumer check using the known dummy value.
- **Forward:** create a fresh bounded conversation and assert the dummy secret is
  available to its intended tool/environment without echoing other environment
  variables. A row in settings alone does not prove forwarding.

## Gotchas

The secret value editor is a plaintext textarea, not a password field. Use no real
service values here for screenshots. An edit form intentionally leaves the value
blank; do not interpret that as data loss. Settings may differ by backend/schema;
unsupported fields need a capability explanation, not guessed substitutes.
