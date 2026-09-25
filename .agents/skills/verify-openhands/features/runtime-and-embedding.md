# F12 — Runtime, installer and embedded-library variants

Keep the primary web proof distinct from other ways users run or embed Canvas.
Source: [package exports/scripts](../../../../package.json),
[production CLI](../../../../bin/agent-canvas.mjs),
[development/embedding guide](../../../../docs/DEVELOPMENT.md),
[release matrix](../../../../docs/TESTING_MATRIX.md).

## Sub-features

- `F12.source-web`: current-checkout static/full stack; authenticated access and cleanup.
- `F12.partial`: frontend-only/backend-only routing and unavailable-service feedback.
- `F12.installers`: published npm and Docker install × supported OS × real agent matrix.
- `F12.desktop`: native Electron launch, shell/editor handoffs and window behavior.
- `F12.embedding`: exported providers/root, scoped styles, auth and host integration.

## How to get to it (user POV)

Launch the actual chosen distribution; do not test one then certify another.
For embedding, mount `AgentServerUIProviders` / `AgentServerUIRoot` in a separate
host app as described in the development guide, with a real owned backend.

## Driving it with Playwright and platform tools

Preconditions: the target distribution/platform and its explicit resource/auth
permissions. The weekly browser default is F12.source-web; record every other
variant as covered, blocked or outside the agreed scope, never implicitly passed.

- **Source web:** execute [the runtime recipe](../references/runtime.md), doctor and
  smoke; verify build revision, backend versions, auth rejection, actual UI mutation
  persistence, then the owned process/port state after shutdown.
- **Partial stack:** follow the production CLI's `--frontend-only` / `--backend-only`
  modes in separate owned instances. Check documented available routes and explicit
  unavailable-service feedback; a generic HTML fallback is not API success.
- **Installers/native:** follow each cell of the existing testing matrix with the
  actual npm/Docker/Electron artifact on its OS. Use a real agent for the required
  conversation; capture native windows if claiming desktop behavior.
- **Embedding:** verify the host's typography/layout remain unchanged outside the
  Canvas scope, then exercise a real Canvas action inside it. Inspect documented
  theme overrides, authentication and host telemetry consent boundaries. Compare
  a host element outside Canvas before/after mounting.

## Gotchas

A Chromium screenshot is not native-Electron or cross-browser certification.
Do not install platform tooling or launch containers without the necessary
permissions. `build:lib` success cannot prove CSS isolation or host behavior; real
host-app evidence is required. Keep distribution version separate from main SHA.
