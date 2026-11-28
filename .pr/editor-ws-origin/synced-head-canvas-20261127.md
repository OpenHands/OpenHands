# Synced-head Canvas capture (2026-11-27)

Same-setup before/after for the merged head after `#20061`, re-run with the
Docker engine available. This records the packaged run the review guide
asks for: stock failure and synced-checkout success from the same image,
state, and projects mount, with the mounted script hashes and the
WebSocket result.

## Setup

- Image: `ghcr.io/openhands/agent-canvas:1.43.0`
  (local image ID `sha256:bcdbb3fb9610f14eda67f39fc609f162ad8c3c28697d6ff3f4adae55c15c3f1d`).
- State volume and projects mount identical for both runs; the only
  difference is the three-file script overlay.
- Stock run published `127.0.0.1:18005` with the image's own scripts.
  Synced run published `127.0.0.1:18006` with the candidate head's
  `scripts/static-server.mjs`, `scripts/proxy-utils.mjs`, and
  `scripts/bind-host.mjs` mounted read-only under `/opt/agent-canvas/`.
- The containers were run one at a time (the stock instance was removed
  before the synced boot), matching the README procedure of stopping the
  first instance before reusing the shared state.

## Mounted script identity (in-container `sha256sum`)

| file | stock | synced overlay |
| --- | --- | --- |
| `/opt/agent-canvas/static-server.mjs` | `4a09c7fc9aa243256fac92ad9787bd0ae0ef788501d92f1d2dbc80ad34473a67` | `da027da4678217604a1250a00bd160c56fbac24711a8d017a1f8170b69d6f805` |
| `/opt/agent-canvas/proxy-utils.mjs` | `e064a4b8db7ec1e6344bf24a2d8490785e44de3e28269a0b6817a0c4eb71363c` | `d1b06403ceaa146f3ed57fb42822d804b581a2d7915a9c8f2b057e2c6cbccf00` |
| `/opt/agent-canvas/bind-host.mjs` | `24d624d66fc0823f7391951ca0b3d62d0ed38e22617b1b10ce2b4222d88521cf` | `170cfdcf07d95775b083964ac679e1186f0843089f8a0f7c2927dc19f9cec114` |

The synced `static-server.mjs` hash equals the candidate head's blob. The
stock scripts contain no `isForeignLoopbackRequest` references; the
synced one contains three (the `#20061` entry guard). Both stacks booted
and answered `/canvas` with HTTP 200 and the agent-server `/alive` probe
with 200.

## Procedure (per run)

1. Open `http://127.0.0.1:<port>/canvas`, open the existing conversation,
   and open the workspace panel.
2. The panel's VS Code control resolves the editor URL through
   `/api/conversations/<id>/vscode/url` and opens the same-origin
   `/vscode/?tkn=<redacted>&folder=/projects/demo` route.
3. The workbench page consumes the token (session cookie exchange), then
   connects its remote workspace over WebSocket through the static
   server's `/vscode` route.

## Result

- Stock (`18005`, image's own scripts): the workbench shell loads and
  then reports a modal error — "An unexpected error occurred that
  requires a reload of this page. The workbench failed to connect to the
  server (Error: WebSocket close with status code 1006)". The remote
  workspace never connects. This is the reported `#19945` symptom.
- Synced overlay (`18006`, candidate head scripts): the workbench
  connects with no error dialog; the Explorer lists the workspace
  folder and the status bar shows the live repository branch, i.e. the
  remote workspace session is established through the preserved client
  identity.
- WebSocket result: close code 1006 on stock versus a connected
  workbench on the synced overlay.
- Auth check intact on both: a raw upgrade probe without the session
  cookie is refused with HTTP 401 through the synced static server, so
  the identity forwarding does not bypass the editor proxy's
  authentication.

## Screenshots

Round-local files (kept with the capture receipts, not tracked on the PR
branch): `shots/stock-workbench-1006.png` (stock error modal) and
`shots/synced-workbench-connected.png` (synced workbench connected).

## Theme note (recorded, not part of the fix)

Both the stock and the synced workbench shells render the light theme in
this packaged build while only the synced one actually connects its
workspace; the theme default alone is therefore not a reliable
loaded-versus-broken indicator for this image.
