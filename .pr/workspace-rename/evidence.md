# Workspace rename — local verification evidence

Temporary PR artifact (removed by the release PR).

This PR could not run `agent-canvas-e2e` or produce a branch image: the workflow needs
maintainer approval, so no `ghcr.io/openhands/agent-canvas:pr-<n>` tag exists. The
evidence below comes from a throwaway container built from this branch, with no live
provider or external API calls.

## How the image was built

* Base: `ghcr.io/openhands/agent-canvas:latest` layers via `docker/Dockerfile`
  (`AGENT_SERVER_IMAGE=ghcr.io/openhands/agent-server:1.62.0-python`,
  `AGENT_SERVER_VERSION=1.62.0`, `AUTOMATION_VERSION=1.32.1`).
* Frontend: built from this branch's source, with `@openhands/typescript-client`
  installed from the SDK PR's `npm pack` tarball (the published `1.62.0` client has no
  `renameWorkspace`, so a stock pin cannot compile this branch).
* Backend: the SDK PR's Python packages installed from
  `git+https://github.com/Sarke/FORK-software-agent-sdk@dcfec261` (sdk, tools,
  workspace, agent-server). The base image ships agent-server as a PyInstaller
  one-file binary, so the local build points `openhands-agent-server` at
  `python -m openhands.agent_server` to serve the installed source.

Baseline container: released `agent-canvas:latest` image, same seed data.

## API matrix (`/api/workspaces`)

| Case | Baseline (`latest`) | This branch |
| --- | --- | --- |
| `PATCH /api/workspaces?path=<seed>/alpha` `{"name":"Alpha Renamed"}` | **405** | **200**, snapshot returns the new label, `path` unchanged |
| `PATCH /api/workspaces/parents?path=<seed>` `{"name":"Dev Roots Renamed"}` | **405** | **200** |
| unknown path | **405** | **404** `{"detail":"Workspace not found"}` |
| `{"name":""}` | **405** | **422** `string_too_short` |
| 300-character name | **405** | **422** `string_too_long` |
| same name repeated | **405** | **200**, `workspaces.json` byte-identical (no rewrite) |
| corrupt `workspaces.json` | **405** | **409** `{"detail":"Workspaces file is corrupted"}` |
| after container restart | — | renamed labels persisted |

`PATCH` returns the updated registry snapshot, so the client can replace its cache from
the response.

## UI flow

* `before-manage-workspaces.png` — released image: no rename affordance at all.
* `after-row-hover.png` — pencil appears when the row is hovered.
* `after-inline-editor.png` — editor seeded with the current name.
* `after-typed.png` — Save enabled once the name differs.
* `after-saved.png` — row shows the new label and the toast confirms.
* `after-dropdown-open-workspace.png` — the dropdown and the dialog show the renamed
  workspace and parent.

Screenshots captured with Playwright against the branch container; the only stubbed
request was `/server_info`, where the version fields are rewritten to `1.63.0`. The
branch reports `1.62.0` because the SDK release train bumps the version at release time,
and the UI deliberately disables rename for backends older than 1.63.0 —
`after-rename-gated-1.62.0.png` shows that gate working against the unmodified
`1.62.0` response (pencil rendered but disabled, with the version tooltip).

Keyboard behaviour verified in the flow above: `Enter` saves, `Escape` cancels, blur does
not commit, and Save is disabled while the name is empty or unchanged.
