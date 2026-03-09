# OpenHands 1.x Release History

Detailed record of files changed in each 1.x release tag commit, compiled from `git show <tag>`.

## Release 1.0.0

- **Tag commit**: `0cb27a4` (Dec 15, 2025)
- **Files changed**: 1
  - `openhands/storage/settings/file_settings_store.py` (unrelated fix, not a version bump)
- **Notes**: The 1.0.0 tag was applied to an existing commit. No version bump commit was made — version files were already set to `1.0.0` by a prior PR.

## Release 1.1.0

- **Tag commit**: `9885dde` (Dec 30, 2025)
- **Files changed**: 7 ✅ (most complete release)
  1. `pyproject.toml` — `version = "1.1.0"`
  2. `frontend/package.json` — `"version": "1.1.0"`
  3. `frontend/package-lock.json` — `"version": "1.1.0"` (2 places)
  4. `Development.md` — runtime tag `1.0-nikolaik` → `1.1-nikolaik`
  5. `docker-compose.yml` — runtime tag `1.0-nikolaik` → `1.1-nikolaik`
  6. `containers/dev/compose.yml` — runtime tag `1.0-nikolaik` → `1.1-nikolaik`
  7. `openhands/runtime/impl/kubernetes/README.md` — runtime tag `1.0-nikolaik` → `1.1-nikolaik`
- **Notes**: This is the gold standard release commit — all version and Docker references were updated.

## Release 1.2.0

- **Tag commit**: `c97d661` (Jan 15, 2026)
- **Files changed**: 7 ✅ (same complete pattern as 1.1.0)
  1. `pyproject.toml` — `version = "1.2.0"`
  2. `frontend/package.json` — `"version": "1.2.0"`
  3. `frontend/package-lock.json` — `"version": "1.2.0"` (2 places)
  4. `Development.md` — runtime tag `1.1-nikolaik` → `1.2-nikolaik`
  5. `docker-compose.yml` — runtime tag `1.1-nikolaik` → `1.2-nikolaik`
  6. `containers/dev/compose.yml` — runtime tag `1.1-nikolaik` → `1.2-nikolaik`
  7. `openhands/runtime/impl/kubernetes/README.md` — runtime tag `1.1-nikolaik` → `1.2-nikolaik`
- **Notes**: Last release to update all Docker image references.

## Release 1.2.1

- **Tag commit**: `87eaf70` (Jan 15, 2026)
- **Files changed**: 3 ⚠️ (Docker image tags NOT updated)
  1. `pyproject.toml` — `version = "1.2.1"`
  2. `frontend/package.json` — `"version": "1.2.1"`
  3. `frontend/package-lock.json` — `"version": "1.2.1"` (2 places)
- **Notes**: Patch release, same day as 1.2.0. Docker image tags left at `1.2` (which was still correct for a patch on `1.2.x`).

## Release 1.3.0

- **Tag commit**: `d063c8c` (Feb 2, 2026)
- **Files changed**: 3 ⚠️ (Docker image tags NOT updated)
  1. `pyproject.toml` — `version = "1.3.0"`
  2. `frontend/package.json` — `"version": "1.3.0"`
  3. `frontend/package-lock.json` — `"version": "1.3.0"` (2 places)
- **Notes**: Docker image tags were left stale at `1.2`. Between 1.2.1 and 1.3.0, the compose files were refactored from `SANDBOX_RUNTIME_CONTAINER_IMAGE` to `AGENT_SERVER_IMAGE_REPOSITORY` + `AGENT_SERVER_IMAGE_TAG` (commit `650bf8c`), and `docker-compose.yml` was switched from runtime to agent-server images (commit `40fb693`). The tags in these files were not bumped to `1.3`.

## Release 1.4.0

- **Tag commit**: `495f48b` (Feb 17, 2026)
- **Files changed**: 3 ⚠️ (Docker image tags NOT updated)
  1. `pyproject.toml` — `version = "1.4.0"`
  2. `frontend/package.json` — `"version": "1.4.0"`
  3. `frontend/package-lock.json` — `"version": "1.4.0"` (2 places)
- **Notes**: Docker image tags remained stale. `Development.md` and `kubernetes/README.md` still reference `1.2-nikolaik`. `docker-compose.yml` references a commit hash `31536c8-python`. `containers/dev/compose.yml` still references `1.2-nikolaik`.

## Key Observations

### Stale Docker Image Tags (as of 1.4.0)

| File | Current value | Should be |
|------|--------------|-----------|
| `Development.md` | `runtime:1.2-nikolaik` | `runtime:1.4-nikolaik` |
| `containers/dev/compose.yml` | `AGENT_SERVER_IMAGE_TAG:-1.2-nikolaik` | Should reference current version |
| `docker-compose.yml` | `AGENT_SERVER_IMAGE_TAG:-31536c8-python` | Should reference release tag |
| `openhands/runtime/impl/kubernetes/README.md` | `runtime:1.2-nikolaik` | `runtime:1.4-nikolaik` |

### V1 Architecture Change (between 1.2.x and 1.3.0)

The compose files were refactored to use `AGENT_SERVER_IMAGE_*` environment variables instead of `SANDBOX_RUNTIME_CONTAINER_IMAGE`. This happened in:
- `650bf8c` — Switched env var names in both compose files
- `40fb693` — Changed `docker-compose.yml` from `runtime` to `agent-server` image

### SDK Bump Pattern

SDK version bumps (`openhands-sdk`, `openhands-agent-server`, `openhands-tools`) happen in separate PRs, not in the release commit. These update:
- `pyproject.toml` (2 sections)
- `openhands/app_server/sandbox/sandbox_spec_service.py` (agent server image tag)
- `poetry.lock`, `uv.lock`, `enterprise/poetry.lock` (auto-regenerated)

### SaaS Release Branches

Each release also has a corresponding `saas-rel-X.Y.Z` branch (e.g., `saas-rel-1.16.0`) used for the SaaS deployment pipeline. These branches are created from `main` and may receive cherry-picked hotfixes.
