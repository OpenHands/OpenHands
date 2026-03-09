---
name: ship-release
description: This skill should be used when the user asks to "ship a release", "cut a new version", "bump the version", "prepare a release", "create a release branch", "what files change for a release", "pin SDK to a commit", "test unreleased SDK", or needs to know the release process for OpenHands 1.x versions.
---

# Ship Release

Prepare and execute an OpenHands 1.x release, or pin SDK packages to unreleased commits for testing.

## Release Commit — Files to Change

A release commit updates the version number across 3 files and verifies compose files use agent-server images. The gold-standard pattern was established in release 1.1.0 (commit `9885dde`) and 1.2.0 (commit `c97d661`).

### Version Numbers (3 files)

| File | What to change |
|------|----------------|
| `pyproject.toml` | `version = "X.Y.Z"` under `[tool.poetry]` |
| `frontend/package.json` | `"version": "X.Y.Z"` |
| `frontend/package-lock.json` | `"version": "X.Y.Z"` in **two** places (root object and `packages[""]`) |

### Compose Files (2 files)

Both compose files should use `ghcr.io/openhands/agent-server` with the current SDK version or commit hash tag.

| File | What to verify |
|------|----------------|
| `docker-compose.yml` | `AGENT_SERVER_IMAGE_REPOSITORY` defaults to agent-server, `AGENT_SERVER_IMAGE_TAG` is current |
| `containers/dev/compose.yml` | Same — must use agent-server, not runtime |

> **CI enforcement:** The `check-version-consistency.yml` workflow validates version consistency and compose file image references on every PR and push to main.

### V0 Legacy Files (not part of V1 release)

The following files reference `SANDBOX_RUNTIME_CONTAINER_IMAGE` / `runtime_container_image` for the V0 local-dev and Kubernetes runtime paths. These are **not** updated as part of a V1 release:

- `Development.md` — example `SANDBOX_RUNTIME_CONTAINER_IMAGE` for local Docker runtime
- `openhands/runtime/impl/kubernetes/README.md` — `runtime_container_image` config example

These are still used by `docker_runtime.py`, `kubernetes_runtime.py`, `remote_runtime.py`, `modal_runtime.py`, and `daytona_runtime.py` but follow a separate update cadence.

## SDK Package Bump (separate PR, before the release)

When a new SDK version is available, land a separate PR before the release commit. This updates 5 files (plus 3 auto-generated lock files). See commit `929dcc3` (SDK 1.11.5 bump) and `aa22d34` (SDK 1.11.4 bump) for examples.

| File | What to change |
|------|----------------|
| `pyproject.toml` | `openhands-sdk`, `openhands-agent-server`, `openhands-tools` in **two** sections: `dependencies` array and `[tool.poetry.dependencies]` |
| `openhands/app_server/sandbox/sandbox_spec_service.py` | `AGENT_SERVER_IMAGE` constant |
| `poetry.lock` | Auto-regenerated |
| `uv.lock` | Auto-regenerated |
| `enterprise/poetry.lock` | Auto-regenerated |

## Release Workflow

### Step 1: Verify the SDK bump has landed

```bash
grep -n "openhands-sdk\|openhands-agent-server\|openhands-tools" pyproject.toml
grep -n "AGENT_SERVER_IMAGE" openhands/app_server/sandbox/sandbox_spec_service.py
```

### Step 2: Bump version numbers and Docker image tags

Update the version in the 3 version files, then commit, tag, and create the SaaS branch:

```bash
git add pyproject.toml frontend/package.json frontend/package-lock.json
git commit -m "Release X.Y.Z"
git tag X.Y.Z
```

Create a `saas-rel-X.Y.Z` branch from the tagged commit for the SaaS deployment pipeline.

### Step 3: CI builds Docker images automatically

The `ghcr-build.yml` workflow triggers on tag pushes and produces:
- `ghcr.io/openhands/openhands:X.Y.Z`, `X.Y`, `X`, `latest`
- `ghcr.io/openhands/runtime:X.Y.Z-nikolaik`, `X.Y-nikolaik`

The tagging logic lives in `containers/build.sh` — when `GITHUB_REF_NAME` matches a semver pattern, it auto-generates major, major.minor, and `latest` tags.

## Development: Pin SDK to an Unreleased Commit

To test an SDK change that has not been released to PyPI, pin the three SDK packages (`openhands-sdk`, `openhands-agent-server`, `openhands-tools`) to a git commit or branch from the [software-agent-sdk](https://github.com/OpenHands/software-agent-sdk) monorepo. Each package lives in a subdirectory of that repo.

### Files to change

Update **both** dependency sections in `pyproject.toml` (the PEP 508 `dependencies` array and the Poetry `[tool.poetry.dependencies]` section). If using `uv`, also add a `[tool.uv.sources]` section.

### Pin to a specific commit

Example from commit `169fb76` (pinning all 3 packages to SDK commit `100e9af`):

**`dependencies` array (PEP 508 format):**
```toml
"openhands-agent-server @ git+https://github.com/OpenHands/software-agent-sdk.git@100e9af#subdirectory=openhands-agent-server",
"openhands-sdk @ git+https://github.com/OpenHands/software-agent-sdk.git@100e9af#subdirectory=openhands-sdk",
"openhands-tools @ git+https://github.com/OpenHands/software-agent-sdk.git@100e9af#subdirectory=openhands-tools",
```

**`[tool.poetry.dependencies]` (Poetry format):**
```toml
openhands-sdk = { git = "https://github.com/OpenHands/software-agent-sdk.git", rev = "100e9af", subdirectory = "openhands-sdk" }
openhands-agent-server = { git = "https://github.com/OpenHands/software-agent-sdk.git", rev = "100e9af", subdirectory = "openhands-agent-server" }
openhands-tools = { git = "https://github.com/OpenHands/software-agent-sdk.git", rev = "100e9af", subdirectory = "openhands-tools" }
```

### Pin to a branch

Example from commit `430ee1c` (pinning to branch `openhands/issue-2228-sdk-settings-schema`):

**`[tool.poetry.dependencies]`:**
```toml
openhands-sdk = { git = "https://github.com/OpenHands/software-agent-sdk.git", branch = "openhands/issue-2228-sdk-settings-schema", subdirectory = "openhands-sdk" }
```

### Using `[tool.uv.sources]` override

When only `uv` needs the override (keep PyPI versions in the main arrays), add a `[tool.uv.sources]` section. Example from commit `1daca49`:

```toml
[tool.uv.sources]
openhands-sdk = { git = "https://github.com/OpenHands/software-agent-sdk.git", subdirectory = "openhands-sdk", rev = "4170cca" }
openhands-agent-server = { git = "https://github.com/OpenHands/software-agent-sdk.git", subdirectory = "openhands-agent-server", rev = "4170cca" }
openhands-tools = { git = "https://github.com/OpenHands/software-agent-sdk.git", subdirectory = "openhands-tools", rev = "4170cca" }
```

### Update the agent-server Docker image

When testing an unreleased SDK commit, also update the agent-server Docker image tag in `openhands/app_server/sandbox/sandbox_spec_service.py` to match. The image is built by CI for every commit pushed to the SDK repo. Example from commit `fb37bbc`:

```python
AGENT_SERVER_IMAGE = 'ghcr.io/openhands/agent-server:<short-commit-hash>-python'
```

### Regenerate lock files

After changing `pyproject.toml`, regenerate the lock files:

```bash
poetry lock
uv lock
cd enterprise && poetry lock && cd ..
```

### CI guard

The existing `check-package-versions.yml` workflow blocks merging to `main` if `[tool.poetry.dependencies]` contains any `rev` fields. This ensures unreleased SDK pins do not accidentally ship in a release.
