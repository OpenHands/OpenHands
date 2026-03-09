---
name: ship-release
description: This skill should be used when the user asks to "ship a release", "cut a new version", "bump the version", "prepare a release", "create a release branch", "what files change for a release", or needs to know the release process for OpenHands 1.x versions.
---

# Ship Release

Prepare and execute an OpenHands 1.x release by bumping version numbers and Docker image tags across the repository.

## Overview

A release involves two categories of changes: the **version bump commit** (tagged as the release) and the **SDK/agent-server bump** (typically a separate PR that lands before the release).

## Files to Change

### Version Bump (the release commit itself)

Update the new version `X.Y.Z` in all of these files:

| # | File | What to change |
|---|------|----------------|
| 1 | `pyproject.toml` | `version = "X.Y.Z"` under `[tool.poetry]` |
| 2 | `frontend/package.json` | `"version": "X.Y.Z"` |
| 3 | `frontend/package-lock.json` | `"version": "X.Y.Z"` in **two** places (root object and `packages[""]`) |

### Docker Image Tags

Update the Docker image tag defaults to match the new version. The runtime images follow the pattern `X.Y-nikolaik` (major.minor only, no patch). The agent-server images may use a commit hash or a version tag.

| # | File | What to change |
|---|------|----------------|
| 4 | `Development.md` | Example `SANDBOX_RUNTIME_CONTAINER_IMAGE` value |
| 5 | `docker-compose.yml` | `AGENT_SERVER_IMAGE_TAG` default value |
| 6 | `containers/dev/compose.yml` | `AGENT_SERVER_IMAGE_TAG` default value |
| 7 | `openhands/runtime/impl/kubernetes/README.md` | `runtime_container_image` example value |

### SDK Package Bump (separate PR, before the release)

When a new SDK version is available, update these before cutting the release:

| # | File | What to change |
|---|------|----------------|
| 8 | `pyproject.toml` | `openhands-sdk`, `openhands-agent-server`, `openhands-tools` versions in **two** sections: `dependencies` array and `[tool.poetry.dependencies]` |
| 9 | `openhands/app_server/sandbox/sandbox_spec_service.py` | `AGENT_SERVER_IMAGE` constant (image tag) |
| 10 | `poetry.lock` | Auto-regenerated after pyproject.toml changes |
| 11 | `uv.lock` | Auto-regenerated after pyproject.toml changes |
| 12 | `enterprise/poetry.lock` | Auto-regenerated after pyproject.toml changes |

## Release Workflow

### Step 1: Verify Prerequisites

Confirm the SDK/agent-server packages have already been bumped in a prior merged PR. Check current values:

```bash
grep -n "openhands-sdk\|openhands-agent-server\|openhands-tools" pyproject.toml
grep -n "AGENT_SERVER_IMAGE" openhands/app_server/sandbox/sandbox_spec_service.py
```

### Step 2: Bump Version Numbers

Replace `OLD` with the previous version and `NEW` with the target version:

1. **`pyproject.toml`** — Update `version = "NEW"` under `[tool.poetry]`
2. **`frontend/package.json`** — Update `"version": "NEW"`
3. **`frontend/package-lock.json`** — Update `"version": "NEW"` in both the root object and `packages[""]` object

### Step 3: Update Docker Image Tags

Compute the major.minor tag (e.g., `1.5` from `1.5.0`):

1. **`Development.md`** — Update the runtime image example
2. **`docker-compose.yml`** — Update `AGENT_SERVER_IMAGE_TAG` default
3. **`containers/dev/compose.yml`** — Update `AGENT_SERVER_IMAGE_TAG` default
4. **`openhands/runtime/impl/kubernetes/README.md`** — Update `runtime_container_image` example

### Step 4: Commit, Tag, and Branch

```bash
git add pyproject.toml frontend/package.json frontend/package-lock.json \
  Development.md docker-compose.yml containers/dev/compose.yml \
  openhands/runtime/impl/kubernetes/README.md
git commit -m "Release X.Y.Z"
git tag X.Y.Z
```

Create a `saas-rel-X.Y.Z` branch from the tagged commit for the SaaS deployment pipeline.

### Step 5: CI Builds Docker Images Automatically

The `ghcr-build.yml` workflow triggers on tag pushes and produces:
- `ghcr.io/openhands/openhands:X.Y.Z`, `X.Y`, `X`, `latest`
- `ghcr.io/openhands/runtime:X.Y.Z-nikolaik`, `X.Y-nikolaik`

No manual Docker build steps are required.

## Historical Notes

Refer to `references/release-history.md` for a detailed record of which files were changed in each 1.x release, including inconsistencies where Docker image tags were not updated.

## Quick Verification

After preparing the release commit, verify all version references are consistent:

```bash
grep 'version = ' pyproject.toml | head -1
grep '"version"' frontend/package.json | head -1
grep -n "AGENT_SERVER_IMAGE_TAG" docker-compose.yml containers/dev/compose.yml
grep "runtime:" Development.md openhands/runtime/impl/kubernetes/README.md
```
