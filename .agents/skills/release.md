---
name: release
description: Guide the release process for @openhands/agent-canvas — review the release-please draft PR, mark it ready, merge it; the tag push publishes to npm and Docker.
triggers:
- release
- new release
- cut a release
- publish release
- bump version
---

# Release Process for @openhands/agent-canvas

## Overview

Releases are **trunk-based and automated by release-please**, via the shared reusable
workflows in [`OpenHands/release-actions`](https://github.com/OpenHands/release-actions):

1. PRs merge to `main` with **Conventional Commit titles** (`feat`, `fix`, `perf`, `docs`, `chore`, `build`, `ci`, `refactor`, `style`, `test`, `revert`). `.github/workflows/pr.yml` lints the title and applies the matching `type:` label; squash merge uses the PR title as the commit message.
2. On every push to `main`, `.github/workflows/release.yml` runs release-please, which maintains a **draft release PR** titled `chore(main): release X.Y.Z` accumulating everything merged since the last release.
3. The release PR pre-stages **every version bump**: `package.json`, `package-lock.json`, `config/defaults.json` (`versions.agentCanvas`), and the Docker image pins in `README.md` / `README.windows.md` (lines annotated with `x-release-please-version`).
4. Marking the release PR **Ready for review** is the explicit cut-a-release signal: `.github/workflows/release-ready.yml` notifies `#proj-agent-canvas` on Slack and labels the PR `release: ready`.
5. Merging the release PR makes release-please push the `vX.Y.Z` tag (using the org release App token) and create the GitHub Release. The same tag push triggers `npm-publish.yml` (npm) and `docker.yml` (multi-arch GHCR images).

The next version is derived from the conventional-commit types merged since the last release: `fix` → patch, `feat` → minor, any `!` suffix or `BREAKING CHANGE` footer → major. Other types (`docs`, `chore`, `refactor`, …) appear in the release notes but do not by themselves produce a release PR. Release notes are grouped by the `type:` labels via `.github/release.yml`; there is no `CHANGELOG.md` file — GitHub Releases are the changelog.

Configuration lives in `release-please-config.json` (version surfaces, draft PR) and `.release-please-manifest.json` (current released version). Maintenance releases for an older line use `release/**` branches (`release.yml` triggers there too).

**Never commit to the release PR's branch by hand** — release-please owns it and force-pushes it on every push to `main`.

---

## Step 1: Find the Release PR

```bash
gh pr list --state open --label "autorelease: pending"
```

If no release PR is open, nothing releasable (`feat`/`fix`/breaking) has merged since the last release — or `release.yml` is failing on `main`:

```bash
gh run list --workflow=release.yml --limit=3
```

---

## Step 2: Review It

Open the release PR and confirm:

- **The version** in the title matches expectations. It is computed from the merged commit types — if it looks wrong, check the conventional types of the PR titles merged since the last release. To force a specific version, merge a commit to `main` whose message contains a `Release-As: X.Y.Z` footer.
- **The staged bumps** cover all version surfaces (`package.json`, `package-lock.json`, `config/defaults.json`, `README.md`, `README.windows.md`) and the notes list the expected changes.

---

## Step 3: Cut the Release

**STOP HERE and confirm with the user before proceeding.** Marking the PR ready and merging it publishes to npm and GHCR.

```bash
gh pr ready <release-pr-number>
```

This fires the release-ready gate: a Slack notification lands in `#proj-agent-canvas` and the PR is labeled `release: ready`. Then merge the release PR (squash, like any other PR).

---

## Step 4: Watch the Pipeline

Merging the release PR triggers `release.yml` on `main`, which pushes the `vX.Y.Z` tag and creates the GitHub Release; the tag push then fires the publish workflows:

```bash
gh run list --workflow=release.yml --limit=3
gh run list --workflow=npm-publish.yml --limit=3
gh run list --workflow=docker.yml --limit=3
```

---

## Step 5: Verify the Release

```bash
# GitHub release
gh release view v<version>

# npm (allow ~2 min for publish to propagate)
npm view @openhands/agent-canvas@<version>
npm view @openhands/agent-canvas dist-tags  # stable releases get `latest`

# Docker
docker pull ghcr.io/openhands/agent-canvas:<version>
```

External install docs on docs.openhands.dev are maintained separately; update them there when closing #1073.

---

## Troubleshooting

### No release PR appears after merging to main
Only `feat`, `fix`, and breaking changes produce a release PR. Also check `release.yml` runs on `main` — the workflow fails by design if the org secrets `RELEASE_APP_ID` / `RELEASE_APP_PRIVATE_KEY` are unavailable (a `GITHUB_TOKEN` fallback would create a tag that never triggers the publish workflows).

### The proposed version is wrong
The version comes from the conventional-commit history since the last release. Fix forward: merge a commit to `main` with a `Release-As: X.Y.Z` footer to pin the next version.

### No Slack message when the PR was marked ready
`SLACK_BOT_TOKEN` is optional by design — the gate still applies the `release: ready` label and the release proceeds normally.

### package.json version doesn't match the tag
This cannot happen in the normal flow: release-please bumps `package.json` in the release PR and tags the resulting merge commit, and `npm-publish.yml` validates they match. If it ever fails, someone pushed a tag by hand — delete the tag and let release-please own tagging.
