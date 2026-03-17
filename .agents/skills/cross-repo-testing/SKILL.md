---
name: cross-repo-testing
description: This skill should be used when the user asks to "test a cross-repo feature", "deploy a feature branch to staging", "test SDK against OH Cloud", "e2e test a cloud workspace feature", "test provider tokens", "test secrets inheritance", or when changes span the SDK and OpenHands server repos and need end-to-end validation against a staging deployment.
triggers:
- cross-repo
- staging deployment
- feature branch deploy
- test against cloud
- e2e cloud
---

# Cross-Repo Testing: SDK ↔ OpenHands Cloud

How to end-to-end test features that span `OpenHands/software-agent-sdk` and `OpenHands/OpenHands` (the Cloud backend).

## Repository Map

| Repo | Role | What lives here |
|------|------|-----------------|
| [`software-agent-sdk`](https://github.com/OpenHands/software-agent-sdk) | Agent core | `openhands-sdk`, `openhands-workspace`, `openhands-tools` packages. `OpenHandsCloudWorkspace` lives here. |
| [`OpenHands`](https://github.com/OpenHands/OpenHands) | Cloud backend | FastAPI server (`openhands/app_server/`), sandbox management, auth, enterprise integrations. Deployed as OH Cloud. |
| [`deploy`](https://github.com/OpenHands/deploy) | Infrastructure | Helm charts + GitHub Actions that build the enterprise Docker image and deploy to staging/production. |

**Data flow:** SDK client → OH Cloud API (`/api/v1/...`) → sandbox agent-server (inside runtime container)

## When You Need This

There are **two flows** depending on which direction the dependency goes:

| Flow | When | Example |
|------|------|---------|
| **A — SDK client → new Cloud API** | The SDK calls an API that doesn't exist yet on production | `workspace.get_llm()` calling `GET /api/v1/users/me?expose_secrets=true` |
| **B — OH server → new SDK code** | The Cloud server needs unreleased SDK packages or a new agent-server image | Server consumes a new tool, agent behavior, or workspace method from the SDK |

Flow A only requires deploying the server PR. Flow B requires pinning the SDK to an unreleased commit in the server PR **and** using the SDK PR's agent-server image. Both flows may apply simultaneously.

---

## Flow A: SDK Client Tests Against New Cloud API

Use this when the SDK calls an endpoint that only exists on the server PR branch.

### A1. Write and test the server-side changes

In the `OpenHands` repo, implement the new API endpoint(s). Run unit tests:

```bash
cd OpenHands
poetry run pytest tests/unit/app_server/test_<relevant>.py -v
```

Push a PR. Wait for the **"Push Enterprise Image" (Docker) CI job** to succeed — this builds `ghcr.io/openhands/enterprise-server:sha-<COMMIT>`.

### A2. Write the SDK-side changes

In `software-agent-sdk`, implement the client code (e.g., new methods on `OpenHandsCloudWorkspace`). Run SDK unit tests:

```bash
cd software-agent-sdk
pip install -e openhands-sdk -e openhands-workspace
pytest tests/ -v
```

Push a PR. SDK CI is independent — it doesn't need the server changes to pass unit tests.

### A3. Deploy the server PR to staging

See [Deploying to a Staging Feature Environment](#deploying-to-a-staging-feature-environment) below.

### A4. Run the SDK e2e test against staging

See [Running E2E Tests Against Staging](#running-e2e-tests-against-staging) below.

---

## Flow B: OH Server Needs Unreleased SDK Code

Use this when the Cloud server depends on SDK changes that haven't been released to PyPI yet. The server's runtime containers run the `agent-server` image built from the SDK repo, so the server PR must be configured to use the SDK PR's image and packages.

### B1. Get the SDK PR merged (or identify the commit)

The SDK PR must have CI pass so its agent-server Docker image is built. The image is tagged with the **merge-commit SHA** from GitHub Actions — NOT the head-commit SHA shown in the PR.

Find the correct image tag:
- Check the SDK PR description for an `AGENT_SERVER_IMAGES` section
- Or check the "Consolidate Build Information" CI job for `"short_sha": "<tag>"`

### B2. Pin SDK packages to the commit in the OpenHands PR

In the `OpenHands` repo PR, update 3 files + regenerate 3 lock files (see the `update-sdk` skill for full details):

**`pyproject.toml`** — pin all 3 SDK packages in **both** `dependencies` and `[tool.poetry.dependencies]`:
```toml
# dependencies array (PEP 508)
"openhands-sdk @ git+https://github.com/OpenHands/software-agent-sdk.git@<COMMIT>#subdirectory=openhands-sdk",
"openhands-agent-server @ git+https://github.com/OpenHands/software-agent-sdk.git@<COMMIT>#subdirectory=openhands-agent-server",
"openhands-tools @ git+https://github.com/OpenHands/software-agent-sdk.git@<COMMIT>#subdirectory=openhands-tools",

# [tool.poetry.dependencies]
openhands-sdk = { git = "https://github.com/OpenHands/software-agent-sdk.git", rev = "<COMMIT>", subdirectory = "openhands-sdk" }
openhands-agent-server = { git = "https://github.com/OpenHands/software-agent-sdk.git", rev = "<COMMIT>", subdirectory = "openhands-agent-server" }
openhands-tools = { git = "https://github.com/OpenHands/software-agent-sdk.git", rev = "<COMMIT>", subdirectory = "openhands-tools" }
```

**`openhands/app_server/sandbox/sandbox_spec_service.py`** — use the SDK's merge-commit SHA:
```python
AGENT_SERVER_IMAGE = 'ghcr.io/openhands/agent-server:<merge-commit-sha>-python'
```

**Regenerate lock files:**
```bash
poetry lock && uv lock && cd enterprise && poetry lock && cd ..
```

### B3. Wait for the OpenHands enterprise image to build

Push the pinned changes. The OpenHands CI will build a new enterprise Docker image (`ghcr.io/openhands/enterprise-server:sha-<OH_COMMIT>`) that bundles the unreleased SDK. Wait for the "Push Enterprise Image" job to succeed.

### B4. Deploy and test

Follow [Deploying to a Staging Feature Environment](#deploying-to-a-staging-feature-environment) using the new OpenHands commit SHA.

### B5. Before merging: remove the pin

**CI guard:** `check-package-versions.yml` blocks merge to `main` if `[tool.poetry.dependencies]` contains `rev` fields. Before the OpenHands PR can merge, the SDK PR must be merged and released to PyPI, then the pin must be replaced with the released version number.

---

## Deploying to a Staging Feature Environment

The `deploy` repo creates preview environments from OpenHands PRs.

**Option A — GitHub Actions UI (preferred):**
Go to `OpenHands/deploy` → Actions → "Create OpenHands preview PR" → enter the OpenHands PR number. This creates a branch `ohpr-<PR>-<random>` and opens a deploy PR.

**Option B — Update an existing feature branch:**
```bash
cd deploy
git checkout ohpr-<PR>-<random>
# In .github/workflows/deploy.yaml, update BOTH:
#   OPENHANDS_SHA: "<full-40-char-commit>"
#   OPENHANDS_RUNTIME_IMAGE_TAG: "<same-commit>-nikolaik"
git commit -am "Update OPENHANDS_SHA to <commit>" && git push
```

**Before updating the SHA**, verify the enterprise Docker image exists:
```bash
gh api repos/OpenHands/OpenHands/actions/runs \
  --jq '.workflow_runs[] | select(.head_sha=="<COMMIT>") | "\(.name): \(.conclusion)"' \
  | grep Docker
# Must show: "Docker: success"
```

The deploy CI auto-triggers and creates the environment at:
```
https://ohpr-<PR>-<random>.staging.all-hands.dev
```

**Wait for it to be live:**
```bash
curl -s -o /dev/null -w "%{http_code}" https://ohpr-<PR>-<random>.staging.all-hands.dev/api/v1/health
# 401 = server is up (auth required). DNS may take 1-2 min on first deploy.
```

## Running E2E Tests Against Staging

**Critical: Feature deployments have their own Keycloak instance.** API keys from `app.all-hands.dev` or `$OPENHANDS_API_KEY` will NOT work. You need a test API key for the specific feature deployment. The user must provide one, or you log in via the feature deployment's browser UI.

```python
from openhands.workspace import OpenHandsCloudWorkspace

STAGING = "https://ohpr-<PR>-<random>.staging.all-hands.dev"

with OpenHandsCloudWorkspace(
    cloud_api_url=STAGING,
    cloud_api_key="<test-api-key-for-this-deployment>",
) as workspace:
    # Test the new feature
    llm = workspace.get_llm()
    secrets = workspace.get_secrets()
    print(f"LLM: {llm.model}, secrets: {list(secrets.keys())}")
```

Or run an example script:
```bash
OPENHANDS_CLOUD_API_KEY="<key>" \
OPENHANDS_CLOUD_API_URL="https://ohpr-<PR>-<random>.staging.all-hands.dev" \
python examples/02_remote_agent_server/10_cloud_workspace_saas_credentials.py
```

### Recording results

Push test output to the SDK PR's `.pr/logs/` directory:
```bash
cd software-agent-sdk
python test_script.py 2>&1 | tee .pr/logs/<test_name>.log
git add -f .pr/logs/<test_name>.log .pr/README.md
git commit -m "docs: add e2e test results" && git push
```

Comment on **both PRs** with pass/fail summary and link to logs.

## Key Gotchas

| Gotcha | Details |
|--------|---------|
| **Feature env auth is isolated** | Each `ohpr-*` deployment has its own Keycloak. Production API keys don't work. |
| **Two SHAs in deploy.yaml** | `OPENHANDS_SHA` and `OPENHANDS_RUNTIME_IMAGE_TAG` must both be updated. The runtime tag is `<sha>-nikolaik`. |
| **Enterprise image must exist** | The Docker CI job on the OpenHands PR must succeed before you can deploy. If it hasn't run, push an empty commit to trigger it. |
| **DNS propagation** | First deployment of a new branch takes 1-2 min for DNS. Subsequent deploys are instant. |
| **Merge-commit SHA ≠ head SHA** | SDK CI tags Docker images with GitHub Actions' merge-commit SHA, not the PR head SHA. Check the SDK PR description or CI logs for the correct tag. |
| **SDK pin blocks merge** | `check-package-versions.yml` prevents merging an OpenHands PR that has `rev` fields in `[tool.poetry.dependencies]`. The SDK must be released to PyPI first. |
| **Flow A: stock agent-server is fine** | When only the Cloud API changes, `OpenHandsCloudWorkspace` talks to the Cloud server, not the agent-server. No custom image needed. |
| **Flow B: agent-server image is required** | When the server needs new SDK code inside runtime containers, you must pin to the SDK PR's agent-server image. |
