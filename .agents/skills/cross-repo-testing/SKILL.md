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

A feature requires cross-repo testing when the SDK calls an API that **doesn't exist yet** on the production Cloud server. Examples:
- `workspace.get_llm()` calling `GET /api/v1/users/me?expose_secrets=true`
- `workspace.get_secrets()` calling `GET /api/v1/sandboxes/{id}/settings/secrets`
- Any new sandbox settings or user-facing API consumed by `OpenHandsCloudWorkspace`

## Step-by-Step Workflow

### 1. Write and test the server-side changes

In the `OpenHands` repo, implement the new API endpoint(s). Run unit tests:

```bash
cd OpenHands
poetry run pytest tests/unit/app_server/test_<relevant>.py -v
```

Push a PR. Wait for the **"Push Enterprise Image" (Docker) CI job** to succeed — this builds `ghcr.io/openhands/enterprise-server:sha-<COMMIT>`.

### 2. Write the SDK-side changes

In `software-agent-sdk`, implement the client code (e.g., new methods on `OpenHandsCloudWorkspace`). Run SDK unit tests:

```bash
cd software-agent-sdk
pip install -e openhands-sdk -e openhands-workspace
pytest tests/ -v
```

Push a PR. SDK CI is independent — it doesn't need the server changes to pass unit tests.

### 3. Deploy the server PR to a staging feature environment

The `deploy` repo has a workflow that creates a preview branch from an OpenHands PR.

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

### 4. Wait for deployment, verify it's live

```bash
curl -s -o /dev/null -w "%{http_code}" https://ohpr-<PR>-<random>.staging.all-hands.dev/api/v1/health
# 401 = server is up (auth required). DNS may take 1-2 min on first deploy.
```

### 5. Run the SDK end-to-end test against staging

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

Or run the example script if one exists:
```bash
OPENHANDS_CLOUD_API_KEY="<key>" \
OPENHANDS_CLOUD_API_URL="https://ohpr-<PR>-<random>.staging.all-hands.dev" \
python examples/02_remote_agent_server/10_cloud_workspace_saas_credentials.py
```

### 6. Record results

Push test output to the SDK PR's `.pr/logs/` directory:
```bash
cd software-agent-sdk
# Capture stdout
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
| **SDK doesn't need a custom agent-server image** | `OpenHandsCloudWorkspace` talks to the Cloud API, not directly to the agent-server. The stock agent-server image works. Only the Cloud server needs the new code. |
