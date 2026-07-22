# OpenHands Review Rules

This file documents repository-specific review rules that the Reviewer Agent
uses when analyzing PRs in the OpenHands/OpenHands repository.

## Frontend: i18n / Translation Key Usage

**Never dynamically construct i18n keys via string interpolation or template literals.**

All translation keys must come from the `I18nKey` enum (`frontend/src/i18n/declaration.ts`)
or from canonical mapping objects like `AGENT_STATUS_MAP` (`frontend/src/utils/status.ts`).

**Flag:**
- Any `t(...)` / `i18next.t(...)` call with runtime-constructed keys
- Any new i18n key referenced in code that does not exist in `frontend/src/i18n/translation.json`

**Correct:**
```ts
import { AGENT_STATUS_MAP } from "#/utils/status";
const message = AGENT_STATUS_MAP[agentState]
  ? t(AGENT_STATUS_MAP[agentState])
  : fallback;
```

**Incorrect:**
```ts
const message = t(`STATUS$${agentState.toUpperCase()}`);
```

## Frontend: Data Fetching Architecture

UI components must never call API client methods (`frontend/src/api/`) directly.
All data access must go through TanStack Query hooks:

```
UI component → TanStack Query hook → API client → API endpoint
```

**Flag:** any component importing directly from `#/api/` without a TanStack Query wrapper.

## Backend: Python Standards

- Follow PEP 8. Use `ruff` and `mypy` — the project's pre-commit config enforces these
- Type hints required for all public functions
- Use async patterns for I/O-bound operations (the app server is async FastAPI)
- Error responses must follow the project's envelope format (success indicator + data + error)

## Docker Image References

- Agent server image must use `ghcr.io/openhands/agent-server:<tag>-python`
- Tags use semantic version for releases (`1.12.0-python`) or merge-commit SHA for dev pins
- CI guard: `check-version-consistency.yml` enforces version alignment
- Flag any Docker image tag that doesn't follow this pattern

## Lockfile Handling

- When `pyproject.toml` dependencies change, all three lockfiles must be regenerated:
  `poetry.lock`, `uv.lock`, `enterprise/poetry.lock`
- Flag PRs that modify `pyproject.toml` without corresponding lockfile updates

## GitHub Actions Security

- Pin external third-party actions to a full 40-character commit SHA
- The trailing comment should include the version tag (e.g., `# v1.2.3`)
- GitHub-authored (`actions/*`) and first-party (`OpenHands/*`) actions are exempt
