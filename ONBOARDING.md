# Contributing to OpenHands

OpenHands is an AI software engineering agent with a **Python backend** (`openhands/`) and a **React frontend** (`frontend/`). This file is a short orientation; full agent and quality rules live in [`AGENTS.md`](AGENTS.md).

## Repository layout

| Area | Path | Notes |
|------|------|--------|
| Backend | `openhands/` | App server, agents, runtime, integrations |
| Frontend | `frontend/` | React Router app, Vitest, Playwright |
| Tests | `tests/` | Python tests; frontend tests under `frontend/__tests__/` and `frontend/tests/` |
| Config | Root `Makefile`, `pyproject.toml`, `docker-compose.yml` | See `AGENTS.md` for run targets |

## Setup and run

- Full stack build: `make build` (see **General Setup** in `AGENTS.md`).
- Running the app locally: **Running OpenHands with OpenHands** in `AGENTS.md` (env vars, `make run`, ports).
- **Windows / Docker Compose**: optional helpers and notes are documented in `AGENTS.md` (Docker Desktop, WSL, compose).

## Pull requests

1. Branch from an up-to-date `main` (prefer `git fetch` + rebase when the remote moves).
2. Stage changes deliberately (`git add <path>`); avoid committing local workspace data.
3. **Do not commit secrets or local workspace state**: the repo ignores `.openhands/` (JWT, keys, DB, exported conversations) and `.debug_console_plus/` — see `.gitignore`.
4. Before pushing: install pre-commit hooks once (`make install-pre-commit-hooks`), then run the checks in `AGENTS.md` for the areas you touched (Python pre-commit config, frontend `npm run lint:fix` / `npm run build`, etc.).

## Where to read next

- [`AGENTS.md`](AGENTS.md) — setup, running, linting, Git practices, `.pr/` artifacts for PR-only notes.
