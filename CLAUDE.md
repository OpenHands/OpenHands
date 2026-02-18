# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

See also: `AGENTS.md` for detailed repository structure, frontend/backend conventions, and enterprise setup.

## Quick Reference Commands

### Build & Run
```bash
make build                    # Full project build (Python + frontend)
make run                      # Start backend + frontend (default: localhost:3000 + :3001)
make start-backend            # Backend only (uvicorn with hot reload)
make start-frontend           # Frontend only (Vite dev server)
```

### Backend (Python)
```bash
# Lint (runs ruff, mypy, pre-commit hooks)
pre-commit run --config ./dev_config/python/.pre-commit-config.yaml          # staged files
pre-commit run --all-files --config ./dev_config/python/.pre-commit-config.yaml  # all files

# Tests
poetry run pytest tests/unit/test_xxx.py            # single test file
poetry run pytest tests/unit/test_xxx.py::test_name  # single test function
poetry run pytest tests/unit/ -x                     # all unit tests, stop on first failure
```

### Frontend (React, in `frontend/`)
```bash
cd frontend
npm install          # install deps
npm run build        # production build
npm run test         # run vitest
npm run test -- -t "TestName"  # specific test
npm run lint:fix     # eslint + prettier fix
npm run make-i18n    # regenerate i18n declarations
```

## Architecture Overview

OpenHands is an AI software engineering platform with a Python backend and React frontend.

### Backend (`openhands/`)
- **server/** — FastAPI REST API + WebSocket (Socket.IO) server
- **agenthub/** — Agent implementations (CodeAct, browsing, delegator, etc.)
- **controller/** — Agent execution loop and state machine
- **runtime/** — Sandboxed execution environments (Docker, local, e2b, modal, daytona)
- **events/** — Event system (actions + observations) for agent-server communication
- **llm/** — LiteLLM-based LLM interface with retry, caching, metrics
- **mcp/** — MCP (Model Context Protocol) server integration
- **microagent/** — Specialized prompt injection system (markdown files loaded by triggers)
- **storage/** — Persistence layer (file, S3, in-memory) + data models
- **core/** — Configuration, logging, and shared utilities

### Frontend (`frontend/`)
- React 19 + React Router 7 + TypeScript + Vite
- UI: HeroUI (component library) + Tailwind CSS
- State: Zustand (local) + TanStack Query (server data)
- Real-time: Socket.IO client for agent communication
- Tests: Vitest + Testing Library + Playwright (e2e)

### Key Patterns
- **Event-driven architecture**: Agents communicate through Action/Observation events, not direct calls
- **Runtime abstraction**: Code execution happens in sandboxed runtimes; the `runtime/` layer abstracts Docker, local, and cloud providers behind a common interface
- **Data fetching**: UI components never call API directly — always through TanStack Query hooks (`hooks/query/`, `hooks/mutation/`) wrapping the data access layer (`src/api/`)

## Linting & Pre-commit

Backend uses ruff (linting + formatting) and mypy (type checking) configured in `dev_config/python/`. The pre-commit config is at `dev_config/python/.pre-commit-config.yaml`.

Frontend uses eslint + prettier with lint-staged via husky.

**Important**: Use `AppMode.OPENHANDS`, never `AppMode.OSS` — there's a pre-commit hook that enforces this.

## Dependencies

- Python 3.12+ managed by Poetry
- Node.js 22+ / npm for frontend
- Docker for runtime sandboxing (can skip with `INSTALL_DOCKER=0`)
