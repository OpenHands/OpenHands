# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**nue** is a self-hosted, customizable fork of [OpenHands](https://github.com/All-Hands-AI/OpenHands) — an AI-driven software development agent framework. It consists of a Python FastAPI backend, a React 19 + TypeScript frontend, a Docker-based sandbox runtime for agent execution, and a BDD test suite.

## Commands

### Setup

```bash
make build                    # Full setup: backend deps + frontend build + pre-commit hooks
make install-pre-commit-hooks # Must run before first commit
```

### Running

```bash
make run            # Full stack (backend on :3000, frontend on :3001)
make start-backend  # Backend only (uvicorn with hot reload)
make start-frontend # Frontend only (Vite dev server)
make docker-run     # Run via Docker Compose
```

### Testing

```bash
# Backend unit tests
poetry run pytest tests/unit/test_foo.py
poetry run pytest tests/unit/test_foo.py::TestClass::test_method

# BDD tests (Gherkin feature files)
make test-bdd          # Full BDD suite (mocked + real services)
make test-bdd-fast     # Mocked-only (fast; @fast marker)
poetry run pytest tests/bdd --gherkin-terminal-reporter -k "scenario name"

# Frontend
cd frontend && npm run test
cd frontend && npm run test -- -t "test name"
```

### Linting

```bash
make lint                     # All linters (backend + frontend)

# Backend only
pre-commit run --all-files --config ./dev_config/python/.pre-commit-config.yaml

# Frontend only
cd frontend && npm run lint:fix && npm run build
```

## Architecture

### Backend (`openhands/`)

The entry point is `openhands/app_server/app.py`, which initialises the FastAPI application with middleware, MCP (Model Context Protocol) support, and mounts the API router.

- `app_server/app_conversation/` — Conversation lifecycle management, session state, and agent orchestration
- `app_server/integrations/` — Git provider adapters: GitHub, GitLab, Forgejo, Azure DevOps, Bitbucket
- `app_server/sandbox/` — Docker sandbox runtime management (spawns isolated containers for agent code execution)
- `app_server/mcp/` — MCP server/client support (tools, resources, prompts exposed via Model Context Protocol)
- `app_server/services/` — Business logic (auth, billing, user management, etc.)
- `app_server/pending_messages/` — Async message queue between agent and client

**LLM abstraction:** `litellm` is used as the universal LLM provider layer, with direct SDK imports for `anthropic`, `openai`, and `google-genai`.

### Frontend (`frontend/`)

React 19 SPA built with Vite and TypeScript.

- `src/api/` — Data access layer (raw fetch wrappers over the backend REST API)
- `src/hooks/` — TanStack Query hooks (`useQuery`/`useMutation`) wrapping `src/api/`; this is the primary data-fetching interface for components
- `src/stores/` — Zustand stores for global UI state (conversation, settings, etc.)
- `src/components/` — UI components including xterm.js terminal, Monaco code editor, and chat interface
- `src/routes/` — React Router v7 route definitions

Component library: `@heroui/react`. Styling: Tailwind CSS.

### BDD Test Suite (`tests/bdd/`)

Tests are written in Gherkin (`.feature` files in `tests/bdd/features/`) and run with `pytest-bdd`. Step implementations live alongside features. Two categories:

- `@fast` — fully mocked (LLM, sandbox, app server via mocked fixtures); runnable without Docker
- `@slow` — real services; requires Docker and network

Playwright is wired into the BDD framework for browser-driven E2E scenarios.

### Skills / Microagents (`skills/`)

Markdown files (with optional YAML frontmatter) that inject knowledge into agent context based on trigger keywords. Project-level skills live in `.openhands/skills/` (V1 format) or `.openhands/microagents/` (V0 format). Global shared skills are in `skills/`.

### Configuration

- Runtime config: `~/.nue/` (auto-migrated from `~/.openhands/`)
- Template: `config.template.toml`
- Database: SQLite in development, PostgreSQL in production
- Key env vars: `RUNTIME` (`docker` default or `local`), LLM provider API keys

## Key Constraints

- Python 3.12 or 3.13 required (managed by Poetry)
- Node.js 22.12.0+ required
- Pre-commit hooks (ruff, mypy, pyproject-fmt) are mandatory and must pass before commits
- The `AppMode.OSS` guard is enforced by a pre-commit hook — do not bypass it
