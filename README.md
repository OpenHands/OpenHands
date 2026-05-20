<a name="readme-top"></a>

<div align="center">
  <h1 align="center" style="border-bottom: none">nue: AI-Driven Development</h1>
  <p>A self-hosted, customizable fork of <a href="https://github.com/OpenHands/OpenHands">OpenHands</a></p>
</div>

<div align="center">
  <a href="https://github.com/GusCayresMindsight/nue-agentic-work/blob/main/LICENSE"><img src="https://img.shields.io/badge/LICENSE-MIT-20B2AA?style=for-the-badge" alt="MIT License"></a>
  <br/>
  <a href="https://github.com/GusCayresMindsight/nue-agentic-work"><img src="https://img.shields.io/badge/Repository-GitHub-000?logo=github&style=for-the-badge" alt="GitHub Repository"></a>
</div>

<hr>

## About nue

**nue** is a self-hosted, customizable fork of [OpenHands](https://github.com/OpenHands/OpenHands), an open-source AI agent framework for software development. This fork is designed for individual development with:

- **Full control**: Self-hosted, no external dependencies
- **Deep customization**: Modify frontend, backend, and agent orchestration
- **Clean infrastructure**: Removal of enterprise features and unnecessary integrations
- **Development-focused**: Purpose-built for iterating on AI agent capabilities

nue maintains compatibility with OpenHands while allowing independent evolution.

## Quick Start

### Prerequisites

- Python ≥3.12, <3.14
- Node.js ≥22.12.0
- Poetry ≥1.8.0
- Docker (for sandbox runtime)

### Setup

```bash
# Install pre-commit hooks (required)
make install-pre-commit-hooks

# Build the entire application (backend + frontend)
make build

# Run the application (backend on port 3000, frontend on port 3001+)
make run
```

Access the web UI at `http://localhost:3001`.

### Environment Variables

Key configuration options:

- `RUNTIME=docker` (default) — Run agents in isolated Docker containers
- `BACKEND_PORT=3000` — Backend server port
- `NUE_DISABLE_TELEMETRY=true` — Disable PostHog analytics (optional)
- `LLM_MODEL` — LLM model to use (e.g., `gpt-4o`, `claude-3-5-sonnet`)

See `openhands/app_server/server_config/server_config.py` for all available options.

## Development

### Project Structure

```
nue-agentic-work/
├── openhands/              # Python backend (AI agent framework)
├── frontend/               # React web UI
├── tests/                  # Test suite (pytest + vitest)
├── dev_config/             # Pre-commit and linting configuration
├── docker-compose.yml      # Local runtime definition
└── Makefile               # Central task runner
```

### Common Commands

```bash
# Run tests
poetry run pytest tests/unit/

# Lint backend (Python)
pre-commit run --config ./dev_config/python/.pre-commit-config.yaml

# Lint frontend (JavaScript/TypeScript)
cd frontend && npm run lint:fix && npm run build

# Rebuild after dependency changes
make build

# Start backend only
make start-backend

# Start frontend only (from frontend/ dir)
npm run dev
```

### Adding Features

- **Frontend changes**: Modify files in `frontend/src/`. Changes to the React app hot-reload during development.
- **Backend changes**: Modify files in `openhands/`. Backend requires restart to apply changes.
- **Tests**: Add pytest-bdd scenarios in `tests/bdd/` for behavioral test coverage.

### Configuration

Configuration directory: `~/.nue/` (auto-migrated from `~/.openhands/` on first run).

Database: SQLite by default (dev), PostgreSQL (production).

## Architecture

nue is built on OpenHands with the following modifications:

- **Enterprise directory removed**: Keycloak auth, Stripe billing, GitLab/Jira integrations, and custom integrations removed.
- **Telemetry gating**: PostHog analytics disabled by default (controlled via `NUE_DISABLE_TELEMETRY`).
- **V0 codebase cleaned**: Deprecated server/controller code removed.
- **Docker-first sandbox**: Agents run in isolated containers by default for safety and reproducibility.

## License

nue is MIT-licensed. This fork builds on [OpenHands](https://github.com/OpenHands/OpenHands), also MIT-licensed.

See [LICENSE](LICENSE) for full text.

## Attribution

**nue** is a fork of [OpenHands](https://github.com/OpenHands/OpenHands), maintained by the OpenHands community.

- **Upstream**: https://github.com/OpenHands/OpenHands
- **Fork base commit**: `15716df79` (upstream/main, May 2026)
- **License**: MIT (same as upstream)

**Modifications from upstream:**
- Enterprise directory removed (507 files, Keycloak/Stripe/GitLab/Jira/Slack/Linear integrations)
- Telemetry gating via `NUE_DISABLE_TELEMETRY` / `OPENHANDS_DISABLE_TELEMETRY`
- Surface rebrand to nue; configuration directory migrated to `~/.nue/`

nue incorporates open-source projects used by OpenHands, including
[SWE-Agent](https://github.com/princeton-nlp/swe-agent) (MIT),
[Aider](https://github.com/paul-gauthier/aider) (Apache 2.0),
[BrowserGym](https://github.com/ServiceNow/BrowserGym) (Apache 2.0),
and evaluation benchmarks (HumanEval, SWE-Bench, DSP, AgentBench, BIRD, and others — see
[OpenHands CREDITS](https://github.com/OpenHands/OpenHands/blob/main/CREDITS.md) for the full list).

Third-party dependency licenses are compatible with MIT. See `pyproject.toml` and `frontend/package.json`.

## Support

- **Issues**: [GitHub Issues](https://github.com/GusCayresMindsight/nue-agentic-work/issues)
- **Documentation**: See `AGENTS.md` and inline code documentation
- **OpenHands upstream**: For features or documentation from the original OpenHands project, see [OpenHands Docs](https://docs.openhands.dev/)

---

**Built on [OpenHands](https://github.com/OpenHands/OpenHands) — AI-driven development for everyone.**
