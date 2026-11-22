# Agent Canvas — Testing Matrix

**Priority key:** P0 = must pass before any release · P1 = must pass before GA · P2 = best-effort

This document is split into two sections:

1. **Automated in CI** — suites that run on pull requests or merges, with the workflow file that executes each. These claims are verifiable and should be kept in sync with `.github/workflows/`.
2. **Manual QA checklist** — checks that a human runs before a release, with no automation behind them.

---

## Automated in CI

| Suite | Trigger | Workflow | What it covers |
|---|---|---|---|
| `vitest` (unit) | PR / push | `.github/workflows/ci.yml` → `ci` job | Unit tests under `__tests__/`, jsdom + node projects |
| `vitest` (script tests) | PRs / pushes touching `.github/scripts/**` | `.github/workflows/ci-script-tests.yml` | Docker entrypoint, download, session-key, and other script contracts |
| `test:e2e:mock-llm` | Push to `main`, the release-please PR, or manual dispatch | `.github/workflows/mock-llm-e2e.yml` | Playwright mock-LLM E2E against local agent-server (OpenHands + ACP agents) |
| `test:e2e:mock-llm:docker` | After a successful Docker build on `main`, or manual dispatch | `.github/workflows/mock-llm-docker-e2e.yml` | Playwright mock-LLM E2E using the Docker-built image |
| `test:e2e:live` | PR / push | `.github/workflows/ci.yml` | Live agent-server E2E (npm, Linux, OpenHands agent only) |
| Docker image build | PR to `main` / push to `main` or `v*` tags | `.github/workflows/docker.yml` | Builds the agent-canvas Docker image |
| Desktop: Linux | PRs touching `electron/**` (path-filtered), manual dispatch, release published | `.github/workflows/desktop-linux.yml` | Builds the Linux desktop package (no test step) |
| Desktop: macOS | PRs touching `electron/**` (path-filtered), manual dispatch, release published | `.github/workflows/desktop-macos.yml` | Builds the macOS desktop package (no test step) |
| Desktop: Windows | PRs touching `electron/**` (path-filtered), manual dispatch, release published | `.github/workflows/desktop-windows.yml` | Builds the Windows desktop package (no test step) |
| Windows install E2E | The release-please PR (`.release-please-manifest.json`), or manual dispatch | `.github/workflows/windows-install-e2e.yml` | Packs an npm tarball from the commit, installs it globally on Windows, and runs the first-time-user smoke test |
| Publish smoke | After a real npm publication (tag push or dispatch) | `.github/workflows/npm-publish.yml` → `publish-smoke` job | Installs the published artifact from the registry on Linux, boots it with a clean temp HOME, polls `/server_info`, and checks the generated session key |
| Verify skill isolation | PR / push | `.github/workflows/verify-skill-isolation.yml` | Ensures `verify-*` skills are not modified in fix PRs |

### Coverage claims (corrected)

| Platform / mode | Covered by CI? | How |
|---|---|---|
| macOS | ✅ | `desktop-macos.yml` builds the macOS package on path-filtered PRs and releases |
| Linux | ✅ | `ci.yml`, `mock-llm-e2e.yml`, `docker.yml`, `desktop-linux.yml` |
| Windows | ✅ (build + install) | `desktop-windows.yml` builds the package; `windows-install-e2e.yml` installs a packed tarball on the release-please PR |
| Public auth mode | ❌ | Manual QA only — no CI workflow exercises `--public` / `playwright.bind-policy.config.ts` |
| Real ACP credentials (Claude Code / Codex / Gemini) | ❌ | Manual QA only — requires live credentials |
| Subscription login paths | ❌ | Manual QA only — requires paid accounts |
| Post-publication npm install (macOS) | ❌ | Manual QA only — `npm-publish.yml` smoke-tests the published artifact on Linux only; macOS install of the published package is untested |

**Previously stale claims corrected:** The old matrix listed macOS and Windows as "not yet covered by CI"; both are now exercised by the desktop workflows and the Windows install E2E listed above.

---

## Manual QA Checklist

These checks have no CI automation and must be run by a human before a release.

### Install × OS × Agent (smoke test)

Each cell = install → onboard → start conversation → agent replies.

| | **macOS** | **Linux** | **Windows** |
|---|:---:|:---:|:---:|
| **npm — OpenHands** | ☐ | ☐ | ☐ |
| **npm — Claude Code** | ☐ | ☐ | ☐ |
| **npm — Codex** | ☐ | ☐ | ☐ |
| **npm — Gemini CLI** | ☐ | ☐ | ☐ |
| **npm — Custom ACP** | ☐ | ☐ | ☐ |
| **Docker — OpenHands** | ☐ | ☐ | ☐ |
| **Docker — Claude Code** | ☐ | ☐ | ☐ |
| **Docker — Codex** | ☐ | ☐ | ☐ |
| **Docker — Gemini CLI** | ☐ | ☐ | ☐ |
| **Docker — Custom ACP** | ☐ | ☐ | ☐ |

### Automations × Install × Agent

Requires full stack (automation backend running). Each cell = create automation → dispatch run → run reaches COMPLETED → conversation link works.

| | **npm** | **Docker** |
|---|:---:|:---:|
| **OpenHands** | ✅ P0 | ✅ P0 |
| **Claude Code** | ✅ P1 | ✅ P1 |
| **Codex** | ✅ P1 | ✅ P1 |
| **Gemini CLI** | ✅ P2 | ✅ P2 |

### Auth Modes

| | **npm** | **Docker** |
|---|---|---|
| **Local (auto-generated key)** | ✅ P0 | ✅ P0 |
| **Public (`--public` + user key)** | ✅ P1 | ✅ P1 |

### Feature Checklist

#### npm

| Feature | OpenHands | Claude Code | Codex | Gemini CLI |
|---|:---:|:---:|:---:|:---:|
| Onboarding | ☐ | ☐ | ☐ | ☐ |
| Conversation — start, resume, history | ☐ | ☐ | ☐ | ☐ |
| Terminal tool | ☐ | ☐ | ☐ | ☐ |
| File editor tool | ☐ | ☐ | ☐ | ☐ |
| Browser tool | ☐ | ☐ | ☐ | ☐ |
| LLM profiles — create / switch | ☐ | — | — | — |
| Secrets — add / delete / forwarded | ☐ | ☐ | ☐ | ☐ |
| Automations — create, dispatch, COMPLETED | ☐ | ☐ | ☐ | ☐ |
| Files tab + Changes/diff tab | ☐ | ☐ | ☐ | ☐ |
| MCP server install | ☐ | ☐ | ☐ | ☐ |
| Image upload in chat | ☐ | ☐ | ☐ | ☐ |
| Key rotation | ☐ | ☐ | ☐ | ☐ |

#### Docker

| Feature | OpenHands | Claude Code | Codex | Gemini CLI |
|---|:---:|:---:|:---:|:---:|
| Onboarding | ☐ | ☐ | ☐ | ☐ |
| Conversation — start, resume, history | ☐ | ☐ | ☐ | ☐ |
| Terminal tool | ☐ | ☐ | ☐ | ☐ |
| File editor tool | ☐ | ☐ | ☐ | ☐ |
| Browser tool | ☐ | ☐ | ☐ | ☐ |
| LLM profiles — create / switch | ☐ | — | — | — |
| Secrets — add / delete / forwarded | ☐ | ☐ | ☐ | ☐ |
| Automations — create, dispatch, COMPLETED | ☐ | ☐ | ☐ | ☐ |
| Files tab + Changes/diff tab | ☐ | ☐ | ☐ | ☐ |
| MCP server install | ☐ | ☐ | ☐ | ☐ |
| Image upload in chat | ☐ | ☐ | ☐ | ☐ |
| Key rotation | ☐ | ☐ | ☐ | ☐ |

---

## Not covered by CI (manual only)

- Public auth mode (`--public` + user key) — no CI workflow exercises it (`playwright.bind-policy.config.ts` is not wired into any workflow)
- Real ACP credentials (Claude Code / Codex / Gemini) — requires live provider credentials
- Subscription login paths — requires paid accounts
