# Agent Canvas — Testing Matrix

**Priority key:** P0 = must pass before any release · P1 = must pass before GA · P2 = best-effort

---

## Install × OS × Agent

Each cell = smoke test: install → onboard → start conversation → agent replies.

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

---

## Automations × Install × Agent

Requires full stack (automation backend running).

| | **npm** | **Docker** |
|---|:---:|:---:|
| **OpenHands** | ✅ P0 | ✅ P0 |
| **Claude Code** | ✅ P1 | ✅ P1 |
| **Codex** | ✅ P1 | ✅ P1 |
| **Gemini CLI** | ✅ P2 | ✅ P2 |

Each cell = create automation → dispatch run → run reaches COMPLETED → conversation link works.

---

## Auth Modes

| | **npm** | **Docker** |
|---|:---:|:---:|
| **Local (auto-generated key)** | ✅ P0 | ✅ P0 |
| **Public (`--public` + user key)** | ✅ P1 | ✅ P1 |

---

## Feature Checklist

### npm

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

### Docker

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

## Automated Coverage

| Suite | Install | OS | Agents | Automations |
|---|---|---|---|---|
| `vitest` (unit) | — | Linux | — | partial |
| `test:e2e:mock-llm` | npm | Linux | OpenHands, ACP (mock) | ✅ full |
| `test:e2e:mock-llm:docker` | Docker | Linux | OpenHands, ACP (mock) | ✅ full |
| `test:e2e:live` | npm | Linux | OpenHands | ❌ |

**Not yet covered by CI:** real ACP credentials (Claude Code / Codex / Gemini), macOS, public auth mode, subscription login paths, Windows.
