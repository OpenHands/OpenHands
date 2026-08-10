# Specs — Fase 3 Electron + device físico + autonomia UI (PROJETOSIN-181)

**ADR:** [0001](../../adrs/0001-plataforma-pentest-ia-extensao-openhands.md) (accepted)  
**Blueprint:** §4.2 Electron · §5.4 autonomia · §6.4 device físico — [blueprint](../../product/blueprint-plataforma-pentest-ia.md)  
**Base git:** fork `klebersjunior/OpenHands` tip `4eb43759b` (Fases 0–2 Done)

## Fora de escopo (Fase 3)

- Farm Corellium / Genymotion
- IPA / iOS
- Fase 4 orquestração avançada (multi-agent playbooks, farm remoto)

## Cards

| Card | Spec | Branch | Worktree | Agente(s) |
|------|------|--------|----------|-----------|
| PROJETOSIN-193 | [193-electron-docker-adb-ipc.md](./193-electron-docker-adb-ipc.md) | `feat/fase3-electron-ipc-193` | `.tmp/worktrees/193` | devops (lead) → backend |
| PROJETOSIN-194 | [194-physical-android-device.md](./194-physical-android-device.md) | `feat/fase3-physical-device-194` | `.tmp/worktrees/194` | backend → frontend (após contrato IPC 193) |
| PROJETOSIN-195 | [195-autonomy-modes-ui.md](./195-autonomy-modes-ui.md) · [195-design-notes.md](./195-design-notes.md) | `feat/fase3-autonomy-ui-195` | `.tmp/worktrees/195` | design → frontend |

## Paralelismo

```
193 DevOps (IPC foundation) ──► 193 Backend (bridge/allowlist helpers)
         │
         └── contrato estável ──► 194 (device físico + reconnect)
195 Design (autonomia UI) ──► 195 Frontend
```

- **Paralelo agora:** 193 (worktree isolada) · Design-195 (worktree isolada).
- **194:** começa implementação após handlers `adb:*` de 193 estáveis no PR/branch; stubs/contratos podem espelhar esta pasta de specs.
- **Frontend-195:** só após `195-design-notes.md` no worktree 195.
- PRs **somente** no fork `klebersjunior/OpenHands`. Gates: **AppSec crítico em 193** → Design (195) → QA → AppSec; **sem auto-assinatura**.

## Dependências cruzadas

```
ADR-0001 notes 1–3 ──► 193 IPC docker:compose:* + adb:*
Emulator UI Fase 2 (main) ──► 194 reusa aba Emulador / status patterns
EngMgr autonomy_mode + PENTEST_AUTONOMY_MODE (187 AppSec) ──► 195 UI + propagate env
```

## Segurança (lembrete global)

- Autonomia **server-side only** via `PENTEST_AUTONOMY_MODE` — **nunca** confiar `autonomy_mode` em args MCP do cliente/agente.
- Hooks Electron **só** com `PENTEST_ELECTRON_HOOKS_ENABLED=true` no modo local; modo servidor → desabilitados (403 / unavailable).
- IPC: allowlist de comandos/args; sem shell arbitrário; sem path traversal em compose files.

## CI (contexto)

- Polyfill `ProgressEvent` / ubuntu Test — já conhecido.
- mock-llm teste 124 flaky residual — não bloquear merge Fase 3 se flaky isolado documentado.
