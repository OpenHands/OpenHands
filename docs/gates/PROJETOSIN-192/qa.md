---
card: PROJETOSIN-192
pr: 8
veredicto: PASS
agente: qa
data: 2026-08-10
repo: klebersjunior/OpenHands
branch: feat/fase2-emulator-ui-192
commit: 647ac4ba7
ci: lint-test-build-ubuntu-pass; windows-pass; mock-llm-1-flake-settings
re-gate: true
prev_veredicto: FAIL
prev_commit: cf4d96bd7
---

# QA Report — PROJETOSIN-192 Emulator UI + APK upload (re-gate #2)

**Veredicto:** PASS  
**PR:** https://github.com/klebersjunior/OpenHands/pull/8  
**Tip avaliado:** `647ac4ba7793861f336426f1ab8b11927d2e8c46`  
**Worktree:** `.tmp/worktrees/192`  
**Revisor:** QA ≠ autor FE. Design PASS **não** auto-assina AC. Evidência própria.

Spec: `docs/specs/fase-2/192-emulator-ui-apk-upload.md`  
Remediação tip `647ac4ba7`: `COPY` de `emulator-proxy.mjs` + `mobile-artifacts-proxy.mjs` no Dockerfile; `QueryClientProvider` em `conversation-tabs-context-menu.test.tsx`.

## Critérios de aceite

| AC | Status | Evidência |
|---|---|---|
| **AC-192-1** Start → iframe | **PASS** | `emulator-panel.test.tsx` — start + iframe |
| **AC-192-2** Sem capability → aba ausente | **PASS** | `conversation-tabs-emulator.test.tsx`; context-menu suite verde com QueryClient |
| **AC-192-3** Unavailable sem spinner infinito | **PASS** | empty unavailable sem spinner infinito |
| **AC-192-4** Upload APK + scan | **PASS** | upload + lista artifact |
| **AC-192-5** IPA rejeitado | **PASS** | zero POST |
| **AC-192-6** Proxy exige auth | **PASS** | unit 401; Docker COPY dos proxies presente (`proxy-script-copies` PASS) |
| **AC-192-7** i18n | **PASS** | `check-translation-completeness` complete |
| **AC-192-8** Vitest painel/tabs/proxy | **PASS** | emulator subset **15/15**; bloqueios regressão **9/9**; combined **24/24** |
| **AC-192-9** Design + D-192-1 | **PASS** | Design gate PASS; rail live toggle Vitest OK |

## Regressão (bloqueios do FAIL anterior)

| Check | Resultado | Evidência |
|---|---|---|
| `docker/Dockerfile` COPY `emulator-proxy.mjs` + `mobile-artifacts-proxy.mjs` | **PASS** | linhas 280–282; imports em `static-server.mjs` / `ingress.mjs` |
| `__tests__/docker/proxy-script-copies.test.ts` | **PASS** | incluso no run 9/9 e 24/24 |
| `conversation-tabs-context-menu.test.tsx` | **PASS** (8) | wrap `QueryClientProvider`; sem `No QueryClient set` |
| Vitest emulator + tabs-emulator + emulator-proxy | **PASS** (15) | exit 0 |
| typecheck (`tsc --noEmit`) | **PASS** | exit 0 |
| eslint escopo emulator + conversation-tabs | **PASS** | no issues |
| `check-translation-completeness` | **PASS** | complete |
| CI `test-and-build (ubuntu)` | **PASS** | run `31421551843` — Lint + Test + Build |
| CI `test-and-build (windows)` | **PASS** | mesmo run |

### mock-LLM E2E (nota — não bloqueia este re-gate)

- Run `31421550444` @ tip `647ac4ba7`: **59 passed / 1 failed** (suite full por `scripts/**` em `runAllSources`).
- Falha: `settings/mock-llm-profile-management.spec.ts` — `"deletion-guard-inactive" should become active after deleting "deletion-guard-active"` (timeout 15s).
- Tip anterior `cf4d96bd7` (mesmo PR): mock-LLM **SUCCESS**.
- Diff `647ac4ba7` vs pai: **somente** `docker/Dockerfile` + `conversation-tabs-context-menu.test.tsx` — sem mudança em settings UI/API. Falha tratada como **flake fora do escopo AC-192**; não reabre bloqueios Docker/context-menu.

## Remediação vs FAIL `cf4d96bd7`

1. ~~Falta COPY `emulator-proxy.mjs`~~ → COPY + `mobile-artifacts-proxy.mjs`; `proxy-script-copies` verde.
2. ~~Context-menu `No QueryClient set`~~ → `QueryClientProvider` no helper de render; 8/8 PASS.
3. CI ubuntu Lint/Test/Build verde no tip sob avaliação.

## Veredicto

**PASS** — AC-192-* verdes; bloqueios Docker COPY + context-menu remediados com evidência própria; CI ubuntu PASS.  
**Próximo:** Tech Lead despacha **AppSec**. QA não mergeia. Remover label `Blocked`.
