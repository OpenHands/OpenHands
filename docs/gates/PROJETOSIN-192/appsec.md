---
card: PROJETOSIN-192
pr: 8
veredicto: PASS
agente: appsec
data: 2026-08-10
tip: b4aba549b
ci: npm-audit-high-clean; review manual emulator-proxy + mobile-artifacts-proxy + ingress/WS
repo: klebersjunior/OpenHands
branch: feat/fase2-emulator-ui-192
---

# AppSecurity — PROJETOSIN-192 (UI Emulador + upload APK)

**Veredicto:** PASS

**Revisor:** AppSec gate (não autor do código; não assina QA/Design). QA permanece em `docs/gates/PROJETOSIN-192/qa.md` (PASS). Design PASS prévio.

## Escopo

Spec `docs/specs/fase-2/192-emulator-ui-apk-upload.md` § Segurança + superfície:

- `scripts/emulator-proxy.mjs` (auth session/cookie, WS, upstream oculto)
- `scripts/mobile-artifacts-proxy.mjs` (auth, APK ext/tamanho, rejeição IPA)
- Wiring `scripts/ingress.mjs` / `scripts/static-server.mjs` (upgrade WS antes do router genérico)
- UI capability `pentest.mobile.dynamic`; sem paths absolutos de host na UI
- `docker/Dockerfile` COPY dos proxies
- `npm audit --audit-level=high`

Worktree `.tmp/worktrees/192` @ tip `b4aba549b` (inclui remediação Docker `647ac4ba7`). PR #8.

## Checklist

- [x] Sem segredos versionados / hardcoded nos proxies / Dockerfile COPY
- [x] `npm audit --audit-level=high` sem high/critical (4 moderate pré-existentes: dompurify/electron — fora do delta 192)
- [x] Session key: `POST /api/emulator/start` exige `X-Session-API-Key`; cookie HttpOnly `Path=/api/emulator`; WS upgrade autentica antes do proxy genérico
- [x] Upstream noVNC só server-side (`EMULATOR_NOVNC_URL` / `runtime_services`); resposta status/start devolve path same-origin `/api/emulator/…` — sem publish 5555/6901/URL interna ao browser
- [x] Mobile artifacts: session obrigatória; `.apk` only + `MAX_APK_BYTES` (200 MB); `.ipa` rejeitado por extensão; list/response sem path absoluto de host
- [x] Aba Emulador gated por `useHasPentestCapability("pentest.mobile.dynamic")`
- [x] Dockerfile COPY `emulator-proxy.mjs` + `mobile-artifacts-proxy.mjs` (sem secrets)

## Findings

### Critical / High

Nenhum. **Sem bloqueio.**

### MEDIUM — MIME do APK só no cliente; proxy valida extensão + tamanho

Spec pede extensão/**MIME**/tamanho. Servidor (`mobile-artifacts-proxy.mjs`) checa `/\.apk$/i` + `MAX_APK_BYTES`; não inspeciona `Content-Type` nem magic bytes ZIP/APK. Cliente (`validateApkFile`) cobre IPA/MIME parcial e tamanho.

**Decisão:** residual MEDIUM (MIME é spoofable; stub in-memory sem execução real). Aceitável no MVP; EngMgr definitivo deve reforçar (magic/ZIP + capability server-side).

### MEDIUM — Capability `pentest.mobile.dynamic` só na UI

Tabs/context-menu usam `useHasPentestCapability`; proxies exigem session mas **não** checam capability. Alinhado a residual documentado em gates anteriores (ex. SAST): enforcement na sessão/UI até EngMgr.

**Decisão:** residual MEDIUM — não FAIL neste card (auth session presente; superfície local single-key). Follow-up: AuthZ de capability no EngMgr / proxy quando RBAC multi-role for load-bearing.

### LOW — Cookie com session key raw (paridade Desktop)

`setAuthCookie` grava a session key (HttpOnly, Path-scoped, SameSite=Lax), igual `desktop-proxy.mjs`. Função `emulatorAuthToken` (hash) existe mas não é usada — legado/paridade. Sem flag `Secure` (stacks HTTP locais).

### LOW — Static assets anônimos + CORS reflect Origin (paridade Desktop)

Mesmo padrão noVNC/`crossorigin`: JS/CSS sob `/api/emulator/*` sem cookie; HTML/WS exigem auth. `access-control-allow-origin: reqOrigin || "*"` espelha desktop (sem `Allow-Credentials` no proxy HTTP).

### LOW — Comentário Dockerfile path

Comentário cita `/api/mobile/artifacts`; mount real é `/api/pentest/engagements`. Cosmético.

## Controles verificados (mapeamento § Segurança)

| Controle | Evidência |
|---|---|
| Cookie HttpOnly path-scoped | `EMULATOR_AUTH_COOKIE` + `Path=/api/emulator`; `HttpOnly`; `SameSite=Lax` |
| WS auth antes do router genérico | `ingress.mjs` / `static-server.mjs` `upgrade`: desktop → emulator → `route()` genérico; `handleUpgrade` 401 sem sessão |
| Sem publish VNC host | `resolveEmulatorUpstreamUrl` só no server; JSON `url` = `EMULATOR_IFRAME_PATH` |
| Extensão / tamanho APK; IPA fora | `basenameOnly` + `/\.apk$/i`; 413 se > 200 MB; cliente rejeita IPA com zero POST |
| Paths absolutos | API stub `path: mobile/{engagementId}/{name}`; UI lista só `filename` |
| Capability UI | `conversation-tabs.tsx` / context-menu: `canUseEmulator` |
| Docker COPY | `647ac4ba7` — proxies no image layer |

## Dependências

`npm audit --audit-level=high`: **PASS** (0 high/critical). Moderates pré-existentes (dompurify via monaco/posthog; electron) — fora do delta 192.

## Ação requerida

Nenhuma para merge AppSec. Residuais MEDIUM → backlog EngMgr (MIME/magic + capability server-side) quando 190/191 saírem do stub.

**Não mergeado por AppSec.** Tech Lead decide merge com Design+QA+AppSec PASS.
