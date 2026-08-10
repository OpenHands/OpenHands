# Spec Técnica — PROJETOSIN-192: UI Emulador + upload APK

**ADR:** docs/adrs/0001-plataforma-pentest-ia-extensao-openhands.md (accepted)  
**Card Plane:** PROJETOSIN-192 — `be7bc815-6c37-4e66-a8d6-7994efab19c7`  
**Agentes:** design (primeiro) → frontend  
**Prioridade:** P1 high — aba Emulador + upload APK → MobSF  
**Base git:** `dbfa75033`  
**Branch:** `feat/fase2-emulator-ui-192`  
**Worktree:** `.tmp/worktrees/192`  
**PR target:** fork `klebersjunior/OpenHands` only

---

## Objetivo

1. Aba **Emulador** no painel direito da conversa (workspace pentest / capability mobile), espelhando o padrão **Desktop VNC** (`docs/spec-browser-desktop.md` / `scripts/desktop-proxy.mjs`).
2. Fluxo de **upload de APK** (AAB opcional MVP+; **IPA fora de escopo**) para o bucket/volume do engagement, disparando análise estática MobSF e opcionalmente `adb install`.
3. Empty states claros quando compose mobile/emulador indisponível (dev sem Docker, profile ≠ mobile).

---

## Ordem de trabalho

1. **Design** — `docs/specs/fase-2/192-design-notes.md` no mesmo worktree: layout aba, estados (loading / unavailable / live), upload dropzone, a11y, tokens HeroUI. Gate UI só após Frontend.
2. **Frontend** — implementar conforme design notes + esta spec.
3. **Backend mínimo no mesmo PR se necessário** — proxy `/api/emulator` espelhando desktop-proxy (pode ser devops/backend no mesmo worktree **só** o necessário para a aba; não reabrir 191).

**Frontend não inicia** até Design notes existirem e Tech Lead/PM confirmarem no card.

---

## Capability e visibilidade

| Superfície | Capability |
|---|---|
| Aba Emulador + upload | `pentest.mobile.dynamic` |
| Sem capability | aba ocultada (mesmo padrão Desktop/Findings) |
| Fora de Docker / sem sidecar | aba visível com empty state `EMULATOR$UNAVAILABLE` (não crash) |

---

## Aba Emulador — padrão Desktop

Espelhar:

| Desktop | Emulador |
|---|---|
| `POST /api/desktop/start` | `POST /api/emulator/start` |
| Cookie `agent-canvas-desktop-auth` Path=`/api/desktop` | Cookie `agent-canvas-emulator-auth` Path=`/api/emulator` |
| iframe `/api/desktop/` | iframe `/api/emulator/` |
| `scripts/desktop-proxy.mjs` | `scripts/emulator-proxy.mjs` (novo) |
| Porta loopback KasmVNC no agent-canvas | Upstream = noVNC do **container emulator do engagement** (URL interna resolvida via EngMgr/runtime_services ou env) |

### Resolução do upstream

Ordem de resolução (documentar no código):

1. `runtime_services.services.android_emulator.url_from_agent` (se EngMgr/ingress anunciar — coordenar 191).
2. Env `EMULATOR_NOVNC_URL` / `VITE_` não expor URL interna no browser — **só** same-origin `/api/emulator/`.
3. Fallback: empty state unavailable.

Auth: validar `X-Session-API-Key` (ou cookie pós-`/start`) como no desktop-proxy. **Nunca** publicar 5555/6901 no host sem proxy.

### UI

```
src/routes/… (tab wiring existente conversation-tabs)
src/components/features/emulator/
  emulator-panel.tsx
  emulator-empty-state.tsx
  emulator-toolbar.tsx          # start / refresh / open external (optional)
src/api/integrations/emulator-service.ts
```

i18n: prefixo `EMULATOR$…` em `translation.json` + `make-i18n`.  
Test ids: `emulator-panel`, `emulator-unavailable`, `emulator-iframe`, `emulator-start-button`.

---

## Upload APK

### UX

- Controlo na aba Emulador **ou** seção “Mobile artifacts” no mesmo painel (Design decide).
- Aceitar `.apk` (MIME `application/vnd.android.package-archive`); rejeitar `.ipa` com mensagem i18n.
- Após upload:
  1. Persistir no storage do engagement (path sob workspace / API EngMgr se existir; MVP: `POST` multipart para endpoint local).
  2. Disparar scan estático MobSF (via API proxy ou enqueue que o mcp-mobile / eng service processa).
  3. Toggle opcional “Instalar no emulador” → confirmação (semi-autonomous) / chama install.

### API client (frontend)

```
src/api/pentest/
  mobile-artifacts-service.ts   # upload + status
  mobile-artifacts-types.ts
```

**Proibido:** fetch cru ao Agent Server; Cloud → `callCloudProxy`. Serviço pentest local: session key + path via ingress.

Contrato MVP sugerido (implementar proxy se ausente):

| Método | Path | Notas |
|---|---|---|
| `POST` | `/api/pentest/engagements/{id}/mobile/apk` | multipart `file`; retorna `{ artifact_id, path, mobsf_scan_id? }` |
| `GET` | `/api/pentest/engagements/{id}/mobile/artifacts` | lista |
| `POST` | `/api/pentest/engagements/{id}/mobile/artifacts/{aid}/install` | adb install (gate) |

Se EngMgr ainda não tiver rotas: stub no ingress que grava em volume + chama MobSF REST com key de servidor (coordenar backend no worktree 192 **mínimo**, ou consumir 190 via documentado env). Preferir **não** duplicar lógica MobSF no frontend.

Query keys: `MOBILE_ARTIFACTS_QUERY_KEYS` em `query-keys.ts`.

---

## Design notes (obrigatório)

Arquivo: `docs/specs/fase-2/192-design-notes.md`

Cobrir:

- Hierarquia visual no painel direito (iframe dominante; upload secundário)
- Estados: unavailable, starting, live, error, upload progress, scan queued
- a11y: foco no CTA start; iframe `title` i18n; dropzone keyboard
- Sem cards desnecessários; HeroUI/tokens existentes
- Mobile viewport: painel empilhado

---

## Critérios de aceite (QA)

1. **AC-192-1:** Com capability + Docker/emulador anunciado, “Abrir Emulador” mostra iframe interativo (ou mock E2E com route).
2. **AC-192-2:** Sem capability → aba ausente.
3. **AC-192-3:** Sem emulator → empty state i18n, sem spinner infinito.
4. **AC-192-4:** Upload `.apk` válido → artifact listado + chamada de scan (mockavel).
5. **AC-192-5:** Upload `.ipa` → rejeição com mensagem; zero POST.
6. **AC-192-6:** Proxy `/api/emulator` exige auth (AppSec).
7. **AC-192-7:** i18n completo (`check-translation-completeness`); sem literal strings UI.
8. **AC-192-8:** Vitest painel/empty/upload; sem `react-router` em `src/components/`.
9. **AC-192-9:** Design notes presentes antes do merge FE; gate Design PASS.

---

## Segurança (AppSec)

- Mesmo modelo do desktop-proxy (cookie HttpOnly path-scoped; WS auth antes do router genérico).
- Validar extensão/MIME/tamanho máximo APK (ex. 200 MB config).
- Não refletir paths absolutos do host na UI.
- Scan/install só com session autenticada + capability.

---

## Fora de escopo

- Device físico / scrcpy host / Electron ADB (Fase 3)
- IPA/iOS
- Farm externo
- Gravação de vídeo como evidência (MVP+)

---

## Dependências

- **Depende de (contrato):** 191 (serviço emulator na rede), 190 (MobSF/findings) — FE pode mockar.
- **Padrão:** Desktop tab + desktop-proxy.
- **Paralelo inicial:** só Design; FE após notes.

**Estimativa:** Design 1–2 d · Frontend 3–4 d
