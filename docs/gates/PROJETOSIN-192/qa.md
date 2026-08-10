---
card: PROJETOSIN-192
pr: 8
veredicto: FAIL
agente: qa
data: 2026-08-10
repo: klebersjunior/OpenHands
branch: feat/fase2-emulator-ui-192
commit: 09141a484
ci: vitest-emulator-pass; typecheck/lint-fail (CI ubuntu)
---

# QA Report — PROJETOSIN-192 Emulator UI + APK upload

**Veredicto:** FAIL  
**PR:** https://github.com/klebersjunior/OpenHands/pull/8  
**Tip avaliado:** `09141a484`  
**Worktree:** `.tmp/worktrees/192`  
**Revisor:** QA ≠ autor FE. Design PASS prévio (`docs/gates/PROJETOSIN-192/design.md`) **não** auto-assina AC.

Spec: `docs/specs/fase-2/192-emulator-ui-apk-upload.md` · Design notes: `docs/specs/fase-2/192-design-notes.md`

## Critérios de aceite

| AC | Status | Evidência |
|---|---|---|
| **AC-192-1** Start → iframe | **PASS** (comportamento Vitest) | `__tests__/…/emulator-panel.test.tsx` — `starts emulator and renders iframe`; `EmulatorService.start` ×1; `emulator-iframe` `src=/api/emulator/` |
| **AC-192-2** Sem capability → aba ausente | **PASS** (comportamento Vitest) | `conversation-tabs-emulator.test.tsx` — hide/show com `pentest.mobile.dynamic`; wiring em `conversation-tabs.tsx` / context-menu |
| **AC-192-3** Unavailable sem spinner infinito | **PASS** | Painel: `shows unavailable without start CTA`; empty state `unavailable` **sem** `LoadingSpinner` (`emulator-empty-state.tsx`); só `loading`/`starting` usam spinner |
| **AC-192-4** Upload APK + scan mockável | **PASS** | Upload chama `MobileArtifactsService.uploadApk`; lista `app.apk` com `scan_status: queued`; client retorna `mobsf_scan_id`/`scan_status` |
| **AC-192-5** IPA → rejeição, zero POST | **PASS** | Drop IPA → `EMULATOR$UPLOAD_REJECT_IPA`; `uploadApk` **not** called; `validateApkFile` unit |
| **AC-192-6** Proxy exige auth | **PASS** (unitário) | `emulator-proxy.test.ts` — `POST /api/emulator/start` sem key → **401** + detalhe `X-Session-API-Key` |
| **AC-192-7** i18n | **PASS** | `npm run check-translation-completeness` → complete; prefixo `EMULATOR$*` + `COMMON$EMULATOR`; UI via `t(I18nKey…)` |
| **AC-192-8** Vitest painel/empty/upload; sem `react-router` em components | **PASS*** / **bloqueado por regressão** | Suites Vitest verdes localmente; sem `react-router` em `src/components/features/emulator/`. \*Typecheck do mock de tabs quebrava CI — QA ajustou tipagem do mock no worktree (ver remediação). |
| **AC-192-9** Design notes + Design PASS | **PASS** | Notes presentes; laudo Design re-gate **PASS** @ `09141a484` |

\*AC comportamentais cobertos, mas o gate de regressão **não** fecha enquanto `npm run lint` (typecheck) falhar.

## Regressão

| Check | Resultado | Evidência |
|---|---|---|
| Vitest escopo emulator/tabs/proxy | **PASS** (15 tests) | `npx vitest run` nos 3 arquivos — exit 0 |
| `check-translation-completeness` | **PASS** | “All translation keys have complete language coverage!” |
| `npm run lint` / typecheck | **FAIL** | CI `test-and-build (ubuntu)` Lint @ run `31418268133`; local eslint escopo: `react/no-unknown-property` (`defaultOpen`), `no-param-reassign`, prettier |
| D-192-1 rail toggle em live | **PASS** (Vitest) | `allows opening artifacts rail while live` — summary click → `open` + dropzone visível |
| mock-LLM E2E | N/A mapping | Sem entrada `emulator` em `test-mapping.json`; não bloqueante vs AC cobertos por Vitest |
| AppSec | Não emitido | AC-192-6 evidenciado em unitário; AppSec formal fora deste gate |

### Typecheck / lint — falhas bloqueantes (produção)

1. **`emulator-panel.tsx`:** `defaultOpen` em `<details>` — TS2322 (`Property 'defaultOpen' does not exist`) + ESLint `react/no-unknown-property`. Remediação Design exigia uncontrolled **ou** `open`+estado+`onToggle`; a API React tipada não aceita `defaultOpen` em `HTMLDetailsElement`.
2. **`emulator-apk-upload.tsx`:** `no-param-reassign` em `event.target.value = ""`.
3. **Prettier** em vários arquivos do escopo (panel, apk-upload, mobile-artifacts-*, emulator-proxy, testes).

CI tip @ `09141a484` (pré-ajuste QA do mock):

- `__tests__/…/conversation-tabs-emulator.test.tsx` — mock `vi.fn(() => false)` sem arg (QA corrigiu tipagem `vi.fn<(cap: string) => boolean>` no worktree; **ainda não pushado** até FE remediação conjunta).
- `emulator-panel.tsx` `defaultOpen` — **permanece FAIL** até FE.

## Ação requerida (FE)

1. Substituir `defaultOpen` por padrão typecheck-safe que preserve D-192-1 (ex.: `open` + `onToggle` com reset só em `live↔rest`, ou remount + atributo DOM via ref). Revalidar teste D-192-1.
2. Corrigir `no-param-reassign` no input file change (cópia local / clear via ref).
3. `prettier --write` no escopo + confirmar `npm run lint` e CI ubuntu verdes.
4. Incluir no mesmo push: ajuste do mock AC-192-2 (já no worktree) + `docs/gates/PROJETOSIN-192/{design,qa}.md`.

**Após remediação:** Tech Lead redespacha **QA** (re-gate). **Não** despachar AppSec neste FAIL. **Não** mergear.

## Veredicto

**FAIL** — AC comportamentais Vitest/i18n/auth unitários OK; regressão **lint/typecheck/CI** vermelha bloqueia o gate.
