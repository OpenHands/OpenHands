---
card: PROJETOSIN-192
pr: 8
veredicto: PASS
agente: design
data: 2026-08-10
repo: klebersjunior/OpenHands
branch: feat/fase2-emulator-ui-192
commit: 09141a484
re-gate: true
prev_veredicto: FAIL
prev_commit: af4d43431
---

# Design Review — PROJETOSIN-192 Emulator tab + APK upload (re-gate)

**Veredicto:** PASS

Re-gate de UI/UX/a11y após remediação FE dos bloqueios D-192-1 e D-192-2. Revisor (Design) ≠ autor do código de produção. Este laudo **não** cobre AC de QA nem AppSec.

Spec: `docs/specs/fase-2/192-design-notes.md` · `docs/specs/fase-2/192-emulator-ui-apk-upload.md`  
Laudo FAIL anterior: tip `af4d43431` · remediação tip `09141a484`

## Escopo reavaliado

| Superfície | Arquivo | Foco |
|---|---|---|
| Rail artifacts | `emulator-panel.tsx` | D-192-1 uncontrolled `defaultOpen` + remount |
| CTA / empty | `emulator-empty-state.tsx` | D-192-2 `focus-visible` + foco programático |
| Toolbar live | `emulator-toolbar.tsx` | D-192-2 `focus-visible` refresh |
| Smoke | upload / lista / i18n / estados | checklist anterior |

## Critérios de gate

| # | Critério | Resultado |
|---|---|---|
| 1 | Hierarquia: Stage/iframe dominante; Mobile artifacts rail secundário | **PASS** (D-192-1 resolvido) |
| 2 | Estados: unavailable / starting / live / error / upload / IPA reject | PASS |
| 3 | Capability gate + empty sem spinner infinito | PASS |
| 4 | a11y: foco CTA, iframe title, dropzone teclado, tokens | **PASS** (D-192-2 resolvido) |
| 5 | Sem cards decorativos / tema fora do DS | PASS |
| 6 | i18n keys (sem literals de UI) | PASS |

## Checklist a11y

- [x] Contraste AA — texto `var(--oh-muted)` / `var(--foreground)`; CTA claro/escuro
- [x] Foco visível / tab order — CTA start/retry e refresh com `focus-visible:outline`; dropzone OK
- [x] Operável por teclado — dropzone `<button>` + Enter/Space; disclosure nativo
- [x] Labels/ARIA — iframe `title` i18n; refresh `aria-label`; dropzone `aria-label`; progress/erro live regions
- [x] Erros associados — reject/upload via `aria-live="assertive"`
- [x] Loading/erro/empty — unavailable sem spinner; starting/loading com spinner curto
- [x] Responsivo / disclosure — `defaultOpen` + `key` live↔rest; usuário abre/fecha em qualquer fase

## Bloqueios remediados

| ID | Antes | Evidência tip `09141a484` | Resultado |
|---|---|---|---|
| **D-192-1** | `<details open={…}>` controlado sem `onToggle` — rail travado fechado em `live` | `defaultOpen={view.kind !== "live"}` + `key={railPhaseKey}` (`live` \| `rest`); uncontrolled; remount só na transição de fase (refetch 1,5s **não** remonta). Em `live` o usuário abre o summary → dropzone acessível. | **PASS** |
| **D-192-2** | CTA/refresh sem `focus-visible`; sem foco inicial em idle/error | Outline `focus-visible` em `emulator-empty-state` e `emulator-toolbar`; `useEffect` + `didFocusCtaRef` foca o CTA **uma vez** quando `canStart`; reset quando CTA some (ex. `starting`/`live`) — não rouba foco após live. | **PASS** |

## Smoke — restante do checklist anterior

- Capability / estados stage / iframe sandbox+title / dropzone IPA / lista plana / tokens `--oh-*` / `t(I18nKey.EMULATOR$…)` — inalterados e ainda OK.
- Polish não bloqueante (D-192-3/4/5: `window.confirm`, `text-red-400`, hint só em `title`) **permanece** — não impede PASS.

## Veredicto

**PASS** — Design desbloqueia. Tech Lead pode despachar **QA**. Não emitir QA/AppSec neste laudo. Não mergear.

Revisor: Design.
