---
card: PROJETOSIN-188
pr: 7
veredicto: PASS
agente: design
data: 2026-08-10
repo: klebersjunior/OpenHands
branch: feat/fase1-findings-ui-188
commit: b85c62c8e
re-gate: true
previous: FAIL @ 01a05d51f / 722351dcd
---

# Design Review — PROJETOSIN-188 Findings panel (`/findings`)

**Veredicto:** PASS

Re-gate de UI/UX/a11y após remediação Frontend de **D-188-1** e **D-188-2** (`b85c62c8e`). Revisor (Design) ≠ autor do código de produção. Este laudo **não** cobre AC de QA nem AppSec.

Spec: `docs/specs/fase-1/188-design-notes.md` · `docs/specs/fase-1/188-findings-panel-ui.md`  
ADR: `docs/adrs/0001-plataforma-pentest-ia-extensao-openhands.md`

## Escopo reavaliado

| Superfície | Arquivo | Foco |
|---|---|---|
| Lista mobile | `findings-table.tsx` | D-188-1 nesting |
| Modal FP | `finding-fp-modal.tsx` | D-188-2 trap/restore |
| Row actions | `findings-row-actions.tsx` | stopPropagation + focus no trigger FP |
| Regressão | page / filters / drawer / empties / badges / sidebar | o que já estava OK no FAIL |

## Remediações

| ID | Status | Evidência |
|---|---|---|
| **D-188-1** | **PASS** | Lista mobile: card `div` + `button` só na zona de detalhe; `FindingsRowActions` é **irmão** (fora do botão). Teste Vitest: `keeps mobile row actions outside the detail button`. |
| **D-188-2** | **PASS** | FP usa HeroUI `Modal`/`ModalContent` (focus trap nativo) + captura/`useLayoutEffect` restore ao `document.activeElement` do open; menu FP refoca o trigger antes do open. Teste: `restores focus to the trigger when the FP modal closes`. |

## Checklist a11y

- [x] Contraste AA em badges (texto + tokens danger/primary/success/border)
- [x] Foco visível / tab order — rows desktop + botão detalhe mobile com `focus-visible`; modal HeroUI com trap; restore no close (filtros sem ring explícito = polish D-188-3)
- [x] Operável por teclado — sem botão aninhado; ações com `stopPropagation`; Escape no menu/modal
- [x] Labels/ARIA — menu `aria-label`, drawer close, severity/status `aria-label`, fieldsets, modal `aria-labelledby`
- [x] Erros associados — FP `aria-invalid` + `aria-describedby` + `role="alert"`
- [x] Loading/erro/empty — testids + `aria-busy` / `role="status"` / `role="alert"`
- [x] Responsivo — lista compacta `< md` com detalhe e ações separados

## Regressão (já OK no FAIL anterior)

- Rota `/findings`; query `engagement_id`; filtros na URL.
- Sidebar Findings com capability gate + `data-testid="sidebar-findings-link"`.
- Hierarquia: forbidden → no engagement → loading → error → empty → filtered-empty → table.
- Triage oculto sem `pentest.findings.triage`.
- FP reason obrigatória (submit disabled + trim).
- Tokens `--oh-*`; empty states; sem tema paralelo.
- Evidence colapsável; deep-link condicional.
- Sem `react-router` em `src/components/`.
- i18n `FINDINGS$*` + testids estáveis.

## Issues não bloqueantes (polish — inalterados)

| ID | Severidade | Issue |
|---|---|---|
| D-188-3 | low | Chips/filtros sem `focus-visible` explícito alinhado às rows. |
| D-188-4 | low | Título do finding no drawer ainda `h3` vs `h2` das notes. |
| D-188-5 | low | Aceitar risco sem ConfirmDialog — OK no MVP. |

## Critérios de gate (§14 design notes)

| # | Critério | Resultado |
|---|---|---|
| 1 | Layout e estados (forbidden / no-engagement) | PASS |
| 2 | Checklist a11y §8 | **PASS** (D-188-1/2 fechados) |
| 3 | FP reason + focus trap | **PASS** |
| 4 | Triage oculta sem capability | PASS |
| 5 | Tokens `--oh-*` / HeroUI; sem tema paralelo | PASS |
| 6 | i18n + `data-testid` | PASS |
| 7 | Sem `react-router` em components | PASS |

## Veredicto

**PASS** — bloqueios de a11y remediados; superfície e estados sem regressão. **Liberar QA.** Polish D-188-3/4/5 opcional pós-MVP, não bloqueia.

Revisor: Design. Não auto-assina QA nem AppSec.
