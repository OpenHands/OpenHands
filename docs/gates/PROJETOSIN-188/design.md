---
card: PROJETOSIN-188
pr: 7
veredicto: FAIL
agente: design
data: 2026-08-10
repo: klebersjunior/OpenHands
branch: feat/fase1-findings-ui-188
commit: 722351dcd
---

# Design Review — PROJETOSIN-188 Findings panel (`/findings`)

**Veredicto:** FAIL

Gate de UI/UX/a11y sobre o painel de Findings. Revisor (Design) ≠ autor do código de produção. Este laudo **não** cobre AC de QA nem AppSec.

Spec: `docs/specs/fase-1/188-design-notes.md` · `docs/specs/fase-1/188-findings-panel-ui.md`  
ADR: `docs/adrs/0001-plataforma-pentest-ia-extensao-openhands.md`

## Escopo revisado

| Superfície | Arquivo |
|---|---|
| Page shell / estados | `src/components/features/findings/findings-page.tsx` |
| Tabela + lista mobile | `findings-table.tsx` |
| Filtros | `findings-filters.tsx` |
| Stats / diff | `findings-diff-banner.tsx` |
| Empty / loading / error / forbidden | `findings-empty-state.tsx` |
| Drawer | `finding-detail-drawer.tsx` |
| Modal FP | `finding-fp-modal.tsx` |
| Row actions / badges | `findings-row-actions.tsx`, `finding-severity-badge.tsx` |
| Rota / sidebar | `src/routes/findings.tsx`, `sidebar-rail-body.tsx` |
| i18n | `FINDINGS$*` em `translation.json` / `declaration.ts` |

## O que está alinhado (não bloqueia)

- Rota `/findings`; query `engagement_id`; filtros refletidos na URL.
- Sidebar **Findings** após Customize, antes de Automations, com `CapabilityGate` + `data-testid="sidebar-findings-link"`.
- Hierarquia de estados: forbidden → no engagement → loading → error → empty → filtered-empty → table.
- `data-testid` estáveis pedidos nas notes (page, loading, empties, error, forbidden, stats, filters, table, rows, actions, drawer, FP modal).
- Tokens `--oh-*`; chips flat; empty via `extensionModuleEmptyStateClassName`; sem tema paralelo / purple clichê.
- Triage **oculto** sem `pentest.findings.triage`.
- FP exige reason (submit disabled + trim); Escape / overlay fecham modal e drawer.
- Evidence colapsável com `aria-expanded`; deep-link ao event stream só com ids.
- Mobile: lista compacta (além do FAIL de nesting abaixo).
- Sem `react-router` em `src/components/`.
- i18n `FINDINGS$*` presentes (72 keys no declaration).

## Checklist a11y

- [x] Contraste AA em badges (texto + tokens danger/primary/success/border)
- [ ] Foco visível / tab order — filtros sem `focus-visible` explícito; modal sem trap
- [ ] Operável por teclado — mobile: botão de ações aninhado em botão da row
- [x] Labels/ARIA — menu `aria-label`, drawer close, severity/status `aria-label`, fieldsets
- [x] Erros associados — FP `aria-invalid` + `aria-describedby` + `role="alert"`
- [x] Loading/erro/empty — testids + `aria-busy` / `role="status"` / `role="alert"`
- [x] Responsivo — lista compacta em `< md` (estrutura OK; nesting FAIL)

## Issues bloqueantes

| ID | Severidade | Issue | Ação para Frontend |
|---|---|---|---|
| **D-188-1** | **high** | Em `findings-table.tsx` (lista mobile), `FindingsRowActions` (contém `<button>`) é filho de um `<button>` da row. Controles interativos aninhados — HTML inválido e falha WCAG (teclado/AT). | Trocar o wrapper da row mobile para `<div role="button" tabIndex={0}>` **ou** separar: área clicável do detalhe + zona de ações **fora** do botão (padrão: card `div` + botão título + menu ao lado). Garantir `stopPropagation` nas ações. |
| **D-188-2** | **medium** | Modal FP é overlay custom **sem focus trap** e **sem restore de foco** ao trigger. Notes §5.5 / §8 / critério de gate §14.3 exigem trap (HeroUI Modal preferido) e devolver foco ao fechar. Tab pode sair do dialog; após Cancel/Submit o foco some. | Preferir HeroUI `Modal` (trap + restore nativos) **ou** trap manual (ciclo Tab no dialog) + guardar `document.activeElement` no open e `.focus()` no close. |

## Issues não bloqueantes (polish)

| ID | Severidade | Issue | Ação sugerida |
|---|---|---|---|
| D-188-3 | low | Chips/botões de filtro e “Só novos” sem `focus-visible:outline`/`ring` explícito (rows da tabela desktop já têm). | Alinhar `focus-visible:outline-2 outline-[var(--oh-color-primary)]` nos controles da toolbar/banner. |
| D-188-4 | low | No drawer, o título do finding é `h3`; notes pedem heading `h2` para o título do finding (o chrome usa `h2` “Finding detail”). | Usar `h2` no título do finding (ou `aria-labelledby` apontando para ele) no follow-up. |
| D-188-5 | low | Aceitar risco sem reason/modal leve — permitido no MVP das notes. | Opcional pós-MVP: ConfirmDialog + `FINDINGS$RISK_REASON_LABEL`. |

## Critérios de gate (§14 design notes)

| # | Critério | Resultado |
|---|---|---|
| 1 | Layout e estados (forbidden / no-engagement) | PASS |
| 2 | Checklist a11y §8 | **FAIL** (D-188-1, D-188-2) |
| 3 | FP reason + focus trap | **FAIL** (reason OK; trap/restore FAIL) |
| 4 | Triage oculta sem capability | PASS |
| 5 | Tokens `--oh-*` / HeroUI; sem tema paralelo | PASS (tabela HTML aceitável por notes) |
| 6 | i18n + `data-testid` | PASS |
| 7 | Sem `react-router` em components | PASS |

## Veredicto

**FAIL** — superfície e estados batem com o design; bloqueios de a11y em mobile (botão aninhado) e no modal FP (focus trap / restore). **Não liberar QA** até Frontend corrigir **D-188-1** e **D-188-2** e Design re-revisar.

### Remediação mínima (Frontend)

1. Corrigir nesting mobile em `findings-table.tsx` (D-188-1).
2. Modal FP com focus trap + restore de foco (D-188-2) — HeroUI Modal recomendado.
3. Smoke: Tab não escapa do modal; Escape fecha; após fechar o foco volta ao trigger `findings-action-mark-fp`; lista mobile: ações e abrir detalhe operáveis sem conflito.
4. Pedir re-gate Design no mesmo PR.
