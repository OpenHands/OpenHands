---
card: PROJETOSIN-183
pr: 3
veredicto: PASS
agente: design
data: 2026-08-10
repo: klebersjunior/OpenHands
branch: feat/fase0-frontend-182-183
---

# Design Review — PROJETOSIN-183 Workspace Type Selector

**Veredicto:** PASS

Gate de UI/UX/a11y sobre o seletor de tipo de workspace, campos pentest, badge e integração nos fluxos de criação. Revisor (Design) ≠ autor funcional do laudo de QA/AppSec; este laudo **não** cobre AC de comportamento nem AppSec.

## Escopo revisado

| Superfície | Arquivo |
|---|---|
| `WorkspaceTypeSelector` | `src/components/features/pentest/workspace-type-selector.tsx` |
| `PentestWorkspaceFields` | `src/components/features/pentest/pentest-workspace-fields.tsx` |
| `WorkspaceTypeBadge` | `src/components/features/pentest/workspace-type-badge.tsx` |
| Home form | `src/components/features/home/workspace-selection-form.tsx` |
| Sidebar/menu local | `src/components/features/conversation-panel/local-new-conversation-menu.tsx` |
| Conversation card | `src/components/features/conversation-panel/conversation-card/conversation-card.tsx` |
| i18n | `WORKSPACE_TYPE$*` em `src/i18n/translation.json` |

Spec: `docs/specs/fase-0/183-workspace-type-selector.md` · ADR: `docs/adrs/0001-plataforma-pentest-ia-extensao-openhands.md`

## Checklist a11y

- [x] Contraste AA — texto/borda via tokens `--oh-foreground`, `--oh-text-secondary`, `--oh-primary`, `--oh-border`, `--oh-status-error`
- [x] Foco visível / tab order — `button`/`select` nativos sem `outline-none`; ordem lógica type → engagement → autonomia → launch
- [x] Operável por teclado — cards `aria-pressed`, chips de autonomia, `<select>` de engagement
- [x] Labels/ARIA — `role="group"` + `aria-label` no seletor; `<label>` no engagement; `<fieldset>`/`<legend>` na autonomia; ícones `aria-hidden`
- [x] Erros associados — `role="alert"` no erro de escopo; Create/Launch desabilitado quando bloqueado
- [x] Loading/erro — loading de engagements via `HOME$LOADING` no placeholder; erro de scope i18n
- [x] Responsivo — cards `flex-1` side-by-side; chips com `flex-wrap` (popover estreito: ver observação)

## Conformidade com design system / spec

- Tokens `--oh-*` (sem tema paralelo / purple clichê).
- Copy via `t(I18nKey.WORKSPACE_TYPE$…)` — sem magic strings de UI.
- Opção Pentest gateada por `CapabilityGate` (`pentest.workspace.create`) — alinhado a AC-183-1/2.
- Autonomia `autonomous` gateada por `pentest.autonomy.autonomous`.
- Badge pentest na conversation card e no header do form (AC-183-6).
- Seleção visual com borda/ring primária — adequado ao padrão do app (botões custom + tokens; HeroUI Card não é obrigatório aqui).

## Issues (não bloqueantes)

| ID | Severidade | Issue | Ação sugerida |
|---|---|---|---|
| D-183-1 | low | `TypeCard` / `AutonomyChip` / `<select>` não usam `focus-visible:ring` explícito com `--oh-focus` (padrão em outras features). Outline nativo permanece. | Opcional: alinhar `focus-visible:ring-1/2 ring-[var(--oh-focus)]` no follow-up. |
| D-183-2 | low | Lista de engagements vazia (não loading) não tem empty-state copy — só placeholder + Create disabled. | Adicionar chave i18n (ex. `WORKSPACE_TYPE$ENGAGEMENT_EMPTY`) quando EngMgr estiver live. |
| D-183-3 | low | `WORKSPACE_TYPE$PENTEST_UNAVAILABLE` existe mas não é usada (hide vs disabled+mensagem). Spec AC prefere hide. | Manter hide; usar a key como `fallback`/`title` se quiser feedback, ou limpar depois. |
| D-183-4 | low | Badge `text-[10px]` — legível o bastante com label textual, mas pequeno. | Aceitável para badge; não reduzir mais. |
| D-183-5 | low | No `LocalNewConversationMenu`, dois cards side-by-side + campos pentest apertam o popover em viewports estreitos. | Considerar stack vertical (`flex-col`) abaixo de ~sm no follow-up. |
| D-183-6 | low | Erro de escopo com `role="alert"` mas sem `aria-describedby` no `<select>`. | Opcional: ligar `aria-describedby` / `aria-invalid` quando `scopeError`. |

Nenhuma issue **medium/high** de a11y ou desvio de design system.

## Veredicto

**PASS** — UI alinhada à spec 183, tokens e i18n corretos, teclado/ARIA suficientes para WCAG 2.1 AA no escopo. Observações acima são polish, não bloqueiam merge do gate Design.

Próximos gates (fora deste laudo): QA · AppSec.
