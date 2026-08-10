---
card: PROJETOSIN-183
pr: 3
veredicto: PASS
agente: qa
data: 2026-08-10
ci: vitest-pentest-subset
repo: klebersjunior/OpenHands
branch: feat/fase0-frontend-182-183
---

# QA Report — PROJETOSIN-183 Workspace Type Selector

**Veredicto:** PASS

**PR:** https://github.com/klebersjunior/OpenHands/pull/3  
**Spec:** `docs/specs/fase-0/183-workspace-type-selector.md`  
**Design gate:** PASS (`docs/gates/PROJETOSIN-183/design.md`)  
**AppSec:** não emitido neste gate (outro agente).  
**Depends on:** PROJETOSIN-182 CapabilityGate (FE PASS neste PR).

## Critérios de aceite

| AC | Status | Evidência |
|---|---|---|
| AC-183-1 — sem `pentest.workspace.create` → sem opção Pentest | PASS | `__tests__/components/pentest/workspace-type-selector.test.tsx` — `queryByTestId("workspace-type-pentest")` ausente |
| AC-183-2 — com capability → Pentest visível e selecionável | PASS | mesmo arquivo — click dispara `onChange("pentest")` |
| AC-183-3 — Pentest sem engagement → criar bloqueado | PASS | `__tests__/components/pentest/pentest-creation-validation.test.ts` (`blocks pentest without engagement`); form usa `isPentestCreationBlocked` em `isDisabled` do launch (`workspace-selection-form.tsx`) |
| AC-183-4 — engagement sem `scope_authorized_at` → erro | PASS | mesmo validation test (`flags unauthorized scope`); UI passa `scopeError` via `hasUnauthorizedScope` → `WORKSPACE_TYPE$SCOPE_UNAUTHORIZED` |
| AC-183-5 — metadata `workspace_type: "pentest"` | PASS | `__tests__/api/conversation-metadata-pentest.test.ts`; create conversation grava via `use-create-conversation.ts` |
| AC-183-6 — badge/ícone distinto | PASS | componente `workspace-type-badge.tsx` + teste `__tests__/components/pentest/workspace-type-badge.test.tsx` (`data-testid="workspace-type-pentest-badge"`); wired em form header e `conversation-card.tsx` |

## i18n / integração

- Keys `WORKSPACE_TYPE$*` presentes em `src/i18n/translation.json` (selector, badge, engagement, autonomy, scope unauthorized).
- Integração: `WorkspaceSelectionForm` + `local-new-conversation-menu.tsx` + metadata store.

## Regressão

```bash
npx vitest run \
  __tests__/hooks/use-pentest-capabilities.test.tsx \
  __tests__/components/pentest/ \
  __tests__/api/conversation-metadata-pentest.test.ts \
  src/api/pentest-service/pentest-service.api.test.ts
```

**Resultado:** PASS (19) FAIL (0)

## Ação requerida

Nenhuma. Merge ainda aguarda AppSec gate (não emitido aqui) + Tech Lead.
