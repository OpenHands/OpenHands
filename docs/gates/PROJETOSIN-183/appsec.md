---
card: PROJETOSIN-183
pr: 3
veredicto: PASS
agente: appsec
data: 2026-08-10
ci: npm-audit-high
repo: klebersjunior/OpenHands
branch: feat/fase0-frontend-182-183
escopo: frontend
---

# AppSecurity Report — PROJETOSIN-183 Workspace Type Selector (FE)

**Veredicto:** PASS

**PR:** https://github.com/klebersjunior/OpenHands/pull/3  
**Worktree:** `OpenHands-wt-frontend` (`feat/fase0-frontend-182-183`)  
**Spec:** `docs/specs/fase-0/183-workspace-type-selector.md`  
**Depends on:** PROJETOSIN-182 CapabilityGate (FE AppSec PASS neste PR)

## Resumo

Seletor código/pentest gated por `pentest.workspace.create`; campos de engagement/autonomia com validação client-side de escopo; metadata pentest persistida só em `localStorage` via `conversation-metadata-store` (não encaminhada como AuthZ ao `createConversation` do agent-server). Sem XSS (`dangerouslySetInnerHTML` ausente; nomes de engagement em nós de texto React). Sem segredos; `npm audit` sem high/critical.

## Checklist

- [x] Sem segredos versionados / hardcoded
- [x] `npm audit --audit-level=high` sem high/critical
- [x] Session key / modo público: inalterado por este card
- [x] Proxies/VNC: não tocados
- [x] Cloud só via callCloudProxy: N/A (sem rotas cloud novas)
- [x] Logs sem secrets

## Findings

| ID | Severidade | Título | Detalhe | Ação |
|---|---|---|---|---|
| AS-FE-183-1 | **Medium** (residual) | Validação de escopo só no client | `isPentestCreationBlocked` / `hasUnauthorizedScope` e o disable do Launch são UX. `useCreateConversation` aceita `workspaceType`/`engagementId` e grava metadata local — não substitui AuthZ de provisionamento/MCP no Engagement Manager / agent-server. | BE deve recusar workspace pentest sem capability + scope autorizado. |
| AS-FE-183-2 | Info | Metadata local forjável | `workspace_type: "pentest"` em localStorage pode ser alterado no browser; hoje só afeta badge/UI (`WorkspaceTypeBadge` / conversation card), não o payload de create do agent-server neste PR. | OK enquanto provisionamento real não confiar nessa metadata. |
| AS-FE-183-3 | Info | XSS engagement name | `engagement.name` renderizado em `<option>` via React (escaped). i18n via `t(I18nKey.…)`. | Nenhuma. |

## Superfície revisada

| Path | Nota de segurança |
|---|---|
| `src/components/features/pentest/workspace-type-selector.tsx` | opção pentest atrás de `CapabilityGate` |
| `src/components/features/pentest/pentest-workspace-fields.tsx` | autonomia `autonomous` gated; select de engagements |
| `src/components/features/pentest/pentest-creation-validation.ts` | regras de bloqueio UX |
| `src/components/features/home/workspace-selection-form.tsx` | integra selector + bloqueio Launch |
| `src/components/features/conversation-panel/local-new-conversation-menu.tsx` | idem |
| `src/hooks/mutation/use-create-conversation.ts` | metadata pentest só localStorage |
| `src/api/conversation-metadata-store.ts` | campos `workspace_type` / `engagement_id` / autonomia |

## Dependências

Mesmo resultado do gate 182: `npm audit --audit-level=high` → exit 0 (sem high/critical).

## Ação requerida

Nenhuma bloqueante para AppSec **FE**. Merge do card 183 ainda depende de gates/ADR e do AuthZ BE para o fluxo completo de provisionamento pentest (ver AS-FE-183-1).
