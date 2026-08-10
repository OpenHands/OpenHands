---
card: PROJETOSIN-182
pr: 3
veredicto: PASS
agente: qa
data: 2026-08-10
ci: vitest-pentest-subset
repo: klebersjunior/OpenHands
branch: feat/fase0-frontend-182-183
escopo: frontend (este PR)
---

# QA Report — PROJETOSIN-182 RBAC + Feature Gating (FE)

**Veredicto:** PASS (escopo frontend deste PR)

**PR:** https://github.com/klebersjunior/OpenHands/pull/3  
**Spec:** `docs/specs/fase-0/182-rbac-feature-gating.md`  
**Design gate:** PASS (`docs/gates/PROJETOSIN-182/design.md`)  
**AppSec:** não emitido neste gate (outro agente).

## Critérios de aceite

| AC | Status | Evidência |
|---|---|---|
| AC-182-1 — sem capability → gate esconde children | PASS | `__tests__/hooks/use-pentest-capabilities.test.tsx` (`returns false for missing capability`); `__tests__/components/pentest/capability-gate.test.tsx` (`hides children when capability is missing`) — falhariam se o hook/gate não existissem |
| AC-182-2 — pentester → children renderizados | PASS | mesmos arquivos (`returns true for pentester…`, `renders children when capability is present`) |
| AC-182-3 — GET capabilities 403 sem pentest | PASS (contrato client) | `src/api/pentest-service/pentest-service.api.test.ts` — client trata 403 como `{ profile: null, capabilities: [] }`. Endpoint server-side real fica para Findings/EngMgr (fora do diff FE) |
| AC-182-4 — middleware Python 403 | FORA DE ESCOPO (este PR) | `services/shared/` só tem README stub — sem `auth_middleware.py` / `capabilities.py`. Backend Fase 0 (184/185) deve cobrir; **não** auto-PASS |
| AC-182-5 — cache invalida no logout | PASS | hook `useInvalidatePentestCapabilities` + teste `clears capability cache on invalidate`; wiring em `src/routes/users-settings.tsx` após `AppLoginService.logout()` |

## Regressão

Comando (worktree `OpenHands-wt-frontend`):

```bash
npx vitest run \
  __tests__/hooks/use-pentest-capabilities.test.tsx \
  __tests__/components/pentest/ \
  __tests__/api/conversation-metadata-pentest.test.ts \
  src/api/pentest-service/pentest-service.api.test.ts
```

**Resultado:** PASS (19) FAIL (0)

Não rodado neste gate: `npm run lint` / `npm test` full / E2E mock-LLM (mapping não exige path E2E exclusivo para estes arquivos; regressão unitária focada no controle).

## Notas

- Asserções são negativas onde importa (queryByTestId ausente; `toBe(false)`), evitando PASS vácuo.
- Card completo 182 ainda depende do middleware server-side (AC-182-4) + endpoint real — rastrear no backend; merge deste PR FE não fecha AppSec nem backend.

## Ação requerida

Nenhuma para o escopo FE. Backend: implementar `services/shared/auth_middleware.py` + `capabilities.py` e testes que falhem sem o Depends.
