---
card: PROJETOSIN-182
prs: [2, 3]
veredicto: PASS
agentes: [qa]
data: 2026-08-10
escopo: frontend + backend (merged)
---

# QA — PROJETOSIN-182 (FE + BE)

**Veredicto agregado:** PASS (AC-182-1…5 cobertos entre PRs #3 e #2)

> Documento unificado no merge do PR #2 sobre main (já com PR #3).

---

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


---

---
card: PROJETOSIN-182
pr: 2
veredicto: PASS
agente: qa
data: 2026-08-10
tip: b0e44356f
fix_commit: d1ee30c39
escopo: backend shared auth/capabilities (+ /me/capabilities)
ci: pytest services/shared (8) + findings capabilities endpoint
repo: klebersjunior/OpenHands
branch: feat/fase0-backend-184-185-182
---

# QA — PROJETOSIN-182 BE (RBAC shared)

**Veredicto:** PASS (backend only)

## Escopo

Parte **backend** do card 182 neste PR #2. UI/hooks (AC-182-1, AC-182-2, AC-182-5) ficam no PR frontend — não avaliados aqui.

## Critérios de aceite (BE)

| AC | Status | Evidência |
|----|--------|-----------|
| AC-182-1 CapabilityGate FE | N/A | Fora deste PR (FE) |
| AC-182-2 hook true pentester FE | N/A | Fora deste PR (FE) |
| AC-182-3 GET `/api/pentest/me/capabilities` → 403 sem caps | PASS | findings `test_capabilities_endpoint` (profile `none` → 403; pentester → 200) |
| AC-182-4 Middleware Python → 403 sem capability | PASS | findings `test_missing_capability_403`; engmgr `test_forbidden_403`; shared authenticate + caps map |
| AC-182-5 cache logout FE | N/A | Fora deste PR (FE) |

## Asserções falsificáveis (AuthZ)

| Asserção | Status | Evidência |
|----------|--------|-----------|
| Analyst + header `admin` **sem** flag → caps analyst (sem `pentest.admin.*`) | PASS | `test_profile_header_escalation_denied_without_flag` |
| Flag=1 permite header só em teste | PASS | `test_profile_header_honored_only_with_explicit_flag` |
| Mapa `PENTEST_SESSION_PROFILES` vence header mesmo com flag | PASS | `test_session_profiles_map_beats_header_even_with_flag` |
| `dev-session-key` fail-fast | PASS | `test_dev_session_key_fail_fast` |

Vacuidade: remover o gate de `PENTEST_ALLOW_PROFILE_HEADER` faz `test_profile_header_escalation_denied_without_flag` falhar (analyst receberia admin).

## Regressão

```text
services/shared: 8 passed
```

## Relação com FE

Gate QA FE (hooks/CapabilityGate) é independente. Este laudo **não** fecha o card 182 completo — só a fatia BE do PR #2.

## Ação requerida

Nenhuma no BE. Card permanece aberto até QA FE dos ACs 182-1/2/5.

