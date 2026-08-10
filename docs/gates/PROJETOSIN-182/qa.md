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
