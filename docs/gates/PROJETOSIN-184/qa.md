---
card: PROJETOSIN-184
pr: 2
veredicto: PASS
agente: qa
data: 2026-08-10
tip: b0e44356f
fix_commit: d1ee30c39
ci: pytest services/findings-service (13) + shared (8)
repo: klebersjunior/OpenHands
branch: feat/fase0-backend-184-185-182
---

# QA — PROJETOSIN-184 (Findings Service)

**Veredicto:** PASS

## Escopo

Gate QA Backend PR #2 após AppSec re-gate PASS. Validação dos ACs da spec `docs/specs/fase-0/184-findings-service.md` + asserções falsificáveis de ownership e AuthZ.

## Critérios de aceite

| AC | Status | Evidência |
|----|--------|-----------|
| AC-184-1 POST → 201 + UUID | PASS | `test_create_finding_201` |
| AC-184-2 dedupe → 409 + `existing_finding_id` | PASS | `test_create_duplicate_409` |
| AC-184-3 triage + 422 inválida | PASS | `test_triage_transitions`, `test_invalid_transition_422` |
| AC-184-4 filtro severity+engagement | PASS | `test_list_filter_severity` |
| AC-184-5 sem `engagement_id` → 422 | PASS | `test_list_requires_engagement_id` |
| AC-184-6 sem key → 401 | PASS | `test_missing_api_key_401` |
| AC-184-7 sem capability → 403 | PASS | `test_missing_capability_403` (profile `none`) |
| AC-184-8 sync DD job + `defectdojo_id` | PASS | `test_sync_defectdojo_queues_and_sets_id` |
| AC-184-9 migração Alembic carrega | PASS | `test_alembic_migration_module_loads` (upgrade/downgrade importáveis; schema via `create_all` nos testes) |

## Asserções falsificáveis (AuthZ / ownership)

| Asserção | Status | Evidência |
|----------|--------|-----------|
| Header `X-Pentest-Profile` não escala sem flag | PASS | shared `test_profile_header_escalation_denied_without_flag` |
| Cross-key get/list/triage → 404 / lista vazia | PASS | `test_cross_key_finding_access_returns_404` |

Vacuidade: se `created_by` deixar de filtrar, o teste cross-key falha; se o header voltar a ser honrado sem `PENTEST_ALLOW_PROFILE_HEADER=1`, o teste de escalation falha.

## Regressão

```text
services/findings-service: 13 passed
services/shared:            8 passed
```

Comando: `uv sync --extra dev && uv run pytest tests/ -v` (findings); shared com `PYTHONPATH=services`.

## Residual (não bloqueante)

- AC-184-9 não executa `alembic upgrade` contra Postgres real neste gate (módulo + `create_all` em sqlite in-memory).
- Membership EngMgr cross-service ainda deferred (ownership por criador — alinhado ao AppSec interim).

## Ação requerida

Nenhuma. Merge sob responsabilidade do Tech Lead após Design (N/A) + AppSec PASS + este QA PASS.
