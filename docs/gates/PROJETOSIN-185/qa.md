---
card: PROJETOSIN-185
pr: 2
veredicto: PASS
agente: qa
data: 2026-08-10
tip: b0e44356f
fix_commit: d1ee30c39
ci: pytest services/engagement-manager (10) + shared (8)
repo: klebersjunior/OpenHands
branch: feat/fase0-backend-184-185-182
---

# QA — PROJETOSIN-185 (Engagement Manager)

**Veredicto:** PASS

## Escopo

Gate QA Backend PR #2 após AppSec re-gate PASS. ACs da spec `docs/specs/fase-0/185-engagement-manager.md`.

## Critérios de aceite

| AC | Status | Evidência |
|----|--------|-----------|
| AC-185-1 POST → draft | PASS | `test_create_engagement_draft` |
| AC-185-2 authorize-scope preenche `scope_authorized_at` | PASS | `test_authorize_scope_sets_timestamp` |
| AC-185-3 workspace sem scope → 400 | PASS | `test_prepare_workspace_requires_scope` |
| AC-185-4 provision → 202 | PASS | `test_provision_and_teardown` (dry-run) |
| AC-185-5 teardown → stopped | PASS | `test_provision_and_teardown` |
| AC-185-6 listagem só do usuário (IDOR) | PASS | `test_list_only_own_engagements` + **`test_cross_key_engagement_access_returns_404`** (falsificável) |
| AC-185-7 deny rule bloqueia destino | PASS | `test_deny_rule_blocks_destination` (`check-destination`) |
| AC-185-8 401 sem auth / 403 sem capability | PASS | `test_unauthorized_401`, `test_forbidden_403` |

## Asserções falsificáveis

| Asserção | Status | Evidência |
|----------|--------|-----------|
| Escalation via `X-Pentest-Profile` bloqueada | PASS | shared `test_profile_header_escalation_denied_without_flag` |
| Cross-key get → 404; list total 0 | PASS | `test_cross_key_engagement_access_returns_404` (adicionado neste gate) |

## Regressão

```text
services/engagement-manager: 10 passed
services/shared:               8 passed
```

## Residual (não bloqueante)

- AC-185-7 valida allowlist em software (`check-destination`), não rede Docker real (Fase 0 / dry-run).
- Provisioner em `PROVISIONER_DRY_RUN=true` nos testes — compose escrito coberto por `test_provisioner_writes_compose`.

## Ação requerida

Nenhuma para este card neste PR.
