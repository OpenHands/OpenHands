---
card: PROJETOSIN-185
pr: 2
veredicto: PASS
agente: appsec
data: 2026-08-10
re_gate: 2026-08-10
fix_commit: d1ee30c39
ci: npm-audit-high-clean; review manual services/engagement-manager + shared
repo: klebersjunior/OpenHands
branch: feat/fase0-backend-184-185-182
---

# AppSecurity — PROJETOSIN-185 (Engagement Manager)

**Veredicto:** PASS

## Re-gate 2026-08-10

Revalidação após `d1ee30c39`.

### Checklist (atualizado)

- [x] Sem tokens DefectDojo/LLM hardcoded
- [x] SQL via SQLAlchemy (sem concatenação crua)
- [x] Provisioner: workdir `compose_work_dir / eng-<uuid8>`; templates usam `tojson` para rules
- [x] `EngagementUpdate` não permite sobrescrever `sandbox_compose_project`
- [x] AuthZ de perfil não controlável pelo cliente — HIGH shared fechado
- [x] Ingress `/api/pentest/engagements` registrado com prefixo longo
- [x] `PROVISIONER_DRY_RUN` default true no Dockerfile
- [x] Fail-fast session key; compose exige secrets (`:?`); sem default de senha no Dockerfile

### HIGH — Privilege escalation via `X-Pentest-Profile` — **FECHADO**

Mesmo middleware shared; `pentest.admin.scope` / provision / teardown deixam de ser auto-atribuíveis via header. Evidência: testes shared de escalation + 9 testes engmgr verdes.

### MEDIUM — Defaults inseguros de sessão/DB — **FECHADO**

- `session_api_key` default `""`; lifespan → `assert_session_api_key_not_insecure_default()`.
- Compose: `SESSION_API_KEY=${SESSION_API_KEY:?…}`, `ENGMGR_DB_PASSWORD=${ENGMGR_DB_PASSWORD:?…}`.
- Dockerfile: sem URL/senha default; `PROVISIONER_DRY_RUN=true` mantido.
- Bind `0.0.0.0` no container permanece (esperado); superfície depende de publish consciente — residual operacional LOW.

### Residual LOW

Render Jinja de scope rules via `tojson` (já mitigado). Validar charset de `target_value` quando provisioner sair de dry-run.

### Nota positiva

Isolamento por `created_by` em get/list/update/scope/provision/teardown permanece. Delete admin só com `pentest.admin.users` (não atribuível via header em runtime).

## Dependências

Sem high/critical no `npm audit` do monorepo.

## Histórico — Gate inicial (FAIL)

**Veredicto na época:** FAIL pelo HIGH do middleware compartilhado e defaults de sessão/DB. Remediação entregue em `d1ee30c39`.
