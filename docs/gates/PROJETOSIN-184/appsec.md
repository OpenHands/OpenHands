---
card: PROJETOSIN-184
pr: 2
veredicto: PASS
agente: appsec
data: 2026-08-10
re_gate: 2026-08-10
fix_commit: d1ee30c39
ci: npm-audit-high-clean; review manual services/findings-service + shared
repo: klebersjunior/OpenHands
branch: feat/fase0-backend-184-185-182
---

# AppSecurity — PROJETOSIN-184 (Findings Service)

**Veredicto:** PASS

## Re-gate 2026-08-10

Revalidação após `d1ee30c39`.

### Checklist (atualizado)

- [x] Sem segredos de produção versionados
- [x] `npm audit --audit-level=high` sem high/critical (moderates pré-existentes: dompurify/electron)
- [x] Session/profile AuthZ defensável — HIGH shared fechado
- [x] Proxies ingress `/api/pentest/findings` e `/api/pentest/me` prefixados antes de `/api`
- [x] Isolamento interim por `created_by` + 404 cross-key
- [x] DefectDojo token só via env
- [x] Fail-fast `dev-session-key`; compose exige `SESSION_API_KEY` / `FINDINGS_DB_PASSWORD` (`:?`); Dockerfile sem defaults de senha

### HIGH — Privilege escalation via `X-Pentest-Profile` — **FECHADO**

Ver laudo 182. Consumido via `shared.auth_middleware`; lifespan chama `assert_session_api_key_not_insecure_default()`.

### MEDIUM — IDOR / ACL findings — **FECHADO (interim)**

- `Finding.created_by` obrigatório; list/get/update/triage/stats/sync filtram por `ctx.user_id`.
- Cross-user → 404 (sem leak de existência).
- Evidência: `test_cross_key_finding_access_returns_404` (16 testes findings/shared verdes).
- Nota: membership EngMgr ainda deferred; ownership por criador é ACL fail-closed aceitável no scaffold.

### MEDIUM — Default `SESSION_API_KEY=dev-session-key` — **FECHADO**

- `config.session_api_key` default `""`; compose `${SESSION_API_KEY:?…}` / `${FINDINGS_DB_PASSWORD:?…}`; Dockerfile sem `ENV` de DB password.
- Boot com `dev-session-key` sem `PENTEST_ALLOW_DEV_SESSION_KEY=1` → `RuntimeError` (teste `test_dev_session_key_fail_fast`).

### Residual LOW

`/health` sem auth; sync job store in-memory; `user_id = session:{key[:8]}` (colisão teórica de prefixo).

## Dependências

`npm audit --audit-level=high`: **PASS** (0 high/critical).

## Histórico — Gate inicial (FAIL)

**Veredicto na época:** FAIL por AuthZ do middleware compartilhado e ausência de isolamento por ownership nos findings. Remediação entregue em `d1ee30c39`.
