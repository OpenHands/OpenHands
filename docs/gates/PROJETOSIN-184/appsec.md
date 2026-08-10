---
card: PROJETOSIN-184
pr: 2
veredicto: FAIL
agente: appsec
data: 2026-08-10
ci: npm-audit-high-clean; review manual services/findings-service + shared
repo: klebersjunior/OpenHands
branch: feat/fase0-backend-184-185-182
---

# AppSecurity — PROJETOSIN-184 (Findings Service)

**Veredicto:** FAIL

## Resumo

Scaffold do Findings Service cobre auth por `X-Session-API-Key`, capabilities por rota e ORM parametrizado. O gate **bloqueia** por AuthZ quebrada no middleware compartilhado (`X-Pentest-Profile` eleva para `admin`) e por ausência de isolamento por engagement nos endpoints de findings (IDOR entre chaves/mapas de perfil).

## Checklist

- [x] Sem segredos de produção versionados (tokens LLM/DD reais ausentes)
- [x] `npm audit --audit-level=high` sem high/critical (apenas moderate pré-existente: dompurify/electron)
- [ ] Session/profile AuthZ defensável em profundidade — **FAIL** (shared)
- [x] Proxies ingress `/api/pentest/findings` e `/api/pentest/me` prefixados antes de `/api`
- [ ] Isolamento de findings por engagement/tenant — **parcial FAIL**
- [x] DefectDojo token só via env (stub offline sem token hardcoded)

## Findings

### HIGH — Privilege escalation via `X-Pentest-Profile` (shared)

- **Onde:** `services/shared/auth_middleware.py` (`resolve_profile_for_key` / `get_auth_context`)
- **Problema:** Com `SESSION_API_KEY` válida e sem entrada em `PENTEST_SESSION_PROFILES`, o cliente escolhe o perfil via header (incl. `admin` → `pentest.admin.*`, `export_dd`, etc.). Viola a premissa da spec 182 (“capabilities nunca vêm só do client-side”).
- **Impacto em 184:** bypass de `require_capability` em triage, sync DD, delete admin.
- **Remediação:** honrar header só com flag explícita de teste (`PENTEST_ALLOW_PROFILE_HEADER=1`); em runtime normal, perfil **somente** via mapa server-side / default env **sem** override do cliente.

### MEDIUM — IDOR / falta de ACL por engagement nos findings

- **Onde:** `app/services/findings_service.py`, routers `findings` / `triage` / sync
- **Problema:** List/get/patch/triage/sync filtram só por capability + `engagement_id`/`finding_id` informados. Não há checagem de membership/`created_by` (EngMgr faz; Findings não). Com múltiplas chaves em `PENTEST_SESSION_PROFILES`, UUID conhecido = leitura/escrita/triage cross-tenant.
- **Remediação:** validar ownership/membership do engagement (serviço interno ou claim no token) antes de mutar/listar; negar 404 em miss.

### MEDIUM — Default `SESSION_API_KEY=dev-session-key`

- **Onde:** `app/config.py`, `docker-compose.fragment.yml` (`${SESSION_API_KEY:-dev-session-key}`), Dockerfile `ENV FINDINGS_DB_URL=...findings:findings...`
- **Problema:** chave e senha de DB previsíveis se deploy omitir secrets.
- **Remediação:** fail-fast se key default em não-dev; sem default de senha no compose/Dockerfile de imagem publicada.

### LOW — `/health` sem auth; sync job store in-memory

Aceitável em scaffold; documentar que status de job não é API pública ainda.

## Dependências

`npm audit --audit-level=high`: **PASS** (0 high/critical). Moderates pré-existentes fora do escopo deste PR.

## Ação requerida (bloqueio)

1. Corrigir AuthZ do shared (header profile) — **obrigatório para PASS**.
2. Definir e implementar isolamento findings↔engagement (ou documentar single-tenant explícito + guardrail).
3. Remover/bloquear defaults inseguros de session key em caminhos de deploy.
