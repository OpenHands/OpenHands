---
card: PROJETOSIN-185
pr: 2
veredicto: FAIL
agente: appsec
data: 2026-08-10
ci: npm-audit-high-clean; review manual services/engagement-manager + shared
repo: klebersjunior/OpenHands
branch: feat/fase0-backend-184-185-182
---

# AppSecurity — PROJETOSIN-185 (Engagement Manager)

**Veredicto:** FAIL

## Resumo

Engagement Manager aplica capability checks, ownership por `created_by` nas rotas de leitura/mutação do criador, gate de scope antes de provision, e `project_name` derivado de UUID (sem path traversal óbvio). O gate **bloqueia** pelo mesmo AuthZ HIGH do middleware compartilhado (`X-Pentest-Profile` → `admin`), que anula a defesa em profundidade de `pentest.admin.scope` / provision / teardown.

## Checklist

- [x] Sem tokens DefectDojo/LLM hardcoded
- [x] SQL via SQLAlchemy (sem concatenação crua)
- [x] Provisioner: workdir `compose_work_dir / eng-<uuid8>`; templates usam `tojson` para rules
- [x] `EngagementUpdate` não permite sobrescrever `sandbox_compose_project`
- [ ] AuthZ de perfil não controlável pelo cliente — **FAIL** (shared)
- [x] Ingress `/api/pentest/engagements` registrado com prefixo longo
- [x] `PROVISIONER_DRY_RUN` default true no Dockerfile (reduz risco de compose real acidental)

## Findings

### HIGH — Privilege escalation via `X-Pentest-Profile` (shared)

- **Onde:** `services/shared/auth_middleware.py` (consumido por `app/middleware/auth.py`)
- **Problema:** Holder da session key eleva para `admin` e obtém `pentest.admin.scope`, podendo autorizar scope / alterar allow-deny e provisionar sandboxes além do perfil pretendido.
- **Remediação:** ver laudo PROJETOSIN-182 / 184 — desabilitar override por header fora de teste.

### MEDIUM — Defaults inseguros de sessão/DB

- **Onde:** `app/config.py` (`session_api_key="dev-session-key"`), compose fragment, Dockerfile `ENGMGR_DB_URL=...engmgr:engmgr...`, uvicorn `--host 0.0.0.0`
- **Problema:** superfície de rede do container + credenciais default se secrets não forem injetados.
- **Remediação:** exigir `SESSION_API_KEY` forte; DB URL só via secret mount; bind/publish consciente.

### LOW — Render Jinja de scope rules

`ALLOW_RULES`/`DENY_RULES` passam por `tojson` — mitigação adequada contra YAML breakout trivial. Manter validação de `target_value` (tamanho/charset) quando o provisioner sair de dry-run.

### Nota positiva

- Isolamento por `created_by` em get/list/update/scope/provision/teardown.
- Delete admin sem filtro de owner é aceitável **se** `pentest.admin.users` não for auto-atribuível pelo cliente (hoje é, via finding HIGH).

## Dependências

Sem high/critical no `npm audit` do monorepo (Python deps locais não auditadas via npm; sem PyPI lock committed — risco residual LOW a acompanhar).

## Ação requerida (bloqueio)

1. Mesma correção de AuthZ shared que 182/184.
2. Fail-closed para session key/DB defaults fora de dev.
