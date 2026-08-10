---
card: PROJETOSIN-187
pr: 5
veredicto: PASS
agente: appsec
data: 2026-08-10
tip: fc1474a13
ci: review-manual + PoC Python + pytest mcp-recon (9) + mcp-webscan (9) = 18 passed
repo: klebersjunior/OpenHands
branch: feat/fase1-mcp-recon-webscan-187
re_gate: true
prev_fail_tip: 9aaeb874b
---

# AppSecurity — PROJETOSIN-187 (mcp-recon + mcp-webscan) — re-gate

**Veredicto:** PASS

Revisor ≠ autor (implementação/fix: Backend @ `fc1474a13`; laudo AppSec emitido neste re-gate). Escopo: `services/mcp-servers/**`, packaging mínimo `docker/runtimes/web/`, `.env.sample`. Spec `docs/specs/fase-1/187-mcp-recon-webscan.md` § Segurança · ADR-0001 · PR [#5](https://github.com/klebersjunior/OpenHands/pull/5).

**FAIL anterior** @ `9aaeb874b` (HIGH-1 startswith bypass; HIGH-2 `autonomy_mode` no schema MCP). Remediação verificada neste tip.

**Mergeable (eixo AppSec):** sim, contanto que QA PASS permaneça válido no tip atual (QA tip anterior `04ab2f4cf`; regressão pytest AppSec no tip do fix = 18 passed — Tech Lead confirma se QA precisa re-gate formal no tip `fc1474a13`).

## Checklist

- [x] Sem segredos versionados / hardcoded (`SESSION_API_KEY` / tokens só via env; `.env.sample` sem valores reais)
- [x] `npm audit` N/A ao diff Python MCP (sem mudança npm de superfície neste card)
- [x] Session key não bakeada em bundle público (stdio MCP server-side)
- [x] Scope allowlist fail-closed **sem bypass** — HIGH-1 remediado
- [x] Confirmation gate não contornável pelo agente — HIGH-2 remediado
- [x] Findings auth: `SESSION_API_KEY` obrigatória; 401/403 → `FindingsAuthError` (não engolido)
- [x] Evidence: sem log de bodies em INFO (`findings_client` só loga dedupe 409)
- [x] Command injection: `create_subprocess_exec` + args list (sem `shell=True`)
- [ ] Rate limit sqlmap/ZAP active — **ausente (MEDIUM residual)**
- [x] Proxies/VNC/Cloud fora de escopo deste card

## Re-gate: HIGH-1 / HIGH-2

### HIGH-1 — `assert_in_scope` / `_host_matches` — **PASS (fechado)**

**Arquivo:** `services/mcp-servers/shared/normalize.py`

- Removido o ramo `any(target.startswith(entry) …)`.
- Matching só via `extract_host` + `_host_matches` (exact, subdomain `host.endswith("." + pat)`, wildcard `*.`, CIDR).
- Entradas URL na allowlist usam hostname parseado (`extract_host(entry)` se `://` presente), nunca prefixo cru.

PoC re-gate (`PENTEST_SCOPE_ALLOWLIST=example.com`):

| Target | Resultado |
|--------|-----------|
| `example.com` / `www.example.com` / `api.example.com` | ALLOW |
| `example.com.evil.com` | DENY `scope_violation` |
| `example.com.attacker.net` | DENY |
| `https://example.com.evil.com/` | DENY |

Regressão: `test_high1_scope_startswith_bypass_rejected`, `test_high1_url_allowlist_matches_host_not_prefix`.

### HIGH-2 — autonomia server-side — **PASS (fechado)**

**Arquivos:** `shared/confirmation.py`, `mcp-webscan/server.py`, runners `zap_active` / `sqlmap` / `nuclei`

- Autonomia só via `PENTEST_AUTONOMY_MODE` (`get_autonomy_mode()`); default `semi_autonomous`; valores desconhecidos fail-closed para semi.
- `autonomy_mode` removido do schema MCP das tools ativas e das assinaturas dos runners.
- `require_confirmation` **não** aceita override de autonomia do caller — params: `tool_name`, `payload`, `confirmation_token` apenas.

PoC / testes: `test_high2_agent_cannot_bypass_gate_via_autonomy_arg`, `test_high2_mcp_schema_omits_autonomy_mode`; sob env `semi_autonomous`, ZAP active sem token → `confirmation_required` + zero POST.

## Residuals (não bloqueiam)

### MEDIUM — Rate limit ausente em sqlmap / ZAP active

Spec § Segurança: rate limit + timeout. Timeout via env/`run_binary`; runners ativos ainda stubs sem rate limit real. Remediar no wire-up de binários.

### MEDIUM — Token de confirmação reutilizável / env sticky

- `_approved_tokens` não é single-use.
- `OPENHANDS_CONFIRMATION_TOKEN` sticky / match estático pode liberar tools gated.
- Aceitável como stub MVP; endurecer no canal UI (single-use, binding `request_id`+tool+target).

### LOW — Achados de recon sem re-checagem de host descoberto

Hosts derivados postados sem `assert_in_scope` individual após validar o domínio raiz. Preferível revalidar cada asset antes do POST.

## Controles OK

| Controle | Evidência |
|----------|-----------|
| Fail-closed se allowlist vazia/ausente | `ScopeViolationError` |
| Findings auth | `session_auth.py` + `FindingsAuthError` |
| Sem segredo hardcoded | AC-187-9; fixtures `test-session-key` |
| Sem shell injection | `asyncio.create_subprocess_exec` |
| Evidence não logada em INFO | só dedupe 409 |
| `.env.sample` | vars documentadas, sem valores secretos |

## Dependências

Escopo Python (`mcp`, `httpx`). Sem `npm audit` aplicável ao diff do card. CVEs de binários ofensivos do runtime-web → residual PROJETOSIN-186.

## Evidência de regressão (re-gate)

```
mcp-webscan: 9 passed
mcp-recon:   9 passed
```

## Ação

1. Label **Blocked** removida no Plane (AppSec PASS).
2. Tech Lead: merge só com QA PASS + AppSec PASS no tip; residuals MEDIUM/LOW não bloqueiam este card.
3. Não auto-assinar QA neste re-gate.
