---
card: PROJETOSIN-187
pr: 5
veredicto: PASS
agente: qa
data: 2026-08-10
tip: ddcb236a0
ci: pytest mcp-recon (9) + mcp-webscan (9) = 18 passed
repo: klebersjunior/OpenHands
branch: feat/fase1-mcp-recon-webscan-187
re_gate: true
prev_qa_tip: 04ab2f4cf
appsec_fix_tip: fc1474a13
---

# QA — PROJETOSIN-187 (mcp-recon + mcp-webscan) — re-gate

**Veredicto:** PASS

Re-gate formal QA após remediação AppSec HIGH-1/HIGH-2. Revisor de papel ≠ autor Backend. Escopo: `services/mcp-servers/**` + packaging mínimo em `docker/runtimes/web/`. Spec `docs/specs/fase-1/187-mcp-recon-webscan.md` · ADR-0001 · PR [#5](https://github.com/klebersjunior/OpenHands/pull/5) @ tip `ddcb236a0` (fix `fc1474a13` + laudo AppSec PASS).

Design N/A (sem UI). **AppSec não auto-assinado** — laudo AppSec já PASS em `docs/gates/PROJETOSIN-187/appsec.md` (revisor separado).

## Critérios de aceite

| AC | Status | Evidência |
|----|--------|-----------|
| AC-187-1 `mcp-recon` ≥3 tools + schema JSON | **PASS** | `test_ac_187_1_recon_exposes_at_least_three_tools` |
| AC-187-2 `mcp-webscan` ZAP/Nuclei/Wapiti/Nikto/sqlmap | **PASS** | `test_ac_187_2_webscan_exposes_dast_tools` |
| AC-187-3 passiva in-scope → POST findings (201/409) | **PASS** | `test_ac_187_3_passive_in_scope_posts_finding`, `test_ac_187_3_passive_zap_posts_finding`, `test_ac_187_3_409_dedupe_idempotent` |
| AC-187-4 fora allowlist → `scope_violation`, zero POST | **PASS** | `test_ac_187_4_out_of_scope_zero_posts`, `test_ac_187_4_nikto_out_of_scope_zero_posts` |
| AC-187-5 active semi sem token → `confirmation_required` | **PASS** | `test_ac_187_5_active_without_token_confirmation_required` (env `PENTEST_AUTONOMY_MODE=semi_autonomous`) |
| AC-187-6 token válido → executa e posta | **PASS** | `test_ac_187_6_with_valid_token_executes_and_posts` |
| AC-187-7 sem/401 Session API Key não engolido | **PASS** | `test_ac_187_7_missing_session_key_auth_fails`, `test_ac_187_7_findings_401_not_swallowed` |
| AC-187-8 unitários normalizers + gate | **PASS** | `test_ac_187_8_normalizer_and_scope`, `test_ac_187_8_gate_unit` + suite 18 passed |
| AC-187-9 sem segredo hardcoded; tokens via env | **PASS** | Auth/`OPENHANDS_CONFIRMATION_TOKEN`/`SESSION_API_KEY` só via env; `.env.sample` documenta vars sem valores reais; grep sem literals secretos |

## Controles pós-AppSec (regressão AC)

| Controle | Status | Evidência |
|----------|--------|-----------|
| Scope DNS-safe (sem startswith bypass) | **PASS** | `test_high1_scope_startswith_bypass_rejected`, `test_high1_url_allowlist_matches_host_not_prefix` — `example.com.evil.com` DENY |
| Autonomy só via env | **PASS** | `test_high2_agent_cannot_bypass_gate_via_autonomy_arg`, `test_high2_mcp_schema_omits_autonomy_mode`, `test_high2_server_autonomous_env_skips_active_gate` |

## Regressão

| Checagem | Resultado |
|----------|-----------|
| `cd services/mcp-servers/mcp-recon && PYTHONPATH=..:. pytest -v` | **9 passed** (0.39s) |
| `cd services/mcp-servers/mcp-webscan && PYTHONPATH=..:. pytest -v` | **9 passed** (0.39s) |
| npm lint/test/build | **N/A** ao escopo pytest MCP |
| CI fork `test-and-build` | **FAILURE** pré-existente — `Cannot find module '@openhands/typescript-client'` (`file:../typescript-client`); **fora do escopo** deste card / pytest |

## Asserções falsificáveis

| Asserção | Status | Evidência |
|----------|--------|-----------|
| Remover `recon_*` de `server.py` quebra AC-187-1 | PASS | `list_tools` asserta nomes canônicos |
| Target `example.com.evil.com` com allowlist `example.com` ainda POST | PASS | HIGH-1 + AC-187-4 exigem DENY / `posts == []` |
| Active tool em semi sem token retorna ok | PASS | AC-187-5 exige `error == confirmation_required` |
| Agente passa `autonomy_mode=autonomous` e bypassa gate | PASS | HIGH-2 — arg removido; autonomia só `PENTEST_AUTONOMY_MODE` |
| 401 Findings engolido como sucesso | PASS | AC-187-7 raise `FindingsAuthError` |

## Residual (não bloqueante)

- Basenames de teste duplicados impedem `pytest` conjunto das duas pastas — README documenta execução por pacote.
- Runners default stubs (sem binário real) salvo env de binários reais.
- Canal de confirmação UI ainda stub; residuals AppSec MEDIUM (rate limit / token sticky) não bloqueiam AC.
- CI npm do fork quebrado por dep local `typescript-client` — não invalida AC MCP.

## Ação requerida

Nenhuma para QA. Tech Lead: **mergeable nos eixos QA + AppSec** (ambos PASS no tip `ddcb236a0`). CI GitHub do fork ainda vermelho por `file:../typescript-client` — tratar à parte se required check bloquear merge.
