---
card: PROJETOSIN-187
pr: 5
veredicto: PASS
agente: qa
data: 2026-08-10
tip: 04ab2f4cf
ci: pytest mcp-recon (7) + mcp-webscan (6) = 13 passed
repo: klebersjunior/OpenHands
branch: feat/fase1-mcp-recon-webscan-187
---

# QA — PROJETOSIN-187 (mcp-recon + mcp-webscan)

**Veredicto:** PASS

Gate formal QA. Revisor de papel ≠ autor Backend. Escopo: `services/mcp-servers/**` + packaging mínimo em `docker/runtimes/web/`. Spec `docs/specs/fase-1/187-mcp-recon-webscan.md` · ADR-0001 · PR [#5](https://github.com/klebersjunior/OpenHands/pull/5) @ `04ab2f4cf`.

Design N/A (sem UI). **AppSec não auto-assinado** — aguarda gate AppSec separado.

## Critérios de aceite

| AC | Status | Evidência |
|----|--------|-----------|
| AC-187-1 `mcp-recon` ≥3 tools + schema JSON | **PASS** | `test_ac_187_1_recon_exposes_at_least_three_tools` — `recon_subfinder`, `recon_httpx`, `recon_reconftw`; `inputSchema.type == object` |
| AC-187-2 `mcp-webscan` ZAP/Nuclei/Wapiti/Nikto/sqlmap | **PASS** | `test_ac_187_2_webscan_exposes_dast_tools` — subset das 7 tools; `engagement_id` em properties |
| AC-187-3 passiva in-scope → POST findings (201/409) | **PASS** | `test_ac_187_3_passive_in_scope_posts_finding`, `test_ac_187_3_passive_zap_posts_finding`, `test_ac_187_3_409_dedupe_idempotent` |
| AC-187-4 fora allowlist → `scope_violation`, zero POST | **PASS** | `test_ac_187_4_out_of_scope_zero_posts`, `test_ac_187_4_nikto_out_of_scope_zero_posts` — `transport.posts == []` |
| AC-187-5 active semi sem token → `confirmation_required` | **PASS** | `test_ac_187_5_active_without_token_confirmation_required` — `web_zap_active_scan` + `web_sqlmap_run` |
| AC-187-6 token válido → executa e posta | **PASS** | `test_ac_187_6_with_valid_token_executes_and_posts` — `approve_confirmation` + re-run |
| AC-187-7 sem/401 Session API Key não engolido | **PASS** | `test_ac_187_7_missing_session_key_auth_fails`, `test_ac_187_7_findings_401_not_swallowed` → `FindingsAuthError` |
| AC-187-8 unitários normalizers + gate | **PASS** | `test_ac_187_8_normalizer_and_scope`, `test_ac_187_8_gate_unit` + suite completa sob `services/mcp-servers/**/tests` |
| AC-187-9 sem segredo hardcoded; tokens via env | **PASS** | Auth via `SESSION_API_KEY` / `OPENHANDS_CONFIRMATION_TOKEN`; fixtures só `test-session-key`; `.env.sample` documenta vars sem valores reais |

## Regressão

| Checagem | Resultado |
|----------|-----------|
| `cd services/mcp-servers/mcp-recon && PYTHONPATH=..:. pytest -v` | **7 passed** (0.37s) |
| `cd services/mcp-servers/mcp-webscan && PYTHONPATH=..:. pytest -v` | **6 passed** (0.38s) |
| npm lint/test/build | **N/A** — escopo Python MCP; sem mudança TS de runtime de produto |

## Asserções falsificáveis

| Asserção | Status | Evidência |
|----------|--------|-----------|
| Remover `recon_*` de `server.py` quebra AC-187-1 | PASS | `list_tools` asserta nomes canônicos |
| Target fora allowlist ainda POST findings | PASS | AC-187-4 exige `posts == []` |
| Active tool em semi sem token retorna ok | PASS | AC-187-5 exige `error == confirmation_required` |
| 401 Findings engolido como sucesso | PASS | AC-187-7 raise `FindingsAuthError` |

## Residual (não bloqueante)

- Basenames de teste duplicados (`test_findings_post.py` / `test_tools_contract.py`) impedem `pytest` conjunto das duas pastas (import mismatch). README documenta execução por pacote — OK para AC-187-8.
- Runners default são stubs (sem binário real) salvo `MCP_RECON_USE_REAL_BINARIES=1` — contrato MCP/Findings validado; integração binária fica no runtime-web (186) / AppSec.
- Canal de confirmação UI ainda stub (`approve_confirmation` / env token) — previsto na spec MVP.

## Ação requerida

Nenhuma para QA. Tech Lead: liberar **AppSec** no mesmo PR; merge só com QA + AppSec PASS.
