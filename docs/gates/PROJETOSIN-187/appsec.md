---
card: PROJETOSIN-187
pr: 5
veredicto: FAIL
agente: appsec
data: 2026-08-10
tip: 9aaeb874b
ci: review-manual services/mcp-servers + PoC Python; npm N/A (escopo Python)
repo: klebersjunior/OpenHands
branch: feat/fase1-mcp-recon-webscan-187
---

# AppSecurity — PROJETOSIN-187 (mcp-recon + mcp-webscan)

**Veredicto:** FAIL

Revisor ≠ autor (implementação: Backend; QA laudo separado). Escopo: `services/mcp-servers/**`, packaging mínimo `docker/runtimes/web/`, `.env.sample`. Spec `docs/specs/fase-1/187-mcp-recon-webscan.md` § Segurança · ADR-0001 · PR [#5](https://github.com/klebersjunior/OpenHands/pull/5) @ `9aaeb874b`.

**Não mergeable** enquanto HIGH abaixo não forem remediados e re-gate PASS. QA PASS permanece válido no eixo AC; AppSec bloqueia merge.

## Checklist

- [x] Sem segredos versionados / hardcoded (`SESSION_API_KEY` / tokens só via env; `.env.sample` sem valores reais)
- [x] `npm audit` N/A ao diff Python MCP (sem mudança npm de superfície neste card)
- [x] Session key não bakeada em bundle público (stdio MCP server-side)
- [ ] Scope allowlist fail-closed **sem bypass** — **FAIL (HIGH-1)**
- [ ] Confirmation gate não contornável pelo agente — **FAIL (HIGH-2)**
- [x] Findings auth: `SESSION_API_KEY` obrigatória; 401/403 → `FindingsAuthError` (não engolido)
- [x] Evidence: sem log de bodies em INFO (`findings_client` só loga dedupe 409)
- [x] Command injection: `create_subprocess_exec` + args list (sem `shell=True`)
- [ ] Rate limit sqlmap/ZAP active — **ausente (MEDIUM)**
- [x] Proxies/VNC/Cloud fora de escopo deste card

## Findings

### HIGH-1 — Bypass de `PENTEST_SCOPE_ALLOWLIST` via `str.startswith` — **BLOCK**

**Arquivo:** `services/mcp-servers/shared/normalize.py` (`assert_in_scope`)

Após falhar `_host_matches`, o código aceita:

```python
if any(target.startswith(entry) or entry == target for entry in allowlist):
    return
```

Com allowlist `example.com`, o target bare `example.com.evil.com` (e `example.com.attacker.net`) é **permitido**. PoC local (tip `9aaeb874b`):

```
ALLOW (bypass?): example.com.evil.com
ALLOW (bypass?): example.com.attacker.net
```

Impacto: agente/tool pode varrer e POST findings contra hosts fora do engagement (SSRF/scan off-scope), violando AC-187-4 e § Segurança (“fail-closed”).

**Remediação:** remover o ramo `startswith` para entradas de host/CIDR; matching só via host normalizado (`extract_host` + `_host_matches`). Se precisar prefixo de URL, exigir entrada com `://` **e** comparar hostname parseado (nunca `startswith` no string cru). Adicionar teste negativo `example.com.evil.com` → `scope_violation`.

### HIGH-2 — Confirmation gate contornável por `autonomy_mode` controlado pelo agente — **BLOCK**

**Arquivos:** `mcp-webscan/server.py` (args MCP), `shared/confirmation.py` (`MAX_RISK_TOOLS` vazio)

Tools ativas expõem `autonomy_mode: str = "semi_autonomous"` como parâmetro MCP. O LLM pode passar `autonomy_mode="autonomous"`; com `MAX_RISK_TOOLS == ∅`, `_needs_gate` retorna `False` e o gate não dispara.

PoC: `run_zap_active(target="example.com.evil.com", autonomy_mode="autonomous")` → `ok: true` + POST Findings (1), sem `confirmation_required`. Combina com HIGH-1.

Impacto: blueprint §5.4 / AC-187-5 (semi sem token → confirmação) é inócuo se o caller escolhe o modo. Scan intrusivo (ZAP active / sqlmap) sem aprovação humana.

**Remediação:** ler autonomia de fonte server-side confiável (`PENTEST_AUTONOMY_MODE` / sessão / engagement), **não** do argumento da tool controlado pelo agente. Remover ou ignorar `autonomy_mode` no schema MCP (ou fixar e validar contra env). Teste: chamada com `autonomy_mode=autonomous` sob env `semi_autonomous` ainda exige token.

### MEDIUM — Rate limit ausente em sqlmap / ZAP active

Spec § Segurança: “sqlmap/ZAP active: rate limit e timeout obrigatórios”. Timeout existe via `MCP_WEBSCAN_TIMEOUT_SEC` / `run_binary`, mas runners ativos atuais são stubs sem rate limit real. Não bloqueia sozinho; remediar junto do wire-up de binários.

### MEDIUM — Token de confirmação reutilizável / env sticky

- `_approved_tokens` não é single-use.
- `OPENHANDS_CONFIRMATION_TOKEN` igual ao token aprovado libera **todas** as tools gated sem passar `confirmation_token` de novo (`confirmation.py` L103–104).
- Match `confirmation_token == env_token` sem exigir que o token tenha sido emitido via `approve_confirmation` (L101–102) — se o env for secreto estático, vira bypass permanente.

Aceitável como stub MVP **após** HIGH-2; endurecer no canal UI real (single-use, binding a `request_id`+tool+target).

### LOW — Achados de recon sem re-checagem de host descoberto

`recon_subfinder` valida o domínio raiz; hosts derivados (`www.{domain}`) são postados sem `assert_in_scope` individual. Baixo risco com allowlist de domínio (suffix match). Preferível revalidar cada asset antes do POST.

## Controles OK (não bloqueantes)

| Controle | Evidência |
|----------|-----------|
| Fail-closed se allowlist vazia/ausente | PoC: unset → `ScopeViolationError` |
| Findings auth | `session_auth.py` + `FindingsAuthError` em missing key / 401/403 |
| Sem segredo hardcoded | AC-187-9; fixtures `test-session-key` |
| Sem shell injection | `asyncio.create_subprocess_exec` |
| Evidence não logada em INFO | só mensagem de dedupe 409 |
| `.env.sample` | vars documentadas, sem valores secretos |

## Dependências

Escopo Python (`mcp`, `httpx` em pyproject). Sem `npm audit` aplicável ao diff do card. CVEs de binários ofensivos do runtime-web ficam no residual de PROJETOSIN-186.

## Ação requerida (bloqueia merge)

1. **Corrigir HIGH-1** (matching de escopo sem `startswith` inseguro) + teste de regressão.
2. **Corrigir HIGH-2** (autonomia server-side; agente não escolhe `autonomous` para pular gate) + teste de regressão.
3. Re-gate AppSec no mesmo PR após tip com fix.
4. Plane: label **Blocked** enquanto FAIL.

Tech Lead: **não mergear** com QA PASS isolado — gate AppSec = FAIL.
