---
card: PROJETOSIN-189
pr: 6
veredicto: PASS
agente: appsec
data: 2026-08-10
tip: 52091bf09
ci: npm-audit-high-clean; review manual mcp-sast + defectdojo_sync
repo: klebersjunior/OpenHands
branch: feat/fase1-mcp-sast-dd-189
---

# AppSecurity — PROJETOSIN-189 (mcp-sast + DefectDojo one-way sync)

**Veredicto:** PASS

**Revisor:** AppSec gate (não autor do código; não assina QA). QA permanece em `docs/gates/PROJETOSIN-189/qa.md` (PASS).

## Escopo

Spec `docs/specs/fase-1/189-mcp-sast-defectdojo.md` § Segurança + AC de superfície:

- `services/mcp-servers/mcp-sast/` (+ `shared/` mínimo: path guard, findings client, session auth)
- `services/findings-service/` sync DefectDojo / triage mirror
- Capability `pentest.sast.run` (Python + mirror TS)
- Foco: token DD em logs/respostas, path traversal, one-way sync, 503 sem token, secrets em env, SSRF/injection via tools

Worktree `.tmp/worktrees/189` @ tip `52091bf09` (PR #6).

## Checklist

- [x] Sem segredos versionados / hardcoded (token DD vazio em samples; só placeholder de teste)
- [x] `npm audit --audit-level=high` sem high/critical (4 moderate pré-existentes: dompurify/electron — fora do delta 189)
- [x] `DEFECTDOJO_API_TOKEN` só via env / Settings; default `""`; compose `${DEFECTDOJO_API_TOKEN:-}`
- [x] Sync sem token → HTTP 503 (mensagem cita o *nome* da env, não o valor)
- [x] Token não aparece em logs de retry/erro (só `request_id`, status, `type(exc).__name__`)
- [x] Erros DD ao client: genéricos (`DefectDojoClientError` + request_id); job async marca `failed` sem body DD
- [x] Path traversal: `resolve_workspace_path` + testes A3 (semgrep/trivy fs) — sem POST Findings
- [x] One-way: só `POST .../reimport-scan/` e `PATCH .../findings/{id}/` (status); sem ingestão DD→Findings
- [x] Subprocess via `create_subprocess_exec` (lista argv) — sem shell=True / injection por path
- [x] TLS verify default `true`; timeout + retry com backoff
- [x] Capability `pentest.sast.run` em admin/pentester; sync exige `pentest.findings.export_dd`

## Findings

### Critical / High

Nenhum. **Sem bloqueio.**

### MEDIUM — Semgrep `config` / Trivy image como superfície de egress (residual aceitável)

- `sast_semgrep_scan(config=...)` passa `--config` ao binário; Semgrep pode aceitar regras remotas (URL) → egress/SSRF do sandbox quando `MCP_SAST_USE_REAL_BINARIES=1`.
- `sast_trivy_scan` trata refs de imagem sem path guard (intencional para scan de imagem) → pull de registry.

**Decisão:** residual MEDIUM, não FAIL. Ferramentas ofensivas esperam egress controlado na camada de rede do runtime SAST; argv list evita RCE por injection; path fs permanece guardado.

### MEDIUM — Enforcement de `pentest.sast.run` só na documentação de registro

`REQUIRED_CAPABILITY` em `server.py` não é checado no invoke das tools (comentário: enforcement no launcher/sessão). Alinhado à spec (“registro na sessão”); residual se o launcher falhar em anexar a capability.

### LOW

- `FindingsClientError` / `FindingsAuthError` guardam `resp.text` no objeto; tools só devolvem `status_code` em auth — erros HTTP não-auth podem bubblar via FastMCP (corpo Findings, não token DD).
- Job store DD in-memory (single-process) — já documentado; sem impacto de segredo.
- `DEFECTDOJO_DRY_RUN` scaffold — default `false`; não setar em produção.

## Controles verificados (mapeamento § Segurança)

| Controle | Evidência |
|----------|-----------|
| Token nunca em logs/respostas | `_request_with_retry` / `sync_finding` logam status + request_id; API 202 só `job_id`/`status`; `FindingOut` sem token |
| Timeout/retry + mensagem genérica | `defectdojo_timeout_seconds`, backoff 0.25×2^n; `DefectDojoClientError(request_id)` |
| Path fora do workspace | `shared/normalize.resolve_workspace_path`; testes `test_path_outside_*` / `test_trivy_path_traversal_*` |
| One-way (Findings master) | Nenhum GET de findings DD para criar locais; mirror só status outbound |
| 503 sem token | `sync_defectdojo` + `test_sync_without_token_returns_503` |
| Secrets env-only | `.env.sample` / `compose.env.example` comentados com token vazio; Settings pydantic |

## Dependências

`npm audit --audit-level=high`: **PASS** (0 high/critical).

## Review GitHub

Intent: `APPROVE`. Conta `gh` = autor do PR (`klebersjunior`) — GitHub bloqueia self-approve. Review formal **COMMENT** com veredicto AppSec **PASS** explícito (mesmo padrão do gate QA). Papel AppSec ≠ autor Backend.

## Ação requerida

Nenhuma para AppSec. Merge sob Tech Lead com **QA PASS** + **AppSec PASS** neste card.
