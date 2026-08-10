# Spec Técnica — PROJETOSIN-187: mcp-recon + mcp-webscan

**ADR:** docs/adrs/0001-plataforma-pentest-ia-extensao-openhands.md (accepted)
**Card Plane:** PROJETOSIN-187 — `7f9694fb-3f81-49c8-96f0-1d51a2b3199c`
**Agente responsável:** backend (+ devops se packaging de imagem)
**Prioridade:** P1 high — MCP ofensivo Web (recon + DAST)
**Base git:** `origin/main` @ `f2a8da86a` (pós merge Fase 0 / PR #4)
**Branch:** `feat/fase1-mcp-recon-webscan-187`
**Worktree:** `.tmp/worktrees/187`
**PR target:** fork `klebersjunior/OpenHands` only (nunca upstream OpenHands/OpenHands)

---

## Objetivo

Implementar dois MCP servers Python (`stdio`) que o Agent Server registra em sessões de workspace **pentest** no runtime Web:

1. **`mcp-recon`** — descoberta de ativos (subfinder, httpx, ReconFTW)
2. **`mcp-webscan`** — DAST (ZAP API, Nuclei, Wapiti, Nikto, sqlmap)

Todo achado normalizado é **POST**ado no Findings Service (`PROJETOSIN-184`, já em `services/findings-service/`). Tools só ficam visíveis se o perfil tiver as capabilities corretas (RBAC Fase 0).

---

## Boundary e localização

```
services/mcp-servers/
├── README.md
├── shared/                         # cliente Findings + auth + confirmation helpers
│   ├── findings_client.py
│   ├── session_auth.py
│   ├── confirmation.py
│   └── normalize.py
├── mcp-recon/
│   ├── pyproject.toml
│   ├── Dockerfile                  # opcional: embutir no runtime-web via COPY
│   ├── server.py                   # entry stdio MCP
│   ├── tools/
│   │   ├── subfinder.py
│   │   ├── httpx_probe.py
│   │   └── reconftw.py
│   └── tests/
│       ├── test_tools_contract.py
│       └── test_findings_post.py
└── mcp-webscan/
    ├── pyproject.toml
    ├── server.py
    ├── tools/
    │   ├── zap_spider.py
    │   ├── zap_passive.py
    │   ├── zap_active.py           # exige confirmation gate
    │   ├── nuclei.py
    │   ├── wapiti.py
    │   ├── nikto.py
    │   └── sqlmap.py               # exige confirmation gate (intrusivo)
    └── tests/
        ├── test_tools_contract.py
        ├── test_active_gate.py
        └── test_findings_post.py
```

**Não** criar repo separado. Empacotar via `uv`/`pyproject` e instalar no `docker/runtimes/web` (Fase 0) ou montar volume no compose do engagement.

---

## Capabilities (canônicas Fase 0)

Aliases do card/blueprint → nomes canônicos em `services/shared/capabilities.py`:

| Alias card | Capability canônica | Onde aplica |
|---|---|---|
| `web.dast` (registro tools) | `pentest.scan.passive` + `pentest.scan.active` | webscan: tools passivas vs ativas |
| recon | `pentest.recon.run` | mcp-recon (todas as tools) |
| findings write | `pentest.scan.passive` (criação) | POST Findings |
| mark FP | `pentest.findings.triage` | fora deste card (UI 188) |

**Registro na sessão:** o launcher / adapter só anexa o MCP server se o usuário autenticado tiver a capability mínima:

- `mcp-recon` → `pentest.recon.run`
- `mcp-webscan` tools passivas (spider, passive ZAP, nuclei info, wapiti/nikto read) → `pentest.scan.passive`
- `mcp-webscan` tools ativas (ZAP active, sqlmap, nuclei intrusive templates) → `pentest.scan.active`

Sem capability → server **não** é listado em MCP config da sessão (agente não “vê” a tool).

---

## Confirmation gate (modo Semi-autônomo)

Blueprint §5.4: recon/scan não-destrutivo livres; ações intrusivas exigem gate.

```python
# services/mcp-servers/shared/confirmation.py
async def require_confirmation(tool_name: str, autonomy_mode: str, payload: dict) -> None:
    """
    autonomy_mode: manual | semi_autonomous | autonomous
    - manual: sempre exige aprovação (via event/UI confirmation)
    - semi_autonomous: exige se tool_name in ACTIVE_TOOLS
    - autonomous: só bloqueia se tool_name in MAX_RISK_TOOLS (fora do MVP Fase 1: vazio)
    ACTIVE_TOOLS = {"zap_active_scan", "sqlmap_run", "nuclei_intrusive"}
    """
```

Implementação MVP: tool retorna erro estruturado `confirmation_required` com `request_id`; o frontend/event stream já existente (ou stub backend) marca aprovação; tool reexecuta com header/`OPENHANDS_CONFIRMATION_TOKEN`. Se o canal de confirmação UI ainda não existir, documentar stub + AC de contrato (QA valida a mensagem de erro e o re-run com token mock).

---

## Contrato MCP — tools

### mcp-recon

| Tool | Args principais | Side-effect |
|---|---|---|
| `recon_subfinder` | `domain`, `engagement_id` | assets → findings `info` (asset discovery) |
| `recon_httpx` | `targets[]`, `engagement_id` | probe HTTP; findings opcionais |
| `recon_reconftw` | `domain`, `engagement_id`, `profile?` | pipeline; findings agregados |

### mcp-webscan

| Tool | Capability | Confirmation |
|---|---|---|
| `web_zap_spider` | passive | não |
| `web_zap_passive_scan` | passive | não |
| `web_zap_active_scan` | active | **sim** (semi) |
| `web_nuclei_scan` | passive (default templates) / active se `severity_filter` inclui critical+intrusive | sim se intrusive |
| `web_wapiti_scan` | passive | não |
| `web_nikto_scan` | passive | não |
| `web_sqlmap_run` | active | **sim** |

Todas as tools exigem `engagement_id` (UUID) e respeitam allowlist de escopo via env `PENTEST_SCOPE_ALLOWLIST` (CSV de hosts/CIDRs). Target fora da allowlist → erro `scope_violation` (não executa).

---

## Normalização → Findings Service

```python
# shared/normalize.py → POST /api/pentest/findings
{
  "engagement_id": "...",
  "source_tool": "nuclei" | "zap" | "wapiti" | "nikto" | "sqlmap" | "subfinder" | "httpx" | "reconftw",
  "title": "...",
  "description": "...",
  "severity": "critical|high|medium|low|info",
  "asset": "host.or.ip",
  "endpoint": "/path",
  "evidence": { "request": "", "response": "", "raw": {} }
}
```

Auth: header `X-Session-API-Key` = `SESSION_API_KEY` (mesmo do stack). Base URL: `FINDINGS_SERVICE_URL` (default `http://findings-service:8000`).

409 dedupe do Findings Service → tratar como sucesso idempotente (log + retornar `existing_finding_id`).

---

## Integração runtime Web

Atualizar `docker/runtimes/web/` **somente o necessário**:

- Instalar bins já previstos (ZAP/Nuclei/…) + pacotes MCP
- `entrypoint.sh` não precisa subir MCP (stdio sob demanda pelo agent-server)
- Documentar em `docker/runtimes/README.md` como registrar:

```toml
# exemplo config.toml fragment (engagement)
[mcp]
# ...
```

Registro programático preferível via settings/MCP API do Agent Canvas quando workspace type = pentest (hook futuro; MVP pode documentar env `PENTEST_MCP_RECON_CMD` / `PENTEST_MCP_WEBSCAN_CMD`).

---

## Critérios de aceite (QA)

1. **AC-187-1:** `mcp-recon` expõe ≥3 tools; schema JSON válido (list_tools)
2. **AC-187-2:** `mcp-webscan` expõe tools ZAP/Nuclei/Wapiti/Nikto/sqlmap
3. **AC-187-3:** Tool passiva com target na allowlist cria finding via Findings Service (201 ou 409 idempotente)
4. **AC-187-4:** Target fora da allowlist → `scope_violation`, **zero** POST findings
5. **AC-187-5:** `web_zap_active_scan` / `web_sqlmap_run` em `semi_autonomous` sem token → `confirmation_required`
6. **AC-187-6:** Com token de confirmação válido → executa e posta findings
7. **AC-187-7:** Sem `X-Session-API-Key` no client Findings → falha autenticada (não engole 401)
8. **AC-187-8:** Testes unitários dos normalizers + gate em `services/mcp-servers/**/tests`
9. **AC-187-9:** Nenhum segredo hardcoded; tokens só via env

---

## Segurança (AppSec)

- Não logar bodies de evidence em INFO
- sqlmap/ZAP active: rate limit e timeout obrigatórios
- Scope allowlist obrigatória em runtime (fail-closed se env ausente em modo pentest)
- Binários: usar os do runtime-web oficial (não curl ad-hoc em produção)

---

## Dependências

- **Depende de:** PROJETOSIN-184 (Findings API), PROJETOSIN-182 (capabilities), PROJETOSIN-186 (runtime-web)
- **Paralelo seguro com:** PROJETOSIN-188 (UI), PROJETOSIN-189 (mcp-sast + DD) — worktrees distintos
- **Não bloqueia:** 188 (UI pode mockar Findings); 189 independente de webscan

**Estimativa:** 4–5 dias
