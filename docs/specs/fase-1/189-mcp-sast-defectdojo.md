# Spec Técnica — PROJETOSIN-189: mcp-sast + Sync DefectDojo one-way

**ADR:** docs/adrs/0001-plataforma-pentest-ia-extensao-openhands.md (accepted)
**Card Plane:** PROJETOSIN-189 — `65a023bf-f25e-4381-9bcf-56c6eae89d30`
**Agente responsável:** backend
**Prioridade:** P2 medium
**Base git:** `origin/main` @ `f2a8da86a`
**Branch:** `feat/fase1-mcp-sast-dd-189`
**Worktree:** `.tmp/worktrees/189`
**PR target:** fork `klebersjunior/OpenHands` only

---

## Objetivo (duas entregas acopladas)

### A) `mcp-sast`

MCP server Python (`stdio`) para **Semgrep** + **Trivy**, rodando no runtime SAST (`docker/runtimes/sast`). Findings normalizados → Findings Service.

### B) Sync DefectDojo one-way (espelho)

Completar/robustecer o job já esboçado em `services/findings-service/app/services/defectdojo_sync.py` para apontar ao **DefectDojo de produção Heimdall** (não provisionar DD novo — ADR §pós-intake #4).

Fluxo: Findings Service (master) → DefectDojo (mirror). Nunca o inverso.

---

## Parte A — mcp-sast

### Localização

```
services/mcp-servers/mcp-sast/
├── pyproject.toml
├── server.py
├── tools/
│   ├── semgrep_scan.py
│   └── trivy_scan.py
└── tests/
    ├── test_tools_contract.py
    └── test_normalize_sast.py
```

Reutilizar `services/mcp-servers/shared/` se 187 já tiver mergeado; senão **duplicar mínimo** (`findings_client`, `normalize`) neste PR e extrair shared em follow-up — **não** bloquear 189 em 187.

### Capabilities

| Alias card | Canônica |
|---|---|
| `web.sast` | Usar `pentest.scan.passive` para registro MVP **ou** estender `services/shared/capabilities.py` + `src/types/pentest-rbac.ts` com `pentest.sast.run` se TL/FE alinharem no mesmo PR |

**Decisão desta spec:** introduzir capability **`pentest.sast.run`** em shared + mirror TS (perfil `pentester`/`admin`). Tools mcp-sast exigem essa capability para registro na sessão.

### Tools

| Tool | Args | Notas |
|---|---|---|
| `sast_semgrep_scan` | `engagement_id`, `path` (default workspace), `config?` | parse JSON Semgrep → findings |
| `sast_trivy_scan` | `engagement_id`, `target` (fs/image), `scanners?` | vulns/misconfig → findings |

Scope: path deve estar sob working dir do engagement (path traversal = erro).

`source_tool`: `semgrep` | `trivy`.

### AC (mcp-sast)

1. **AC-189-A1:** list_tools expõe semgrep + trivy
2. **AC-189-A2:** scan fixture → ≥1 POST Findings com severidade mapeada
3. **AC-189-A3:** path fora do workspace → erro, sem POST
4. **AC-189-A4:** testes unitários de mapeamento severidade Semgrep/Trivy → enum Findings

---

## Parte B — DefectDojo sync

### Boundary

- Código em `services/findings-service/` (já existe stub)
- **Não** docker-compose de DefectDojo novo
- Env (já previstos / estender `.env.sample`):

```
DEFECTDOJO_API_URL=https://<dd-producao-heimdall>
DEFECTDOJO_API_TOKEN=<secret>
DEFECTDOJO_PRODUCT_TYPE_DEFAULT=...
DEFECTDOJO_VERIFY_TLS=true
```

### Comportamento

1. **Endpoint existente** `POST /api/pentest/findings/sync-defectdojo` com `{ engagement_id, status_filter }` (default `["confirmed"]`)
2. Job assíncrono (store in-memory MVP ok; documentar limite single-process)
3. Preferir **`/api/v2/reimport-scan/`** com `auto_create_context=true`
4. Mapeamento contexto:
   - Product_Type ← cliente (metadata engagement / env default)
   - Product ← alvo
   - Engagement ← contrato/janela
   - Test ← `source_tool` + execução
5. **Parsers nativos** quando `source_tool` ∈ {zap, nikto, nuclei, nmap, trivy, semgrep, mobsf} **e** `evidence.raw` contiver artefato bruto reconhecível; caso contrário **Generic Findings Import** JSON (já parcial no stub)
6. Após sucesso: preencher `defectdojo_id`, `defectdojo_synced_at`
7. **Espelho de status:** em `POST .../triage`, se finding já tem `defectdojo_id`, propagar FP/mitigado/active conforme mapa:

| Findings status | DD (aprox.) |
|---|---|
| `false_positive` | false_positive |
| `risk_accepted` | risk_accepted / mitigated (doc choice: `risk_accepted`) |
| `confirmed` | active |
| `duplicate` | duplicate |

Falhas de mirror DD **não** revertem triage local (log + job retry flag).

### Segurança

- Token nunca em logs/respostas
- Timeout/retry com backoff; não vazar body de erro DD com dados sensíveis ao client (mensagem genérica + request id)

### AC (DD)

1. **AC-189-B1:** sync com mock httpx → findings confirmed recebem `defectdojo_id`
2. **AC-189-B2:** status_filter exclui `new` / `false_positive` por default
3. **AC-189-B3:** triage FP em finding com `defectdojo_id` dispara update DD (mock assert)
4. **AC-189-B4:** sem `DEFECTDOJO_API_TOKEN` → 503 claro no sync endpoint
5. **AC-189-B5:** testes em `services/findings-service/tests/test_defectdojo_sync.py` estendidos (não inverter asserts antigos)

---

## Critérios de aceite agregados (QA)

- Todos AC-189-A* e AC-189-B*
- `npm` suite do canvas **não** precisa ficar vermelha (escopo Python services); se tocar TS capabilities, `npm test` + lint
- Laudo em `docs/gates/PROJETOSIN-189/` por QA/AppSec (não pelo autor)

---

## Dependências

- **Depende de:** PROJETOSIN-184 (serviço + stub sync), PROJETOSIN-182 (capabilities), PROJETOSIN-186 (runtime-sast)
- **Independente de:** 187 (mcp-recon/webscan) e 188 (UI) — paralelo em worktree própria
- **Credenciais DD:** usar secrets do ambiente; não commitar tokens. Se URL/token de staging ausentes no CI, testes com httpx mock apenas

**Estimativa:** 3–4 dias (A+B)
