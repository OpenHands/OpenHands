# Spec Técnica — PROJETOSIN-190: mcp-mobile + MobSF

**ADR:** docs/adrs/0001-plataforma-pentest-ia-extensao-openhands.md (accepted)  
**Card Plane:** PROJETOSIN-190 — `3897aea7-4e03-484d-87d6-92cec50ffa90`  
**Agente responsável:** backend  
**Prioridade:** P1 high — MCP ofensivo Mobile (estático + dinâmico via ADB genérico)  
**Base git:** `dbfa75033` (fork tip pós Fase 1)  
**Branch:** `feat/fase2-mcp-mobile-190`  
**Worktree:** `.tmp/worktrees/190`  
**PR target:** fork `klebersjunior/OpenHands` only (nunca upstream)

---

## Objetivo

1. Implementar **`mcp-mobile`** (stdio MCP) no padrão de `mcp-recon` / `mcp-webscan` / `mcp-sast`.
2. Cliente REST do **OWASP MobSF** (upload → scan → report) + normalização → Findings Service.
3. Tools ADB/Frida/apktool/jadx contra um **endpoint ADB genérico** (`ADB_HOST:ADB_PORT`) — não acoplar a container vs host (Fase 3 físico fica transparente).
4. Fragmento compose / env docs para o sidecar **MobSF** (serviço no engagement; imagem oficial). Emulador em si é **PROJETOSIN-191**.

---

## Boundary e localização

```
services/mcp-servers/
├── README.md                          # atualizar tabela capabilities + env
├── shared/                            # reusar findings_client, confirmation, normalize, session_auth
└── mcp-mobile/
    ├── pyproject.toml
    ├── server.py                      # FastMCP stdio; REQUIRED_CAPABILITY
    ├── mobsf_client.py                # REST: upload, scan, report, score
    ├── tools/
    │   ├── mobsf_static.py            # upload+scan estático → findings
    │   ├── mobsf_dynamic.py           # dynamic analysis (gate confirmation)
    │   ├── adb_connect.py             # adb connect + devices
    │   ├── adb_install.py             # adb install APK (gate semi)
    │   ├── adb_shell.py               # adb shell (allowlist de cmds)
    │   ├── frida_attach.py            # frida list/attach/script (gate)
    │   ├── apktool_decode.py
    │   └── jadx_decompile.py
    ├── fixtures/                      # JSON MobSF report mínimo para testes
    └── tests/
        ├── conftest.py
        ├── test_tools_contract.py
        ├── test_mobsf_client.py
        ├── test_findings_post.py
        └── test_active_gate.py
```

**Compose fragment (docs + opcional arquivo):** `docker/runtimes/mobile/compose.mobsf.fragment.yml` (ou referência em EngMgr via 191). MobSF **não** vai baked na imagem `runtime-mobile` (já documentado no Dockerfile 186).

---

## Capabilities

| Tool / grupo | Capability canônica |
|---|---|
| Registro do server `mcp-mobile` | `pentest.mobile.dynamic` |
| `mobile_mobsf_static`, `mobile_apktool_decode`, `mobile_jadx_decompile` | `pentest.mobile.dynamic` (MVP: um flag para arsenal mobile) |
| `mobile_adb_*`, `mobile_frida_*`, `mobile_mobsf_dynamic` | `pentest.mobile.dynamic` |
| POST Findings | session key; leitura UI via `pentest.findings.view` (fora deste card) |

Sem capability → launcher **não** anexa `PENTEST_MCP_MOBILE_CMD` à sessão.

---

## Confirmation gate

Reusar `services/mcp-servers/shared/confirmation.py`.

| Tool | `semi_autonomous` |
|---|---|
| `mobile_mobsf_static`, decode/decompile, `adb_connect`, `adb_devices` | não |
| `mobile_adb_install`, `mobile_adb_shell` (cmds mutantes), `mobile_frida_attach`, `mobile_mobsf_dynamic` | **sim** → `confirmation_required` |

`PENTEST_AUTONOMY_MODE` só via env (nunca arg do agente).

---

## Contrato MCP — tools

| Tool | Args principais | Side-effect |
|---|---|---|
| `mobile_mobsf_static` | `engagement_id`, `apk_path` | upload MobSF + scan; POST findings |
| `mobile_mobsf_dynamic` | `engagement_id`, `apk_path?`, `package?` | dynamic; confirmation |
| `mobile_adb_connect` | `host?`, `port?` | `adb connect` (default env) |
| `mobile_adb_devices` | — | lista devices |
| `mobile_adb_install` | `engagement_id`, `apk_path` | install; confirmation |
| `mobile_adb_shell` | `engagement_id`, `command` | shell allowlisted; confirmation se mutante |
| `mobile_frida_list` | — | processos |
| `mobile_frida_attach` | `engagement_id`, `package`, `script?` | attach; confirmation |
| `mobile_apktool_decode` | `engagement_id`, `apk_path`, `out_dir?` | decode local |
| `mobile_jadx_decompile` | `engagement_id`, `apk_path`, `out_dir?` | decompile local |

Todas exigem `engagement_id` quando escrevem findings ou mutam device. Paths de APK: sob `PENTEST_WORKSPACE_DIR` / volume do engagement (path traversal → erro).

**ADB genérico:**

```
ADB_HOST=android-emulator   # ou host.docker.internal na Fase 3
ADB_PORT=5555
MOBSF_URL=http://mobsf:8000
MOBSF_API_KEY=…             # só env; nunca hardcoded
```

---

## MobSF client

```python
# mobsf_client.py — endpoints típicos da API REST MobSF
# POST /api/v1/upload  (multipart + Authorization: API key)
# POST /api/v1/scan
# POST /api/v1/report_json
# POST /api/v1/scorecard  (opcional)
```

Normalizar issues do report → Findings:

```json
{
  "engagement_id": "...",
  "source_tool": "mobsf",
  "title": "...",
  "description": "...",
  "severity": "critical|high|medium|low|info",
  "asset": "package.name|apk_filename",
  "endpoint": null,
  "evidence": { "raw": { } }
}
```

Auth Findings: `X-Session-API-Key`. Base: `FINDINGS_SERVICE_URL`. 409 → sucesso idempotente.

---

## Integração runtime Mobile

- Instalar pacote `mcp-mobile` no `docker/runtimes/mobile/` **somente o necessário** (COPY/uv sync), sem embutir MobSF/emulador.
- Documentar `PENTEST_MCP_MOBILE_CMD` no README dos MCP servers.
- Coordenar com **191** para hostname de serviço `mobsf` / `android-emulator` na rede interna do engagement.

---

## Critérios de aceite (QA)

1. **AC-190-1:** `mcp-mobile` lista ≥8 tools; schemas JSON válidos.
2. **AC-190-2:** `mobile_mobsf_static` com fixture/report mock → ≥1 POST Findings (201/409).
3. **AC-190-3:** Sem `MOBSF_API_KEY` → erro estruturado (não crash).
4. **AC-190-4:** Path fora de `PENTEST_WORKSPACE_DIR` → rejeitado.
5. **AC-190-5:** Tools com gate em `semi_autonomous` sem token → `confirmation_required`.
6. **AC-190-6:** Com token → executa (mock adb/mobsf ok).
7. **AC-190-7:** Client MobSF unit-tested (httpx mock); zero segredo no repo.
8. **AC-190-8:** `REQUIRED_CAPABILITY = "pentest.mobile.dynamic"` documentado + README atualizado.
9. **AC-190-9:** Fragmento/docs MobSF compose (sem publicar porta host em produção — só rede internal; UI sobe via proxy 192).

---

## Segurança (AppSec)

- `MOBSF_API_KEY` / session key só env.
- Não logar APK bytes nem report completo em INFO.
- `adb_shell`: allowlist (ex. `pm list`, `am start`, `logcat -d`); bloquear `rm -rf /`, reboot destrutivo sem gate.
- Fail-closed se `MOBSF_URL` ausente quando tool MobSF é chamada.
- MobSF e emulator **não** bind `0.0.0.0` no host no template de produção (ver 191).

---

## Dependências

- **Depende de:** PROJETOSIN-184 (Findings), PROJETOSIN-182 (capabilities), PROJETOSIN-186 (runtime-mobile).
- **Paralelo:** 191 (compose emulator/MobSF), Design-192.
- **Contrato para 192:** upload UI chama MobSF (via API eng/proxy ou tool); findings aparecem no painel 188.

**Estimativa:** 4–5 dias
