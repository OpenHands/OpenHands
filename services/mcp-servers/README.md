# MCP Servers — Pentest (PROJETOSIN-187 / 189 / 190)

stdio MCP servers for offensive runtimes. **ADR-0001**.

```
services/mcp-servers/
├── shared/          # findings client, session auth, confirmation, normalize/scope
├── mcp-recon/       # subfinder / httpx / reconftw
├── mcp-webscan/     # ZAP / Nuclei / Wapiti / Nikto / sqlmap
├── mcp-sast/        # Semgrep + Trivy (PROJETOSIN-189)
└── mcp-mobile/      # MobSF + ADB/Frida/apktool/jadx (PROJETOSIN-190)
```

## Capabilities

| Server / tools | Capability |
|----------------|------------|
| `mcp-recon` (all) | `pentest.recon.run` |
| webscan passive (spider, passive ZAP, nuclei default, wapiti, nikto) | `pentest.scan.passive` |
| webscan active (`web_zap_active_scan`, `web_sqlmap_run`, nuclei intrusive) | `pentest.scan.active` |
| `mcp-sast` (Semgrep / Trivy) | `pentest.sast.run` |
| `mcp-mobile` (MobSF / ADB / Frida / apktool / jadx) | `pentest.mobile.dynamic` |

Session registration should only attach a server when the authenticated profile
has the minimum capability (Fase 0 RBAC). Without `pentest.mobile.dynamic`, the
launcher must **not** attach `PENTEST_MCP_MOBILE_CMD`.

## Environment

| Variable | Purpose |
|----------|---------|
| `SESSION_API_KEY` | Sent as `X-Session-API-Key` to Findings Service |
| `FINDINGS_SERVICE_URL` | Default `http://findings-service:8000` |
| `PENTEST_SCOPE_ALLOWLIST` | CSV of hosts/CIDRs (fail-closed if empty) |
| `PENTEST_WORKSPACE_DIR` | Workspace root for path guard (default `/workspace/project`) |
| `PENTEST_AUTONOMY_MODE` | Server-side only: `manual` \| `semi_autonomous` \| `autonomous` (default semi). Never taken from agent tool args. |
| `OPENHANDS_CONFIRMATION_TOKEN` | Optional env token after UI approval |
| `PENTEST_MCP_RECON_CMD` | Override launch command for mcp-recon |
| `PENTEST_MCP_WEBSCAN_CMD` | Override launch command for mcp-webscan |
| `PENTEST_MCP_SAST_CMD` | Override launch command for mcp-sast |
| `PENTEST_MCP_MOBILE_CMD` | Override launch command for mcp-mobile |
| `MOBSF_URL` | MobSF base URL (e.g. `http://mobsf:8000`); required for MobSF tools |
| `MOBSF_API_KEY` | MobSF API key (env only; never hardcoded) |
| `ADB_HOST` | Generic ADB endpoint host (default `android-emulator`) |
| `ADB_PORT` | Generic ADB endpoint port (default `5555`) |
| `MCP_WEBSCAN_TIMEOUT_SEC` | Timeout for intrusive web tools (default 300) |
| `MCP_MOBILE_USE_REAL_BINARIES` | Set `1` to invoke real `adb`/`apktool`/`jadx`/`frida` |
| `DEFECTDOJO_API_URL` | One-way DefectDojo mirror base URL |
| `DEFECTDOJO_API_TOKEN` | DefectDojo API token |
| `DEFECTDOJO_PRODUCT_TYPE_DEFAULT` | Default product type (e.g. `Pentest`) |
| `DEFECTDOJO_VERIFY_TLS` | TLS verify for DefectDojo (`true`/`false`) |
| `DEFECTDOJO_DRY_RUN` | `1` to skip real DefectDojo writes |

## Local run

```bash
# from repo root — PYTHONPATH must include services/mcp-servers
export PYTHONPATH=services/mcp-servers:services/mcp-servers/mcp-recon
export PENTEST_SCOPE_ALLOWLIST=example.com
export SESSION_API_KEY=dev-key
export FINDINGS_SERVICE_URL=http://127.0.0.1:18002
python services/mcp-servers/mcp-recon/server.py

export PYTHONPATH=services/mcp-servers:services/mcp-servers/mcp-webscan
python services/mcp-servers/mcp-webscan/server.py

export PYTHONPATH=services/mcp-servers:services/mcp-servers/mcp-sast
export PENTEST_WORKSPACE_DIR=/workspace/project
python services/mcp-servers/mcp-sast/server.py

export PYTHONPATH=services/mcp-servers:services/mcp-servers/mcp-mobile
export MOBSF_URL=http://mobsf:8000
export MOBSF_API_KEY=dev-mobsf-key
export ADB_HOST=android-emulator
export ADB_PORT=5555
python services/mcp-servers/mcp-mobile/server.py
```

## Register with Agent Canvas / Agent Server

Until workspace-type hooks auto-register MCP for `pentest` workspaces, set
stdio commands via settings / MCP API or env:

```bash
PENTEST_MCP_RECON_CMD='python /opt/mcp-servers/mcp-recon/server.py'
PENTEST_MCP_WEBSCAN_CMD='python /opt/mcp-servers/mcp-webscan/server.py'
PENTEST_MCP_SAST_CMD='python /opt/mcp-servers/mcp-sast/server.py'
PENTEST_MCP_MOBILE_CMD='python /opt/mcp-servers/mcp-mobile/server.py'
```

Example `config.toml` fragment (engagement):

```toml
[mcp.mcp-recon]
command = "python"
args = ["/opt/mcp-servers/mcp-recon/server.py"]

[mcp.mcp-webscan]
command = "python"
args = ["/opt/mcp-servers/mcp-webscan/server.py"]

[mcp.mcp-sast]
command = "python"
args = ["/opt/mcp-servers/mcp-sast/server.py"]

[mcp.mcp-mobile]
command = "python"
args = ["/opt/mcp-servers/mcp-mobile/server.py"]
```

## Confirmation gate (stub)

Intrusive tools (`zap_active_scan`, `sqlmap_run`, `nuclei_intrusive`,
`mobsf_dynamic`, `adb_install`, `adb_shell_mutant`, `frida_attach`) in
`semi_autonomous` mode return:

```json
{"ok": false, "error": "confirmation_required", "request_id": "..."}
```

Approve via `shared.confirmation.approve_confirmation(request_id)` (test/stub)
or set `OPENHANDS_CONFIRMATION_TOKEN` / pass `confirmation_token` on re-run.

## MobSF sidecar

Compose fragment (internal network only — no host port publish in production):
`docker/runtimes/mobile/compose.mobsf.fragment.yml`. Emulator lifecycle is
PROJETOSIN-191; UI/upload proxy is PROJETOSIN-192.

## Tests

```bash
cd services/mcp-servers/mcp-recon && PYTHONPATH=..:. pytest -q
cd ../mcp-webscan && PYTHONPATH=..:. pytest -q
cd ../mcp-sast && PYTHONPATH=..:. pytest -q
cd ../mcp-mobile && PYTHONPATH=..:. pytest -q
```
