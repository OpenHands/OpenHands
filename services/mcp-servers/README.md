# MCP Servers — Pentest (PROJETOSIN-187)

stdio MCP servers for the Web offensive runtime. **ADR-0001**.

```
services/mcp-servers/
├── shared/          # findings client, session auth, confirmation, normalize/scope
├── mcp-recon/       # subfinder / httpx / reconftw
└── mcp-webscan/     # ZAP / Nuclei / Wapiti / Nikto / sqlmap
```

## Capabilities

| Server / tools | Capability |
|----------------|------------|
| `mcp-recon` (all) | `pentest.recon.run` |
| webscan passive (spider, passive ZAP, nuclei default, wapiti, nikto) | `pentest.scan.passive` |
| webscan active (`web_zap_active_scan`, `web_sqlmap_run`, nuclei intrusive) | `pentest.scan.active` |

Session registration should only attach a server when the authenticated profile
has the minimum capability (Fase 0 RBAC).

## Environment

| Variable | Purpose |
|----------|---------|
| `SESSION_API_KEY` | Sent as `X-Session-API-Key` to Findings Service |
| `FINDINGS_SERVICE_URL` | Default `http://findings-service:8000` |
| `PENTEST_SCOPE_ALLOWLIST` | CSV of hosts/CIDRs (fail-closed if empty) |
| `PENTEST_AUTONOMY_MODE` | Server-side only: `manual` \| `semi_autonomous` \| `autonomous` (default semi). Never taken from agent tool args. |
| `OPENHANDS_CONFIRMATION_TOKEN` | Optional env token after UI approval |
| `PENTEST_MCP_RECON_CMD` | Override launch command for mcp-recon |
| `PENTEST_MCP_WEBSCAN_CMD` | Override launch command for mcp-webscan |
| `MCP_WEBSCAN_TIMEOUT_SEC` | Timeout for intrusive tools (default 300) |

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
```

## Register with Agent Canvas / Agent Server

Until workspace-type hooks auto-register MCP for `pentest` workspaces, set
stdio commands via settings / MCP API or env:

```bash
PENTEST_MCP_RECON_CMD='python /opt/mcp-servers/mcp-recon/server.py'
PENTEST_MCP_WEBSCAN_CMD='python /opt/mcp-servers/mcp-webscan/server.py'
```

Example `config.toml` fragment (engagement):

```toml
[mcp.mcp-recon]
command = "python"
args = ["/opt/mcp-servers/mcp-recon/server.py"]

[mcp.mcp-webscan]
command = "python"
args = ["/opt/mcp-servers/mcp-webscan/server.py"]
```

## Confirmation gate (stub)

Intrusive tools (`zap_active_scan`, `sqlmap_run`, `nuclei_intrusive`) in
`semi_autonomous` mode return:

```json
{"ok": false, "error": "confirmation_required", "request_id": "..."}
```

Approve via `shared.confirmation.approve_confirmation(request_id)` (test/stub)
or set `OPENHANDS_CONFIRMATION_TOKEN` / pass `confirmation_token` on re-run.

## Tests

```bash
cd services/mcp-servers/mcp-recon && PYTHONPATH=..:. pytest -q
cd ../mcp-webscan && PYTHONPATH=..:. pytest -q
```
