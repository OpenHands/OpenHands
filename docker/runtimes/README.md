# Offensive runtime images (PROJETOSIN-186)

Slim, domain-scoped Docker images provisioned by the Engagement Manager
(`services/engagement-manager` → `ghcr.io/heimdall/runtime-{profile}:latest`).

| Profile | Image | Arsenal (in image) | Healthz |
|---------|-------|--------------------|---------|
| web | `ghcr.io/heimdall/runtime-web` | ZAP, Nuclei, Wapiti, Nikto, sqlmap | `:8090/healthz` |
| network | `ghcr.io/heimdall/runtime-network` | nmap, Metasploit (`msfconsole` / `msfrpcd`) | `:8091/healthz` |
| mobile | `ghcr.io/heimdall/runtime-mobile` | adb, Frida tools, apktool, jadx | `:8092/healthz` |
| sast | `ghcr.io/heimdall/runtime-sast` | Semgrep, Trivy | `:8094/healthz` |

**ADR:** `docs/adrs/0001-plataforma-pentest-ia-extensao-openhands.md`  
**Spec:** `docs/specs/fase-0/186-dockerfiles-runtimes.md`

## GHCR package permissions

CI pushes to `ghcr.io/heimdall/runtime-*`. That namespace requires packages under the
`heimdall` GitHub org (or a user/org that owns the `heimdall` GHCR path). On forks
without org package write access, builds still succeed locally and in CI; **push is
limited to `main`** and may fail until package permissions / org linkage are granted.
Tags remain exactly:

- `ghcr.io/heimdall/runtime-web:latest` (+ `sha-<commit>`)
- `ghcr.io/heimdall/runtime-network:latest` (+ `sha-<commit>`)
- `ghcr.io/heimdall/runtime-mobile:latest` (+ `sha-<commit>`)
- `ghcr.io/heimdall/runtime-sast:latest` (+ `sha-<commit>`)

Do not rename tags without updating Engagement Manager provisioner templates.

## Build locally

```bash
docker build -t ghcr.io/heimdall/runtime-web:latest docker/runtimes/web
docker build -t ghcr.io/heimdall/runtime-network:latest docker/runtimes/network
docker build -t ghcr.io/heimdall/runtime-mobile:latest docker/runtimes/mobile
docker build -t ghcr.io/heimdall/runtime-sast:latest docker/runtimes/sast
```

Smoke checks (AC-186):

```bash
docker run --rm --entrypoint zap ghcr.io/heimdall/runtime-web:latest -version
docker run --rm --entrypoint nuclei ghcr.io/heimdall/runtime-web:latest -version
docker run --rm --entrypoint nmap ghcr.io/heimdall/runtime-network:latest --version
docker run --rm --entrypoint msfconsole ghcr.io/heimdall/runtime-network:latest -v
docker run --rm --entrypoint adb ghcr.io/heimdall/runtime-mobile:latest version
docker run --rm --entrypoint apktool ghcr.io/heimdall/runtime-mobile:latest --version
docker run --rm --entrypoint semgrep ghcr.io/heimdall/runtime-sast:latest --version
docker run --rm --entrypoint trivy ghcr.io/heimdall/runtime-sast:latest -v
```

## Secrets (runtime env only)

| Variable | Image | Purpose |
|----------|-------|---------|
| `MSF_PASSWORD` | network | Password for `msfrpcd` (required to enable RPC; never baked into the image) |
| `MOBSF_API_KEY` | MobSF **sidecar** | API key for MobSF container (not used by `runtime-mobile` itself) |

## Privilege model

- **web / mobile / sast:** run as non-root user `runtime` (uid 1000).
- **network:** runs as **root**. Metasploit RPC and OpenVAS/GVM-style tooling expect elevated privileges; document this for AppSec reviews.

## Network / OpenVAS-GVM notes

Debian `bookworm-slim` does not provide a turnkey GVM (gvmd + PostgreSQL + Redis +
NVT feeds) without pulling a large service stack and long first-boot feed sync.
The network Dockerfile:

1. Always installs **nmap** and **Metasploit Framework** (AC-186-3).
2. Attempts `openvas-scanner` + `gvm-tools` from Debian apt when resolvable.
3. Does **not** auto-start `gvmd` / feed sync in the entrypoint.

For production Greenbone/OpenVAS, prefer the official Greenbone community containers
as a sidecar next to `runtime-network`, similar to MobSF for mobile.

Metasploit RPC (optional):

```bash
docker run --rm -e MSF_PASSWORD='<from-secret-store>' -p 55553:55553 -p 8091:8091 \
  ghcr.io/heimdall/runtime-network:latest
```

## Mobile sidecars (PROJETOSIN-191 — EngMgr compose)

MobSF and the Android emulator stay **out of** `runtime-mobile` to keep the image slim
and avoid privileged tooling in the main agent workspace container. Canonical template:
`services/engagement-manager/app/templates/compose-mobile-runtime.yml.j2`.

Image pins (source of truth): `config/defaults.json` → `images.androidEmulator`,
`images.mobsf`.

**Production MobSF fragment (no host port publish):** see
`docker/runtimes/mobile/compose.mobsf.fragment.yml` (PROJETOSIN-190). Reach MobSF at
`http://mobsf:8000` on the engagement network; UI/proxy is PROJETOSIN-192.

```yaml
# Production EngMgr fragment — NO host port publish (AC-191-3).
# ADB :5555 and noVNC :6080 are reachable only on the engagement internal network.
# Authenticated UI proxy is PROJETOSIN-192 (/api/emulator).
  eng-xxxx-emulator:
    image: budtmo/docker-android:emulator_13.0   # defaults.json images.androidEmulator
    privileged: true
    devices: ["/dev/kvm"]                          # omit + EMULATOR_ACCEL=false if ALLOW_SLOW_EMULATOR
    environment:
      EMULATOR_DEVICE: "Samsung Galaxy S10"
      WEB_VNC: "true"
    networks: [eng-xxxx-internal]
    # no ports:

  eng-xxxx-mobsf:
    image: opensecurity/mobile-security-framework-mobsf:latest
    environment:
      MOBSF_API_KEY: "<injected-by-provisioner>"
    volumes:
      - eng-xxxx-mobsf-data:/home/mobsf/.MobSF
    networks: [eng-xxxx-internal]
```

Only the emulator sidecar should be privileged — not `runtime-mobile`.

`runtime-mobile` installs MCP SDK deps (`mcp`, `httpx`) via `requirements.txt`.
Mount `services/mcp-servers` at `/opt/mcp-servers` and set:

```bash
PENTEST_MCP_MOBILE_CMD='python /opt/mcp-servers/mcp-mobile/server.py'
MOBSF_URL=http://mobsf:8000
MOBSF_API_KEY=<from-secret-store>
ADB_HOST=android-emulator
ADB_PORT=5555
```

### KVM / boot / fallback

| Host | Behavior |
|------|----------|
| Linux + `/dev/kvm` | Preferred: `devices: [/dev/kvm]` + `privileged: true` |
| Linux without KVM | EngMgr fail-fast unless `ALLOW_SLOW_EMULATOR=1` (software accel — often 5–15+ min) |
| Docker Desktop (Win/mac) | Nested virt / hypervisor; treat as slow path; physical device bridge = Fase 3 |

Cold boot of `budtmo/docker-android` is commonly **60–180 seconds** before `adb devices`
shows `device`. Runtime clients should retry ADB connect; EngMgr healthcheck uses
`start_period: 180s`.

Teardown: `docker compose … down -v` removes the MobSF named volume for that project.

## CI

Workflow: `.github/workflows/docker-runtimes.yml`

- Matrix: `web`, `network`, `mobile`, `sast`
- Build + **Trivy image scan** before push
- Push to GHCR only on `main` (`latest` + `sha-<sha>`)
- GHA layer cache (`cache-from` / `cache-to` type=gha)

## Build notes (pins)

- Base images: `python:3.12-slim-bookworm` (web/sast) and `debian:bookworm-slim` (network/mobile). Rolling `python:3.12-slim` tracks Debian trixie and drops packages such as `openjdk-17`.
- ZAP pinned to **2.17.0** (SHA-256 verified); older 2.15.0 Linux tarball is no longer published.
- Trivy pinned to **0.73.0** (0.57.0 asset returned 404).
- Nikto installed from upstream git (`sullo/nikto`) — not in Debian bookworm main.
- Nuclei pinned to **3.3.9** (linux amd64).

## MCP servers (PROJETOSIN-187)

`runtime-web` installs the Python MCP SDK (`mcp`, `httpx`) via `requirements.txt`.
The stdio servers live in-repo at `services/mcp-servers/` (`mcp-recon`, `mcp-webscan`)
and are launched on demand by the agent-server — **not** by `entrypoint.sh`.

Mount or bake the package tree at `/opt/mcp-servers` (compose volume is fine for MVP):

```yaml
volumes:
  - ./services/mcp-servers:/opt/mcp-servers:ro
environment:
  PENTEST_MCP_RECON_CMD: "python /opt/mcp-servers/mcp-recon/server.py"
  PENTEST_MCP_WEBSCAN_CMD: "python /opt/mcp-servers/mcp-webscan/server.py"
  PENTEST_SCOPE_ALLOWLIST: "example.com,10.0.0.0/8"
  FINDINGS_SERVICE_URL: "http://findings-service:8000"
  SESSION_API_KEY: "${SESSION_API_KEY}"
```

Example Agent Server / engagement `config.toml` fragment:

```toml
[mcp.mcp-recon]
command = "python"
args = ["/opt/mcp-servers/mcp-recon/server.py"]

[mcp.mcp-webscan]
command = "python"
args = ["/opt/mcp-servers/mcp-webscan/server.py"]
```

Attach `mcp-recon` only when the session has `pentest.recon.run`; attach passive
webscan tools with `pentest.scan.passive` and active tools with `pentest.scan.active`.

## Size targets (AC-186-6)

| Image | Soft limit |
|-------|------------|
| web / mobile / sast | &lt; 2 GB soft target (web may land ~2.1–2.3 GB with full ZAP + JDK) |
| network | ≤ 4 GB (Metasploit exception) |

Measured local builds (approx.): mobile/sast &lt; 1 GB; network ~1.9 GB; web ~2.2 GB (ZAP Linux bundle).
