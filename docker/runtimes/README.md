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

## Mobile sidecars (not in this image)

MobSF and the Android emulator stay **out of** `runtime-mobile` to keep the image slim
and avoid privileged tooling in the main agent workspace container.

```yaml
# fragment compose mobile
  mobsf:
    image: opensecurity/mobile-security-framework-mobsf:latest
    environment:
      - MOBSF_API_KEY=${MOBSF_API_KEY}
    ports:
      - "8093:8000"
    volumes:
      - mobsf-data:/home/mobsf/.MobSF

  android-emulator:
    image: budtmo/docker-android:emulator_13.0
    privileged: true
    ports:
      - "5555:5555"   # ADB over TCP
      - "6901:6901"   # noVNC (GUI web)
    environment:
      - DEVICE=Samsung Galaxy S10
```

Only the emulator sidecar should be privileged — not `runtime-mobile`.

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

## Size targets (AC-186-6)

| Image | Soft limit |
|-------|------------|
| web / mobile / sast | &lt; 2 GB soft target (web may land ~2.1–2.3 GB with full ZAP + JDK) |
| network | ≤ 4 GB (Metasploit exception) |

Measured local builds (approx.): mobile/sast &lt; 1 GB; network ~1.9 GB; web ~2.2 GB (ZAP Linux bundle).
