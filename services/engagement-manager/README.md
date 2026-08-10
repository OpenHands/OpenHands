# services/engagement-manager

Engagement Manager — provisionamento e ciclo de vida de sandboxes isolados por engagement.

**ADR:** ADR-0001 (accepted)  
**Cards:** PROJETOSIN-185 · PROJETOSIN-191 (mobile emulator + MobSF)  
**Specs:** `docs/specs/fase-0/185-engagement-manager.md` · `docs/specs/fase-2/191-android-emulator-engmgr.md`

## Stack

- Python 3.12 + FastAPI + SQLAlchemy (async) + Alembic
- PostgreSQL 16
- Docker socket: `/var/run/docker.sock` (provisiona compose por engagement)
- Porta: 18003 (`config/defaults.json` → `ports.engagementManager`)

## Setup (executor — backend + devops)

```bash
cd services/engagement-manager
uv venv && uv pip install -e ".[dev]"
# Subir DB local:
docker compose -f docker-compose.fragment.yml up engmgr-db -d
# Migrations:
alembic upgrade head
# Dev server:
uvicorn app.main:app --reload --port 18003
```

## Endpoints principais

- `GET /api/pentest/engagements` — listar engagements
- `POST /api/pentest/engagements` — criar engagement
- `POST /api/pentest/engagements/{id}/authorize-scope` — registrar RoE + allowlist
- `POST /api/pentest/engagements/{id}/provision` — provisionar sandbox Docker
- `POST /api/pentest/engagements/{id}/teardown` — derrubar sandbox (`compose down -v`)

## Runtime profiles

| Profile | Template | Sidecars |
|---------|----------|----------|
| `web` | `compose-web-runtime.yml.j2` | egress-proxy |
| `network` | `compose-network-runtime.yml.j2` | (perfil) |
| `mobile` | `compose-mobile-runtime.yml.j2` | **android-emulator** + **mobsf** (rede internal) |
| `sast` | `compose-sast-runtime.yml.j2` | (perfil) |

### Mobile (PROJETOSIN-191)

Ao provisionar `runtime_profile=mobile`, o provisioner renderiza três serviços na rede `*-internal` (runtime também em `*-egress`):

1. `eng-…-runtime` — `ghcr.io/heimdall/runtime-mobile:latest` com `ADB_HOST` / `ADB_PORT` / `MOBSF_URL` / `MOBSF_API_KEY`
2. `eng-…-emulator` — pin `images.androidEmulator` em `config/defaults.json` (`budtmo/docker-android:emulator_13.0`); `privileged: true`; **sem** `ports:` no host
3. `eng-…-mobsf` — pin `images.mobsf`; volume `*-mobsf-data` (removido no teardown `-v`)

Metadata para o proxy UI (card 192) é gravada em `<compose_work>/<project>/runtime-metadata.json` (`adb`, `vnc_internal`, `mobsf.url_internal`). A GUI **não** é publicada no host neste card.

| Host | Comportamento |
|------|----------------|
| Linux com `/dev/kvm` | `devices: [/dev/kvm]` + privileged |
| Sem KVM | Fail-fast no provision log, a menos que `ALLOW_SLOW_EMULATOR=1` (`EMULATOR_ACCEL=false`, software — muito lento) |
| Windows/mac Docker Desktop | Hypervisor do Desktop; aceleração KVM nativa ausente — ver `docker/runtimes/README.md` (Fase 3 Electron) |

Boot do emulador: tipicamente **60–180s** (`healthcheck.start_period: 180s`). Dry-run (`PROVISIONER_DRY_RUN=true`) usa `MOBSF_API_KEY=test-mobsf-key` e **não** loga a chave.

Overrides de imagem: `ANDROID_EMULATOR_IMAGE`, `MOBSF_IMAGE`.

## Segurança CRÍTICA

O container monta `/var/run/docker.sock`. Nunca expor este serviço diretamente para internet.
Gate AppSec obrigatório antes de merge (ver AGENTS.md).
Privileged **somente** no serviço emulator; runtime permanece non-root. MobSF/VNC só na rede Docker internal.