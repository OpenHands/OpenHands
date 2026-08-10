# services/engagement-manager

Engagement Manager — provisionamento e ciclo de vida de sandboxes isolados por engagement.

**ADR:** ADR-0001 (accepted)
**Card:** PROJETOSIN-185
**Spec:** `docs/specs/fase-0/185-engagement-manager.md`

## Stack

- Python 3.12 + FastAPI + SQLAlchemy (async) + Alembic
- PostgreSQL 16
- Docker socket: `/var/run/docker.sock` (provisiona compose por engagement)
- Porta: 18003

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
- `POST /api/pentest/engagements/{id}/teardown` — derrubar sandbox

## Segurança CRÍTICA

O container monta `/var/run/docker.sock`. Nunca expor este serviço diretamente para internet.
Gate AppSec obrigatório antes de merge (ver AGENTS.md).
