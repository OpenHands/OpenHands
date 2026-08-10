# services/findings-service

Findings Service — fonte de verdade de vulnerabilidades da Plataforma de Pentest com IA.

**ADR:** ADR-0001 (accepted)
**Card:** PROJETOSIN-184
**Spec:** `docs/specs/fase-0/184-findings-service.md`

## Stack

- Python 3.12 + FastAPI + SQLAlchemy (async) + Alembic
- PostgreSQL 16
- Porta: 18002

## Setup (executor — backend)

```bash
cd services/findings-service
uv venv && uv pip install -e ".[dev]"
# Subir DB local:
docker compose -f docker-compose.fragment.yml up findings-db -d
# Migrations:
alembic upgrade head
# Dev server:
uvicorn app.main:app --reload --port 18002
```

## Endpoints principais

- `GET /api/pentest/findings?engagement_id=<uuid>` — listar findings
- `POST /api/pentest/findings` — criar finding (via MCP ou ferramenta)
- `POST /api/pentest/findings/{id}/triage` — triar / marcar FP
- `POST /api/pentest/findings/sync-defectdojo` — push para DefectDojo produção

Ver spec completa para contratos detalhados.
