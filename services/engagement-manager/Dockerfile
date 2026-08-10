# Build from services/ directory:
#   docker build -f engagement-manager/Dockerfile -t engagement-manager ..
FROM python:3.12-slim

WORKDIR /app

COPY engagement-manager/pyproject.toml ./
COPY engagement-manager/app ./app
COPY engagement-manager/alembic ./alembic
COPY engagement-manager/alembic.ini ./
COPY shared /services/shared

RUN pip install --no-cache-dir .

ENV PYTHONPATH=/services
ENV ENGMGR_DB_URL=postgresql+asyncpg://engmgr:engmgr@engmgr-db/engmgr
ENV PROVISIONER_DRY_RUN=true

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
