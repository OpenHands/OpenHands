# Build from services/ directory:
#   docker build -f findings-service/Dockerfile -t findings-service ..
FROM python:3.12-slim

WORKDIR /app

COPY findings-service/pyproject.toml ./
COPY findings-service/app ./app
COPY findings-service/alembic ./alembic
COPY findings-service/alembic.ini ./
COPY shared /services/shared

RUN pip install --no-cache-dir .

ENV PYTHONPATH=/services
ENV FINDINGS_DB_URL=postgresql+asyncpg://findings:findings@findings-db/findings

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
