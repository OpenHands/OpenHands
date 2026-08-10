# Spec Técnica — PROJETOSIN-184: Findings Service (URGENT)

**ADR:** docs/adrs/0001-plataforma-pentest-ia-extensao-openhands.md (accepted)
**Card Plane:** PROJETOSIN-184 — `31387141-2b4c-4c0b-a6e4-4f3a1942fb49`
**Agente responsável:** backend
**Prioridade:** P0 URGENT — fonte de verdade de findings, desbloqueia integração DefectDojo

---

## Boundary e localização

Vive em `services/findings-service/` neste repositório (Agent Canvas). É um serviço FastAPI + Postgres independente, containerizado, sem acoplar ao `openhands-automation`.

```
services/findings-service/
├── Dockerfile
├── pyproject.toml (uv / hatch)
├── alembic/
│   ├── env.py
│   └── versions/
│       └── 001_initial_schema.py
├── app/
│   ├── main.py
│   ├── config.py
│   ├── models/
│   │   ├── finding.py       # SQLAlchemy ORM
│   │   └── engagement.py    # FK reference only (owned by EngMgr)
│   ├── schemas/
│   │   ├── finding.py       # Pydantic v2 request/response
│   │   └── sync.py          # DefectDojo sync payload
│   ├── routers/
│   │   ├── findings.py      # CRUD + status transitions
│   │   ├── triage.py        # FP workflow
│   │   └── sync.py          # Push to DefectDojo
│   ├── services/
│   │   ├── findings_service.py
│   │   ├── dedup_service.py
│   │   └── defectdojo_sync.py
│   └── middleware/
│       └── auth.py          # X-Session-API-Key + capabilities
└── tests/
    ├── test_findings_crud.py
    ├── test_triage.py
    └── test_defectdojo_sync.py
```

---

## Modelo de dados (PostgreSQL)

```sql
-- migrations/001_initial_schema.py

CREATE TABLE findings (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    engagement_id   UUID NOT NULL,          -- FK lógica para EngMgr (PROJETOSIN-185)
    source_tool     VARCHAR(64) NOT NULL,   -- 'nuclei','zap','nmap','cai','mobsf', etc.
    title           TEXT NOT NULL,
    description     TEXT,
    severity        VARCHAR(16) NOT NULL,   -- 'critical','high','medium','low','info'
    asset           TEXT,                   -- IP, domínio, endpoint
    endpoint        TEXT,                   -- URL/path específico
    evidence        JSONB,                  -- { "request": "", "response": "", "screenshot_url": "" }
    status          VARCHAR(32) NOT NULL DEFAULT 'new',
                    -- 'new' | 'triaging' | 'confirmed' | 'false_positive'
                    -- | 'duplicate' | 'risk_accepted'
    dedupe_hash     VARCHAR(64),            -- SHA-256 de (engagement_id+title+asset+endpoint)
    fp_reason       TEXT,
    triaged_by      VARCHAR(256),           -- user identifier
    triaged_at      TIMESTAMPTZ,
    defectdojo_id   INTEGER,               -- ID no DD após sync
    defectdojo_synced_at TIMESTAMPTZ,
    cvss_score      DECIMAL(4,1),
    cve_ids         TEXT[],
    tags            TEXT[],
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT findings_status_check CHECK (
        status IN ('new','triaging','confirmed','false_positive','duplicate','risk_accepted')
    ),
    CONSTRAINT findings_severity_check CHECK (
        severity IN ('critical','high','medium','low','info')
    )
);

CREATE INDEX idx_findings_engagement_id ON findings(engagement_id);
CREATE INDEX idx_findings_status ON findings(status);
CREATE INDEX idx_findings_severity ON findings(severity);
CREATE UNIQUE INDEX idx_findings_dedupe ON findings(dedupe_hash) WHERE dedupe_hash IS NOT NULL;
```

---

## API — Endpoints

### Base path: `/api/pentest/findings`

| Method | Path | Capability | Descrição |
|--------|------|------------|-----------|
| GET | `/` | `pentest.findings.view` | Listar findings (paginado, filtros) |
| POST | `/` | `pentest.scan.passive` | Criar finding (via MCP ou tool) |
| GET | `/{id}` | `pentest.findings.view` | Detalhe de finding |
| PATCH | `/{id}` | `pentest.findings.triage` | Atualizar status/triage |
| DELETE | `/{id}` | `pentest.admin.users` | Remover (admin only) |
| POST | `/{id}/triage` | `pentest.findings.triage` | Marcar FP / confirmar |
| POST | `/sync-defectdojo` | `pentest.findings.export_dd` | Push para DefectDojo |
| GET | `/stats` | `pentest.findings.view` | Contagem por severity/status |
| GET | `/capabilities` | (qualquer usuário autenticado) | Capabilities do usuário |

### GET /api/pentest/findings

Query params: `engagement_id` (obrigatório), `status`, `severity`, `source_tool`, `page`, `page_size` (max 100)

**Response 200:**
```json
{
  "items": [{ ...finding }],
  "total": 42,
  "page": 1,
  "page_size": 20,
  "next_page": 2
}
```

### POST /api/pentest/findings (criar finding)

```json
{
  "engagement_id": "uuid",
  "source_tool": "nuclei",
  "title": "SQL Injection in /api/search",
  "description": "...",
  "severity": "high",
  "asset": "target.heimdall.local",
  "endpoint": "/api/search?q=",
  "evidence": {
    "request": "GET /api/search?q=' OR 1=1...",
    "response": "500 Internal Server Error..."
  }
}
```

**Response 201:** `{ ...finding, id: "uuid" }`

Auto-calcula `dedupe_hash` = SHA-256(`engagement_id:title:asset:endpoint`). Se duplicado, retorna 409 com `existing_finding_id`.

### POST /api/pentest/findings/{id}/triage

```json
{
  "new_status": "false_positive",
  "fp_reason": "Ambiente de desenvolvimento, não aplicável em produção",
  "triaged_by": "user@heimdall.com"
}
```

**Transições válidas:**
```
new → triaging → confirmed | false_positive | duplicate | risk_accepted
triaging → confirmed | false_positive | duplicate | risk_accepted
confirmed → false_positive (re-triage) | risk_accepted
```

### POST /api/pentest/findings/sync-defectdojo

```json
{
  "engagement_id": "uuid",
  "status_filter": ["confirmed"]
}
```

Dispara sync assíncrono. Retorna `{ "job_id": "uuid", "status": "queued" }`.

---

## DefectDojo Sync Service

```python
# services/findings_service/app/services/defectdojo_sync.py

class DefectDojoSyncService:
    DD_API_BASE: str         # de config, apontando para DD produção Heimdall
    DD_API_TOKEN: str        # de secrets (env var DEFECTDOJO_API_TOKEN)

    async def sync_finding(self, finding: Finding) -> int:
        """
        Usa /reimport-scan com scan_type='Generic Findings Import'
        para findings sem parser nativo no DD.
        Retorna defectdojo_id.
        """

    async def sync_engagement_findings(
        self,
        engagement_id: str,
        status_filter: list[str] = ["confirmed"]
    ) -> SyncResult:
        """Batch sync de todos os findings confirmados de um engagement."""

    def _build_generic_finding_payload(self, finding: Finding) -> dict:
        """Formata para DefectDojo Generic Findings Import JSON."""
```

**Config via env vars:**
```
DEFECTDOJO_API_URL=https://defectdojo.heimdall.local
DEFECTDOJO_API_TOKEN=<token>
FINDINGS_DB_URL=postgresql+asyncpg://user:pass@localhost/findings
SESSION_API_KEY=<shared com agent server>
```

---

## Dedup Service

```python
# services/findings_service/app/services/dedup_service.py
import hashlib

def compute_dedupe_hash(engagement_id: str, title: str, asset: str, endpoint: str) -> str:
    key = f"{engagement_id}:{title}:{asset}:{endpoint}"
    return hashlib.sha256(key.encode()).hexdigest()[:64]
```

---

## Docker Compose (fragment — integrar em docker-compose.yml raiz)

```yaml
# services/findings-service/docker-compose.fragment.yml
services:
  findings-service:
    build:
      context: ./services/findings-service
    environment:
      - FINDINGS_DB_URL=postgresql+asyncpg://findings:${FINDINGS_DB_PASSWORD}@findings-db/findings
      - SESSION_API_KEY=${SESSION_API_KEY}
      - DEFECTDOJO_API_URL=${DEFECTDOJO_API_URL}
      - DEFECTDOJO_API_TOKEN=${DEFECTDOJO_API_TOKEN}
    depends_on:
      findings-db:
        condition: service_healthy
    ports:
      - "18002:8000"

  findings-db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: findings
      POSTGRES_USER: findings
      POSTGRES_PASSWORD: ${FINDINGS_DB_PASSWORD}
    volumes:
      - findings-db-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U findings"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  findings-db-data:
```

---

## Critérios de aceite (QA)

1. **AC-184-1:** `POST /findings` com dados válidos → 201 + finding com `id` UUID
2. **AC-184-2:** `POST /findings` com mesmo engagement+title+asset+endpoint → 409 + `existing_finding_id`
3. **AC-184-3:** `POST /findings/{id}/triage` → transição de status correta; 422 para transição inválida
4. **AC-184-4:** `GET /findings?engagement_id=X&severity=high` → retorna apenas findings high daquele engagement
5. **AC-184-5:** `GET /findings` sem `engagement_id` → 422 (obrigatório)
6. **AC-184-6:** Chamada sem `X-Session-API-Key` → 401
7. **AC-184-7:** Chamada com key sem capability `pentest.findings.view` → 403
8. **AC-184-8:** `POST /sync-defectdojo` → job enfileirado, `defectdojo_id` preenchido após conclusão
9. **AC-184-9:** Migrações Alembic aplicam schema limpo sem erros

---

## Segurança (AppSec)

- `DEFECTDOJO_API_TOKEN` e `FINDINGS_DB_PASSWORD` **nunca** em código ou logs
- Evidence JSONB: não logar request/response bodies em nível INFO
- Validação de `engagement_id` para evitar IDOR (verificar que engagement pertence ao usuário)
- Rate limiting: 100 req/min por `X-Session-API-Key` (via middleware ou API gateway futuro)

---

## Dependências

- **Depende de:** PROJETOSIN-182 (middleware auth compartilhado em `services/shared/`)
- **Não depende de:** PROJETOSIN-185 (referência `engagement_id` é UUID lógico; validação cruzada opcional)
- **Paralelo seguro com:** PROJETOSIN-185 (workdirs distintos)

**Estimativa:** 4–5 dias (esqueleto FastAPI + models + migrations + sync DD)
