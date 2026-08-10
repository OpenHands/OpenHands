# Spec Técnica — PROJETOSIN-185: Engagement Manager (URGENT)

**ADR:** docs/adrs/0001-plataforma-pentest-ia-extensao-openhands.md (accepted)
**Card Plane:** PROJETOSIN-185 — `29b3d561-7f2f-463e-afc6-622c892bdc78`
**Agente responsável:** backend (serviço) + devops (compose fragments)
**Prioridade:** P0 URGENT — provisionamento de ambiente isolado por cliente

---

## Boundary e localização

Serviço novo, **não** embutido no `openhands-automation`. Vive em `services/engagement-manager/`.

```
services/engagement-manager/
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
│   │   ├── engagement.py
│   │   └── scope.py
│   ├── schemas/
│   │   ├── engagement.py
│   │   └── scope.py
│   ├── routers/
│   │   ├── engagements.py
│   │   ├── scope.py
│   │   └── runtime.py      # provision / teardown sandbox
│   ├── services/
│   │   ├── engagement_service.py
│   │   ├── runtime_provisioner.py   # docker-compose up/down por engagement
│   │   └── scope_validator.py
│   └── middleware/
│       └── auth.py          # compartilha services/shared/auth_middleware.py
└── tests/
    ├── test_engagements_crud.py
    ├── test_scope.py
    └── test_runtime_provisioner.py
```

---

## Modelo de dados (PostgreSQL)

```sql
CREATE TABLE engagements (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name                TEXT NOT NULL,
    client_name         TEXT NOT NULL,
    description         TEXT,
    status              VARCHAR(32) NOT NULL DEFAULT 'draft',
                        -- 'draft' | 'active' | 'paused' | 'completed' | 'archived'
    scope_authorized_at TIMESTAMPTZ,             -- NULL = sem autorização registrada
    scope_document_url  TEXT,                    -- link/ref ao Rules of Engagement
    autonomy_mode       VARCHAR(32) NOT NULL DEFAULT 'semi_autonomous',
    runtime_profile     VARCHAR(32) NOT NULL DEFAULT 'web',
                        -- 'web' | 'network' | 'mobile' | 'sast'
    sandbox_status      VARCHAR(32) DEFAULT 'stopped',
                        -- 'stopped' | 'provisioning' | 'running' | 'error'
    sandbox_compose_project TEXT,                -- nome do compose project (ex: "eng-abc123")
    defectdojo_engagement_id INTEGER,           -- ID no DD após criação
    created_by          TEXT NOT NULL,           -- user identifier
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT engagements_status_check CHECK (
        status IN ('draft','active','paused','completed','archived')
    ),
    CONSTRAINT engagements_autonomy_check CHECK (
        autonomy_mode IN ('manual','semi_autonomous','autonomous')
    ),
    CONSTRAINT engagements_runtime_check CHECK (
        runtime_profile IN ('web','network','mobile','sast')
    )
);

CREATE TABLE scope_rules (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    engagement_id   UUID NOT NULL REFERENCES engagements(id) ON DELETE CASCADE,
    rule_type       VARCHAR(16) NOT NULL,  -- 'allow' | 'deny'
    target_type     VARCHAR(16) NOT NULL,  -- 'ip' | 'cidr' | 'domain' | 'url'
    target_value    TEXT NOT NULL,
    note            TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_engagements_status ON engagements(status);
CREATE INDEX idx_scope_rules_engagement_id ON scope_rules(engagement_id);
```

---

## API — Endpoints

### Base path: `/api/pentest/engagements`

| Method | Path | Capability | Descrição |
|--------|------|------------|-----------|
| GET | `/` | `pentest.engagement.view` | Listar engagements do usuário |
| POST | `/` | `pentest.engagement.create` | Criar engagement |
| GET | `/{id}` | `pentest.engagement.view` | Detalhe |
| PATCH | `/{id}` | `pentest.engagement.create` | Atualizar (status, meta) |
| DELETE | `/{id}` | `pentest.admin.users` | Remover (admin only) |
| GET | `/{id}/scope` | `pentest.engagement.view` | Listar scope rules |
| POST | `/{id}/scope` | `pentest.admin.scope` | Adicionar scope rule |
| DELETE | `/{id}/scope/{rule_id}` | `pentest.admin.scope` | Remover scope rule |
| POST | `/{id}/authorize-scope` | `pentest.admin.scope` | Registrar autorização (RoE) |
| POST | `/{id}/provision` | `pentest.engagement.create` | Provisionar sandbox |
| POST | `/{id}/teardown` | `pentest.engagement.create` | Derrubar sandbox |
| GET | `/{id}/sandbox-status` | `pentest.engagement.view` | Status do sandbox |

---

## Contratos de payload chave

### POST /api/pentest/engagements

```json
{
  "name": "WebApp Audit — Cliente ACME Q3-2026",
  "client_name": "ACME Corp",
  "description": "Auditoria de segurança da aplicação web principal",
  "runtime_profile": "web",
  "autonomy_mode": "semi_autonomous"
}
```

**Response 201:** `{ ...engagement, id: "uuid", status: "draft" }`

### POST /api/pentest/engagements/{id}/authorize-scope

```json
{
  "scope_document_url": "https://drive.heimdall.local/roe/acme-q3-2026.pdf",
  "scope_rules": [
    { "rule_type": "allow", "target_type": "domain", "target_value": "*.acme.com" },
    { "rule_type": "allow", "target_type": "cidr", "target_value": "10.100.0.0/24" },
    { "rule_type": "deny", "target_type": "domain", "target_value": "prod-payments.acme.com" }
  ]
}
```

Seta `scope_authorized_at = NOW()`. Workspace pentest só pode ser criado após esta chamada.

### POST /api/pentest/engagements/{id}/provision

Inicia provisionamento assíncrono do sandbox Docker.

```json
{}
```

**Response 202:**
```json
{
  "job_id": "uuid",
  "status": "provisioning",
  "sandbox_compose_project": "eng-<short_id>"
}
```

O `RuntimeProvisioner` gera um `docker-compose.yml` a partir do template do runtime_profile e faz `docker compose -p eng-<id> up -d`.

---

## Runtime Provisioner

```python
# services/engagement-manager/app/services/runtime_provisioner.py

RUNTIME_TEMPLATES = {
    "web": "templates/compose-web-runtime.yml.j2",
    "network": "templates/compose-network-runtime.yml.j2",
    "mobile": "templates/compose-mobile-runtime.yml.j2",
    "sast": "templates/compose-sast-runtime.yml.j2",
}

class RuntimeProvisioner:
    async def provision(self, engagement: Engagement) -> None:
        """
        1. Renderiza template Jinja2 do compose com:
           - network name: eng-{engagement.id[:8]}
           - volume prefix: eng-{engagement.id[:8]}
           - scope_rules serialized para egress proxy config
        2. Escreve compose file em /tmp/eng-{id}/docker-compose.yml
        3. Executa: docker compose -p eng-{id} up -d
        4. Aguarda health checks (asyncio timeout 120s)
        5. Atualiza sandbox_status → 'running'
        """

    async def teardown(self, engagement: Engagement) -> None:
        """docker compose -p eng-{id} down -v"""
```

**Templates de compose para Fase 0** (stubs — runtime images definidas em PROJETOSIN-186):
- `templates/compose-web-runtime.yml.j2` — usa `ghcr.io/heimdall/runtime-web:latest`
- `templates/compose-network-runtime.yml.j2` — usa `ghcr.io/heimdall/runtime-network:latest`
- `templates/compose-mobile-runtime.yml.j2` — usa `ghcr.io/heimdall/runtime-mobile:latest` + emulador
- `templates/compose-sast-runtime.yml.j2` — usa `ghcr.io/heimdall/runtime-sast:latest`

---

## Isolamento de rede por engagement

Cada engagement sobe com network isolada:
```yaml
# fragment gerado pelo provisioner
networks:
  eng-abc123-internal:
    driver: bridge
    internal: true     # sem egress direto
  eng-abc123-egress:
    driver: bridge
```

Egress proxy (ex.: `tinyproxy` ou `dante`) aplica allowlist de scope_rules — somente IPs/CIDRs/domínios autorizados.

---

## Docker Compose raiz (fragment)

```yaml
# services/engagement-manager/docker-compose.fragment.yml
services:
  engagement-manager:
    build:
      context: ./services/engagement-manager
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock  # para provisionar compose por engagement
      - engagement-manager-data:/data
    environment:
      - ENGMGR_DB_URL=postgresql+asyncpg://engmgr:${ENGMGR_DB_PASSWORD}@engmgr-db/engmgr
      - SESSION_API_KEY=${SESSION_API_KEY}
      - DEFECTDOJO_API_URL=${DEFECTDOJO_API_URL}
      - DEFECTDOJO_API_TOKEN=${DEFECTDOJO_API_TOKEN}
    depends_on:
      engmgr-db:
        condition: service_healthy
    ports:
      - "18003:8000"

  engmgr-db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: engmgr
      POSTGRES_USER: engmgr
      POSTGRES_PASSWORD: ${ENGMGR_DB_PASSWORD}
    volumes:
      - engmgr-db-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U engmgr"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  engagement-manager-data:
  engmgr-db-data:
```

---

## Routing no ingress

Adicionar ao `scripts/ingress.mjs`:
```
/api/pentest/engagements/* → engagement-manager :18003
/api/pentest/findings/*    → findings-service :18002
/api/pentest/me/*          → findings-service :18002 (capabilities endpoint)
```

---

## Critérios de aceite (QA)

1. **AC-185-1:** `POST /engagements` cria engagement com status `draft`
2. **AC-185-2:** `POST /engagements/{id}/authorize-scope` preenche `scope_authorized_at`
3. **AC-185-3:** Criar workspace pentest com engagement sem `scope_authorized_at` → 400
4. **AC-185-4:** `POST /engagements/{id}/provision` retorna 202 e inicia sandbox
5. **AC-185-5:** `POST /engagements/{id}/teardown` derruba containers do engagement
6. **AC-185-6:** Listagem de engagements retorna apenas engagements do usuário autenticado (IDOR prevention)
7. **AC-185-7:** Scope rules com `rule_type=deny` bloqueiam destino na rede do sandbox
8. **AC-185-8:** Chamada sem autenticação → 401; sem capability → 403

---

## Segurança (AppSec)

- **Docker socket** montado no container — superfície crítica. Sandbox: o EngMgr container deve ser o único com acesso ao socket; não expor diretamente ao frontend
- Scope rules validadas antes de provisionar — allowlist aplicada via egress proxy, não apenas no software
- `ENGMGR_DB_PASSWORD` e `DEFECTDOJO_API_TOKEN` via env/secrets, nunca em código
- Logs de provision não devem conter credenciais do cofre

---

## Dependências

- **Depende de:** PROJETOSIN-182 (middleware auth)
- **Paralelo seguro com:** PROJETOSIN-184 (workdirs distintos, sem conflito)
- **Desbloqueia:** PROJETOSIN-183 (workspace pentest list de engagements), PROJETOSIN-186 (imagens usadas nos templates)

**Estimativa:** 5–6 dias (serviço + modelos + provisioner + testes)
