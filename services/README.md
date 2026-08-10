# services/

Serviços backend da Plataforma de Pentest com IA (ADR-0001).

Todos vivem neste monorepo conforme decisão pós-intake do PO (2026-08-10).

## Estrutura

```
services/
├── shared/                  # Middleware Python compartilhado (auth, capabilities)
├── findings-service/        # PROJETOSIN-184 — FastAPI + Postgres, fonte de verdade de findings
└── engagement-manager/      # PROJETOSIN-185 — FastAPI + Postgres, provisionamento de sandboxes
```

## Specs técnicas

- `docs/specs/fase-0/182-rbac-feature-gating.md`
- `docs/specs/fase-0/183-workspace-type-selector.md`
- `docs/specs/fase-0/184-findings-service.md`
- `docs/specs/fase-0/185-engagement-manager.md`
- `docs/specs/fase-0/186-dockerfiles-runtimes.md`

## Portas (dev local)

| Serviço | Porta |
|---------|-------|
| findings-service | 18002 |
| engagement-manager | 18003 |

## Deploy

Cada serviço tem um `docker-compose.fragment.yml` que é integrado ao compose principal.
O Engagement Manager monta `/var/run/docker.sock` para provisionar sandboxes por engagement.
