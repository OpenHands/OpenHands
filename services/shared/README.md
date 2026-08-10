# services/shared

Middleware Python compartilhado entre `findings-service` e `engagement-manager`.

## Arquivos a implementar (PROJETOSIN-182 — backend)

- `auth_middleware.py` — FastAPI Depends que valida `X-Session-API-Key` e capabilities
- `capabilities.py` — `PROFILE_CAPABILITIES` dict (Python espelho de `src/types/pentest-rbac.ts`)

## Uso

```python
from services.shared.auth_middleware import require_capability

@router.get("/findings")
async def list_findings(
    _: None = require_capability("pentest.findings.view")
):
    ...
```

Ver spec completa: `docs/specs/fase-0/182-rbac-feature-gating.md`
