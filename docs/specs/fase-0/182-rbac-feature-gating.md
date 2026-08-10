# Spec Técnica — PROJETOSIN-182: RBAC + Feature Gating

**ADR:** docs/adrs/0001-plataforma-pentest-ia-extensao-openhands.md (accepted)
**Card Plane:** PROJETOSIN-182 — `7ffb28b2-aa8c-444d-aa8f-5879bd9e7ebb`
**Agente responsável:** frontend (UI + gate client-side) + backend (capabilities middleware no Findings/Engagement services)
**Prioridade:** P1 — fundação que desbloqueia 183, 184, 185

---

## Contexto

A autenticação já está implementada no Agent Canvas. Este card adiciona:
1. **Capabilities por perfil** — lista tipada de permissões que um usuário carrega na sessão
2. **Feature gating client-side** — hook `useHasPentestCapability` esconde UI de pentest
3. **Middleware server-side nos serviços Fase 0** — recusa chamadas de quem não tem a capability (defesa em profundidade)

---

## Perfis canônicos e capabilities

```ts
// src/types/pentest-rbac.ts
export type PentestCapability =
  | "pentest.workspace.create"    // criar workspace de pentest
  | "pentest.engagement.create"   // criar engagement
  | "pentest.engagement.view"     // ver engagements
  | "pentest.recon.run"           // executar recon
  | "pentest.scan.passive"        // scan passivo
  | "pentest.scan.active"         // DAST ativo (semi-autônomo)
  | "pentest.exploit.active"      // exploração ativa (gate humano)
  | "pentest.findings.view"       // ver findings
  | "pentest.findings.triage"     // triar FP
  | "pentest.findings.export_dd"  // push para DefectDojo
  | "pentest.mobile.dynamic"      // análise dinâmica mobile
  | "pentest.autonomy.autonomous" // modo autônomo
  | "pentest.admin.users"         // gerenciar usuários e perfis
  | "pentest.admin.scope";        // editar allowlist de escopo

export type PentestProfile = "admin" | "pentester" | "analyst" | "client";

export const PROFILE_CAPABILITIES: Record<PentestProfile, PentestCapability[]> = {
  admin: [/* todas */],
  pentester: [
    "pentest.workspace.create", "pentest.engagement.create", "pentest.engagement.view",
    "pentest.recon.run", "pentest.scan.passive", "pentest.scan.active", "pentest.exploit.active",
    "pentest.findings.view", "pentest.findings.triage", "pentest.findings.export_dd",
    "pentest.mobile.dynamic", "pentest.autonomy.autonomous",
  ],
  analyst: [
    "pentest.engagement.view", "pentest.findings.view", "pentest.findings.triage",
  ],
  client: ["pentest.engagement.view", "pentest.findings.view"],
};
```

---

## Contratos de API (Agent Server / settings extension)

### GET /api/pentest/me/capabilities
Retorna capabilities do usuário autenticado. Implementado como endpoint leve no **Findings Service** (PROJETOSIN-184) e/ou annotation nas settings existentes.

**Response 200:**
```json
{
  "profile": "pentester",
  "capabilities": ["pentest.workspace.create", "pentest.engagement.view", ...]
}
```

**Response 401:** usuário não autenticado
**Response 403:** autenticado mas sem capabilities de pentest (perfil padrão de código)

---

## Frontend — hook e componentes

### Hook principal
```ts
// src/hooks/use-pentest-capabilities.ts
export function usePentestCapabilities(): PentestCapability[]
export function useHasPentestCapability(cap: PentestCapability): boolean
```

- Carrega via TanStack Query (`PENTEST_CAPABILITIES_QUERY_KEY`) de `GET /api/pentest/me/capabilities`
- Cache 5 minutos; refetch on window focus
- Retorna array vazio quando unauthenticated (sem erro visual)

### Wrapper de gate
```tsx
// src/components/features/pentest/capability-gate.tsx
interface CapabilityGateProps {
  capability: PentestCapability;
  fallback?: ReactNode;
  children: ReactNode;
}
export function CapabilityGate({ capability, fallback, children }: CapabilityGateProps)
```

Uso: envolver botões/abas/rotas de pentest — rendem `null` (ou `fallback`) se capability ausente.

---

## Middleware server-side (Python — nos serviços Fase 0)

```python
# services/shared/auth_middleware.py
from fastapi import Depends, HTTPException, Header
from typing import Optional
from .capabilities import PROFILE_CAPABILITIES, PentestCapability

def require_capability(cap: PentestCapability):
    async def _check(x_session_api_key: Optional[str] = Header(None)):
        user_caps = await get_user_capabilities(x_session_api_key)
        if cap not in user_caps:
            raise HTTPException(status_code=403, detail=f"Missing capability: {cap}")
    return Depends(_check)
```

Header authn: `X-Session-API-Key` (mesmo mecanismo do Agent Server).

---

## Arquivos a criar/modificar

| Arquivo | Ação |
|---|---|
| `src/types/pentest-rbac.ts` | Criar — tipos e mapa de capabilities |
| `src/hooks/use-pentest-capabilities.ts` | Criar — hook TanStack Query |
| `src/hooks/query/query-keys.ts` | Adicionar `PENTEST_CAPABILITIES_QUERY_KEYS` |
| `src/components/features/pentest/capability-gate.tsx` | Criar — wrapper de gate |
| `src/i18n/translation.json` | Sem adições de UI neste card (capabilities são internal) |
| `services/shared/auth_middleware.py` | Criar — shared across Findings + EngMgr |
| `services/shared/capabilities.py` | Criar — PROFILE_CAPABILITIES em Python |
| `__tests__/hooks/use-pentest-capabilities.test.ts` | Criar — testes TDD |
| `__tests__/components/pentest/capability-gate.test.tsx` | Criar — testes gate |

---

## Critérios de aceite (QA)

1. **AC-182-1:** `useHasPentestCapability("pentest.workspace.create")` retorna `false` para usuário sem capability → `CapabilityGate` não renderiza children
2. **AC-182-2:** `useHasPentestCapability("pentest.workspace.create")` retorna `true` para pentester → children renderizados
3. **AC-182-3:** `GET /api/pentest/me/capabilities` retorna 403 para usuário sem capabilities de pentest
4. **AC-182-4:** Middleware Python retorna HTTP 403 quando capability requerida está ausente
5. **AC-182-5:** Cache de capabilities invalida após logout

---

## Segurança (AppSec)

- Capabilities **nunca** vêm só do client-side; server-side sempre valida
- Header `X-Session-API-Key` nunca exposto em logs
- Scope: não adicionar nova superfície de auth — reutilizar mecanismo existente do Agent Server

---

## Dependências de outros cards

- **Bloqueia:** 183 (workspace type selector precisa do gate), 184/185 (middleware shared)
- **Não bloqueia:** 186 (Dockerfiles independentes)

**Estimativa:** 2–3 dias (frontend hook + componente + middleware Python)
