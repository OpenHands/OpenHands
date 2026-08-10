---
card: PROJETOSIN-182
pr: 3
veredicto: PASS
agente: appsec
data: 2026-08-10
ci: npm-audit-high
repo: klebersjunior/OpenHands
branch: feat/fase0-frontend-182-183
escopo: frontend (este PR)
---

# AppSecurity Report — PROJETOSIN-182 RBAC + Feature Gating (FE)

**Veredicto:** PASS (escopo frontend deste PR)

**PR:** https://github.com/klebersjunior/OpenHands/pull/3  
**Worktree:** `OpenHands-wt-frontend` (`feat/fase0-frontend-182-183`)  
**Spec:** `docs/specs/fase-0/182-rbac-feature-gating.md`  
**Nota:** AuthZ server-side (middleware Python) é escopo do PR backend / AppSec BE — **não** reavaliado aqui. Label `Blocked` no card permanece pelo FAIL AppSec BE; este laudo **não** remove Blocked.

## Resumo

O FE implementa feature gating de apresentação (`CapabilityGate` + hooks) alimentado por `GET /api/pentest/me/capabilities` via `PentestService`. Em 401/403/404 o client falha fechado para `{ profile: null, capabilities: [] }`. Não há segredos hardcoded, nem high/critical em `npm audit`. **UI hide ≠ AuthZ** — documentado como residual Medium; enforcement real depende do backend.

## Checklist

- [x] Sem segredos versionados / hardcoded (fixtures de teste usam `session-key` / `test-key`)
- [x] `npm audit --audit-level=high` sem high/critical (só moderate: dompurify via monaco/posthog, electron — pré-existentes / fora do diff)
- [x] Session key não baked em modo público por este diff; client lê `backend.apiKey` e envia `X-Session-API-Key` só quando presente
- [x] Proxies/VNC/desktop: não tocados neste PR
- [x] Cloud: não tocado; pentest client usa backend local efetivo
- [x] Logs: sem dump de secrets / conversation content neste escopo

## Findings

| ID | Severidade | Título | Detalhe | Ação |
|---|---|---|---|---|
| AS-FE-182-1 | **Medium** (residual) | UI hide ≠ AuthZ | `CapabilityGate` / `useHasPentestCapability` só controlam render. Bypass via DevTools/`mutate` não é prevenido no FE — esperado. AuthZ de API/MCP/workspace é obrigação do middleware BE (`services/shared/`). | Não bloqueia este PR FE. Manter Blocked do AppSec BE até PASS server-side. |
| AS-FE-182-2 | Low | `profile` sem whitelist | `normalizeCapabilitiesResponse` faz cast de qualquer string para `PentestProfile`. Hoje o FE **não** autoriza por `profile` (só por lista `capabilities` filtrada com prefixo `pentest.`). | Endurecer whitelist se UI passar a ramificar por perfil. |
| AS-FE-182-3 | Info | Espelho `PROFILE_CAPABILITIES` | Mapa client em `src/types/pentest-rbac.ts` não é usado para grant em runtime (só fixtures de teste). Drift vs Python BE é risco de inconsistência de docs/testes, não de AuthZ FE. | Manter sync com `services/shared/capabilities.py` no BE. |

## Superfície revisada

| Path | Nota de segurança |
|---|---|
| `src/api/pentest-service/pentest-service.api.ts` | axios allowlisted em `no-direct-agent-server-calls.test.ts`; header session key; fail-closed 401/403/404; filter de capabilities |
| `src/hooks/use-pentest-capabilities.ts` | query fail-closed; `meta.disableToast`; invalidate no logout |
| `src/components/features/pentest/capability-gate.tsx` | gate de apresentação apenas |
| `src/types/pentest-rbac.ts` | tipos + mapa de perfil (não usado para grant runtime) |
| `src/routes/users-settings.tsx` | `useInvalidatePentestCapabilities` após `AppLoginService.logout()` (AC-182-5) |

## Dependências

```text
npm audit --audit-level=high → exit 0
# 4 moderate (dompurify×3 chain, electron) — sem high/critical
```

## Ação requerida

Nenhuma para merge do **escopo FE** deste PR do ponto de vista AppSec FE. Card 182 continua bloqueado pelo AppSec **BE** até o middleware AuthZ PASS. Tech Lead: não tratar este PASS FE como AuthZ completa do card.
