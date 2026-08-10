---
card: PROJETOSIN-188
pr: 7
veredicto: PASS
agente: appsec
data: 2026-08-10
tip: 94333d134
ci: npm-audit-high+secret-grep+diff-review
repo: klebersjunior/OpenHands
branch: feat/fase1-findings-ui-188
design: PASS
qa: PASS
---

# AppSecurity — PROJETOSIN-188 Findings panel UI (`/findings`)

**Veredicto:** PASS

Gate de segurança após Design PASS + QA PASS. Revisor (AppSec) ≠ autor do código de produção (Frontend). Este laudo **não** cobre Design nem QA.

Spec § Segurança: `docs/specs/fase-1/188-findings-panel-ui.md` · tip `94333d134` (QA @ `226734331` + pin `typescript-client@1.36.1`)  
PR: https://github.com/klebersjunior/OpenHands/pull/7

## Resumo

Superfície UI de findings consome API pentest com session key da backend registry, gates de capability no nav/rota, evidence colapsada por default e renderização React text-safe (sem HTML cru). Sem critical/high, sem segredos versionados. Residuais medium documentados (defense-in-depth IDOR no `get`/`triage`, redaction de headers em evidence, AuthZ definitiva no backend 184).

## Checklist

| Item | Status |
|------|--------|
| Sem segredos versionados / hardcoded | **PASS** — só `test-key` em fixture Vitest |
| `npm audit` sem high/critical | **PASS** — 4× moderate (dompurify/monaco/posthog, electron); pré-existentes / fora do escopo do diff |
| Session key não vazada em modo público / bundle | **PASS** — header `X-Session-API-Key` de `getEffectiveLocalBackend().apiKey`; sem `localStorage` novo; sem bake de key no client findings |
| Proxies autenticados; VNC | **N/A** — escopo não toca desktop-proxy/VNC |
| Cloud só via `callCloudProxy` | **N/A** — client ad-hoc allowlisted (`findings-service.ts`) para serviço local pentest, padrão `pentest-service.api.ts` |
| Logs sem secrets / conteúdo sensível | **PASS** — sem log de evidence/session key no client |
| XSS title/description/evidence | **PASS** — `{finding.title}` / description / `JSON.stringify` em `<pre>` (text nodes); zero `dangerouslySetInnerHTML` |
| Evidence sem plaintext sem colapso | **PASS** — `evidenceOpen` inicia `false`; expand explícito |
| IDOR / `engagement_id` | **PASS*** — list/stats sempre com `engagement_id`; queries desligadas sem engagement/`canView`; 403 → `FindingsForbidden`. *Ver residual M-188-1 |
| AuthZ UI gates | **PASS** — sidebar `CapabilityGate(pentest.findings.view)`; page `canView`/`canTriage`; triage actions ocultas sem cap |

## Findings

| ID | Sev | Título | Detalhe | Ação |
|----|-----|--------|---------|------|
| — | — | (nenhum critical/high) | — | — |
| **M-188-1** | Medium | `getFinding` / `triage` sem `engagement_id` | Spec pede UI sempre enviar `engagement_id` do contexto; `GET/POST …/findings/{id}` só usa id. Lista/stats OK. AuthZ real é backend (184). UI não revalida `finding.engagement_id === engagementId` no drawer. | Follow-up: passar/validar engagement no client + backend; não bloqueia merge UI |
| **M-188-2** | Medium | Evidence expandida mostra request/response crus | Colapso cumpre spec; sem redaction de `Authorization`/cookies/tokens no JSON. Risco de shoulder-surfing / screenshot. | Follow-up: máscara de headers sensíveis ao renderizar |
| **M-188-3** | Medium | Capability/triage só no client | Esperado em SPA; `useTriageFinding` não re-checa cap na mutation. Backend deve rejeitar sem `pentest.findings.triage`. | Confiar em 182/184; já coberto por AC UI |
| **L-188-1** | Low | `VITE_FINDINGS_SERVICE_URL` + session key | Override de host envia a mesma session key. Misconfig operacional → key para host errado. | Ops: só hosts confiáveis |
| **L-188-2** | Low | Deep-link conversa via `conversation_id`/`event_id` | `navigate(/conversations/${id}?event=…)` confia no payload evidence. IDs tipicamente UUID; path oddity residual. | Opcional: validar formato UUID |

## Dependências

```
npm audit --audit-level=high  → exit 0
# moderate only: dompurify (monaco-editor, posthog-js), electron GHSA-r4w5-6pfg-jxp5
```

Nenhuma high/critical introduzida por este PR. Sem novas deps de produção no diff findings.

## Segredos / superfície revisada

- Client: `src/api/pentest/findings-service.ts` — axios allowlisted; `encodeURIComponent(findingId)`; header session key.
- UI: `findings-page`, drawer, FP modal, row actions, sidebar gate.
- Hooks: `use-findings` — list/stats gated por `engagement_id`; `isFindingsForbiddenError(403)`.
- Testes: fixtures sem tokens reais.

## Ação requerida

Nenhuma bloqueante. **Tech Lead:** mergeável após Design+QA+AppSec PASS (este laudo). Residuais M-188-* como follow-up / card backend se desejado.

Revisor: AppSec. Não cobre Design/QA.
