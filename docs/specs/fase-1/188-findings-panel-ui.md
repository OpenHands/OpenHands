# Spec Técnica — PROJETOSIN-188: Painel de Findings UI

**ADR:** docs/adrs/0001-plataforma-pentest-ia-extensao-openhands.md (accepted)
**Card Plane:** PROJETOSIN-188 — `43f9f5b0-4eb9-41f6-9ba4-77ea9164caed`
**Agentes:** design (antes) → frontend
**Prioridade:** P1 high — triagem humana + visualização
**Base git:** `origin/main` @ `f2a8da86a`
**Branch:** `feat/fase1-findings-ui-188`
**Worktree:** `.tmp/worktrees/188`
**PR target:** fork `klebersjunior/OpenHands` only

---

## Objetivo

Nova superfície na UI Agent Canvas para listar, filtrar e triar findings por engagement, consumindo o Findings Service (PROJETOSIN-184). Diff entre scans e link para evidência no event stream.

---

## Ordem de trabalho

1. **Design** — wireframe textual / estados / a11y em `docs/specs/fase-1/188-design-notes.md` (mesmo worktree); gate UI só depois do Frontend.
2. **Frontend** — implementar conforme design + esta spec.
3. **Gates:** Design (UI) → QA → AppSec (revisor ≠ autor). Tech Lead **não** auto-assina.

---

## Rota e navegação

| Item | Decisão |
|---|---|
| Rota canônica | `/findings` (top-level, abaixo de Skills/MCP no sidebar quando capability `pentest.findings.view`) |
| Contexto engagement | Query `?engagement_id=` **obrigatória** para dados; sem ela → empty state “Selecione um engagement” |
| Aba conversa | Opcional MVP+: link “Ver findings” a partir de workspace pentest → `/findings?engagement_id=…` |
| Capability gate | Sidebar + rota envolvidas em `CapabilityGate` / `useHasPentestCapability("pentest.findings.view")` |
| Triagem | Botões de ação gateados por `pentest.findings.triage` (alias card `findings.mark_fp`) |

Sem `react-router` imports em `src/components/` — usar `NavigationProvider` / `NavigationLink`.

---

## API client (frontend)

**Proibido:** `fetch`/`axios` cru para Agent Server. Findings Service é serviço local pentest — criar client dedicado:

```
src/api/pentest/
  findings-service.ts      # list/get/triage/stats
  findings-types.ts
```

Base URL: derivado de `runtime_services` **ou** env `VITE_FINDINGS_SERVICE_URL` com fallback relativo `/api/pentest/findings` via ingress proxy (preferir proxy no `scripts/ingress.mjs` / `static-server.mjs` se ainda não existir — **coordenar com backend se precisar de rota de proxy**; MVP pode chamar host configurado com session key).

Auth: `X-Session-API-Key` da backend registry ativa (mesmo padrão session key).

### Endpoints usados

| UI | API |
|---|---|
| Lista | `GET /api/pentest/findings?engagement_id&status&severity&source_tool&page&page_size` |
| Stats chips | `GET /api/pentest/findings/stats?engagement_id` |
| Detalhe drawer | `GET /api/pentest/findings/{id}` |
| Triagem | `POST /api/pentest/findings/{id}/triage` |

Query keys: constantes em `src/hooks/query/query-keys.ts` (`FINDINGS_QUERY_KEYS`) — sem magic strings inline.

---

## Componentes (estrutura)

```
src/routes/findings.tsx
src/components/features/findings/
  findings-page.tsx
  findings-filters.tsx
  findings-table.tsx
  findings-row-actions.tsx
  finding-detail-drawer.tsx
  finding-severity-badge.tsx
  findings-empty-state.tsx
  findings-diff-banner.tsx      # novo/resolvido/reaberto
```

### Tabela — colunas

Severidade | Título | Ativo | Endpoint | Ferramenta (`source_tool`) | Status | Atualizado

Ordenação default: severidade desc + `updated_at` desc.

### Filtros

- Severidade (multi)
- Status: `new` | `triaging` | `confirmed` | `false_positive` | `duplicate` | `risk_accepted`
- Ferramenta
- Ativo (texto)
- Busca título (client-side na página atual **ou** query `q` se backend já suportar; senão client-side MVP)

### Ações de triagem (por linha / drawer)

| Ação | `new_status` | Extra |
|---|---|---|
| Confirmar | `confirmed` | — |
| Marcar FP | `false_positive` | modal obrigatório `fp_reason` |
| Duplicata | `duplicate` | opcional reason |
| Aceitar risco | `risk_accepted` | reason recomendada |

Payload:

```json
{
  "new_status": "false_positive",
  "fp_reason": "...",
  "triaged_by": "<user id ou email da sessão>"
}
```

Invalidar queries após sucesso; toast i18n.

### Diff entre scans

MVP: comparar dois “snapshots” via query params `since` / `until` **ou** campo derivado se API ainda não tiver endpoint dedicado:

- Se `GET /stats` ou lista não expuser diff → UI mostra banner “Diff indisponível” + issue note; **não** inventar endpoint sem alinhamento TL.
- Preferência: backend 184 já lista por `created_at`; Frontend calcula localmente na página carregada: `new` = status new; “reaberto” fora do MVP se sem histórico.

Meta Fase 1 mínima: **banner** com contagem por status (stats) + filtro rápido “Só novos”. Diff completo scan-a-scan pode ser follow-up se API não cobrir.

### Link evidência → event stream

Se `evidence` tiver `event_id` / `conversation_id`, botão navega para conversa + highlight. Senão, exibir evidence JSON colapsável no drawer (sem inventar deep-link).

---

## i18n

Todas as strings via `t(I18nKey.…)` + `src/i18n/translation.json`. Prefixo sugerido: `FINDINGS$…` (ex. `FINDINGS$TITLE`, `FINDINGS$FILTER_SEVERITY`, `FINDINGS$ACTION_MARK_FP`, `FINDINGS$FP_REASON_REQUIRED`, `FINDINGS$EMPTY`).

Rodar `npm run make-i18n` após keys.

---

## Design system / a11y (Design define; Frontend cumpre)

- HeroUI Table / Dropdown / Modal / Badge / Drawer existentes
- Tokens `--oh-*`; sem cards decorativos desnecessários
- Teclado: ações na row focáveis; modal FP com focus trap
- `aria-label` em ícones de ação
- Contraste severidade (critical/high) WCAG AA
- Empty / loading / error states obrigatórios (`data-testid` estáveis)

---

## Testes

- Vitest: filtros, gate de capability (sem triage buttons), modal FP exige reason
- MSW handlers para Findings API em `__tests__/` / mock handlers
- E2E mock-LLM: **somente** se mapping exigir; caso contrário Vitest suficiente + nota no laudo QA

---

## Critérios de aceite (QA)

1. **AC-188-1:** Usuário com `pentest.findings.view` vê item Findings no nav e rota `/findings`
2. **AC-188-2:** Sem capability → item ausente; deep-link mostra empty/forbidden (não vaza dados)
3. **AC-188-3:** Lista respeita filtros severidade/status/ferramenta
4. **AC-188-4:** Triagem FP exige reason; chama triage e atualiza status na UI
5. **AC-188-5:** Sem `pentest.findings.triage` → ações de triagem ocultas/disabled
6. **AC-188-6:** Sem `engagement_id` → empty state orientando seleção
7. **AC-188-7:** i18n keys completas (`check-translation-completeness`)
8. **AC-188-8:** Sem imports `react-router` em `src/components/`
9. **AC-188-9:** `npm run lint` + `npm test` + `npm run build` verdes no escopo

---

## Segurança (AppSec)

- Não renderizar secrets de evidence em plaintext sem colapso
- Session key só via client options / headers — nunca `localStorage` novo
- IDOR: UI sempre envia `engagement_id` do contexto; erros 403 tratados

---

## Dependências

- **Depende de:** PROJETOSIN-184 (API), PROJETOSIN-182 (gates UI)
- **Design antes de Frontend** no mesmo worktree (um escritor por vez)
- **Paralelo com:** 187 / 189 em outras worktrees

**Estimativa:** Design 0.5–1d + Frontend 3–4d
