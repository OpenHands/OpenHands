---
card: PROJETOSIN-188
pr: 7
veredicto: PASS
agente: qa
data: 2026-08-10
tip: 516e0a95d
ci: vitest-findings+eslint-escopo+build+i18n-completeness
repo: klebersjunior/OpenHands
branch: feat/fase1-findings-ui-188
design: PASS @ 516e0a95d
---

# QA — PROJETOSIN-188 Findings panel UI (`/findings`)

**Veredicto:** PASS

Gate AC/regressão após Design PASS (re-gate UI). Revisor (QA) ≠ autor do código de produção (Frontend). Este laudo **não** cobre Design nem AppSec.

Spec: `docs/specs/fase-1/188-findings-panel-ui.md` · design notes · ADR-0001  
PR: https://github.com/klebersjunior/OpenHands/pull/7 · tip `516e0a95d`

## Critérios de aceite

| AC | Status | Evidência |
|----|--------|-----------|
| **AC-188-1** Nav + rota `/findings` com `pentest.findings.view` | **PASS** | `src/routes.ts` registra `route("findings", …)`; sidebar `CapabilityGate` + `sidebar-findings-link` (`sidebar-rail-body.tsx`). Vitest: table/`findings-page` com capability. |
| **AC-188-2** Sem capability → sem dados | **PASS** | Vitest `shows forbidden when view capability is missing` — `findings-forbidden`, sem `findings-table`. |
| **AC-188-3** Filtros severidade/status/ferramenta | **PASS** | `FindingsFilters` + URL params em `findings.tsx`; API `severity`/`status`/`source_tool` em `useFindingsList`. Vitest cobre severidade; status/ferramenta verificados por inspeção + wiring. |
| **AC-188-4** FP reason + triage + UI | **PASS** | Modal: submit disabled sem reason (Vitest). `runTriage` → `useTriageFinding` → invalidação de queries + toast. Service: `advances new → triaging before terminal triage`. |
| **AC-188-5** Sem `pentest.findings.triage` → ações ocultas | **PASS** | Vitest `hides triage actions without triage capability`. |
| **AC-188-6** Sem `engagement_id` → empty | **PASS** | Vitest `findings-empty-no-engagement`. |
| **AC-188-7** i18n completo | **PASS** | 72 keys `FINDINGS$*` × 15 locales; `npm run check-translation-completeness` → *All translation keys have complete language coverage!* |
| **AC-188-8** Sem `react-router` em `src/components/` | **PASS** | Grep zero matches; rota owns `useSearchParams` em `src/routes/findings.tsx`. |
| **AC-188-9** lint + test + build no escopo | **PASS*** | ESLint escopo findings limpo; Vitest 10/10; `npm run build` OK. Ver nota dep abaixo. |

\* Escopo solicitado: findings Vitest + lint do diff. Suite Vitest completa / `npm ci` limpo não exigidos quando `file:` bloqueia CI do fork.

## Regressão

| Checagem | Resultado |
|----------|-----------|
| `npx vitest run __tests__/components/findings/ __tests__/api/pentest/` | **PASS** — 2 files, 10 tests |
| ESLint arquivos do diff findings | **PASS** — no issues |
| `npm run build` | **PASS** (~30s client + SSR flatten) |
| `check-translation-completeness` | **PASS** |
| E2E mock-LLM | **N/A** — `test-mapping.json` sem findings/pentest; spec: Vitest suficiente |
| Design gate | **PASS** (pré-condição) — não reassinado |

## Asserções falsificáveis

| Asserção | Como falharia se controle ausente | Resultado |
|----------|-----------------------------------|-----------|
| Forbidden sem view | Remover gate → teste espera `findings-forbidden` / sem table | PASS |
| Triage oculto sem triage cap | Sempre renderizar actions → `queryByTestId("findings-row-actions")` falha | PASS |
| FP exige reason | Submit sempre enabled → `toBeDisabled` / `onSubmit` not called falha | PASS |
| Empty sem engagement | Lista sem gate → `findings-empty-no-engagement` ausente | PASS |

## Dependência / CI fork

- `package.json` ainda declara `"@openhands/typescript-client": "file:../typescript-client"`.
- Sibling `../typescript-client` **ausente** no worktree; `npm ls` reporta `invalid`; `node_modules` resolve **1.37.1** (cache/install prévio).
- **Pin `1.36.1` não está neste branch** — não alterado por QA (fora de escopo de produção).
- `npm ci` full no fork pode falhar por `file:`; evidência local usa deps já instaladas + build/vitest verdes.
- `tsc --noEmit` acusou `./+types/*` ausentes em rotas fora do escopo (pré-existente / codegen RR); build Vite OK.

## Residual (não bloqueante)

- Vitest de filtros cobre severidade explicitamente; status/tool via código + URL (gap de cobertura, não de comportamento).
- Page tests mockam hooks `use-findings` (padrão do arquivo); mutation real coberta no service test.
- Polish Design D-188-3/4/5 permanece opcional.

## Ação requerida

Nenhuma para Frontend. **Liberar AppSec.** Remover `Blocked` se aplicado. Tech Lead: merge só após AppSec PASS.
