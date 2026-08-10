# Design Notes — PROJETOSIN-188: Painel de Findings

**Card:** PROJETOSIN-188 · `43f9f5b0-4eb9-41f6-9ba4-77ea9164caed`  
**Spec técnica:** [188-findings-panel-ui.md](./188-findings-panel-ui.md)  
**ADR:** [0001-plataforma-pentest-ia-extensao-openhands.md](../../adrs/0001-plataforma-pentest-ia-extensao-openhands.md)  
**Agente:** Design · branch `feat/fase1-findings-ui-188`  
**Status:** Pronto para Frontend

---

## Design — Findings panel (`/findings`)

### 1. Objetivo de UX

Superfície top-level para o pentester **listar, filtrar e triar** findings de um engagement. Fluxo principal: escolher/chegar com engagement → escanear lista (stats + filtros) → abrir detalhe → triar (confirmar / FP / duplicata / risco). Sem inventar tema paralelo: tokens `--oh-*`, HeroUI, tipografia e empty states no mesmo idioma visual de Automations / MCP / Skills.

### 2. Navegação e entrada

| Elemento | Decisão |
|---|---|
| Rota | `/findings` |
| Query obrigatória para dados | `?engagement_id=<uuid>` |
| Sidebar | Item **Findings** top-level, **logo após Customize (Skills/MCP)** e antes de Automations; só renderiza se `pentest.findings.view` |
| Deep-link sem capability | Página renderiza estado **forbidden** (sem dados, sem skeleton de tabela) |
| Link MVP+ (opcional) | Workspace pentest / conversa → “Ver findings” → `/findings?engagement_id=…` (não bloqueia MVP se ausente) |
| Router em components | Proibido `react-router` em `src/components/` — `NavigationLink` / `NavigationProvider` |

**Ordem visual do rail (capability-gated):**

```
Conversations
Customize (Skills / MCP / …)
Findings          ← novo, se pentest.findings.view
Automations
…
```

Ícone sugerido: glyph de “lista / alerta” estilo outline (stroke `currentColor`, 24×24), mesmo tamanho `ICON_SIZE` do rail. `data-testid="sidebar-findings-link"`. Label: `FINDINGS$NAV_LABEL`.

### 3. Layout da página

Espelhar o shell de páginas de extensão (`settingsLikeMainScrollClassName`):

```
┌─────────────────────────────────────────────────────────────┐
│ Header                                                      │
│  Título: Findings                                           │
│  Subtítulo: engagement name ou id truncado (se houver)      │
│  [Seletor de engagement — MVP: read-only chip + hint]       │
├─────────────────────────────────────────────────────────────┤
│ Stats banner (chips)  [Só novos]                            │
├─────────────────────────────────────────────────────────────┤
│ Toolbar filtros (wrap) + busca título                       │
├─────────────────────────────────────────────────────────────┤
│ Tabela (scroll vertical no main; sticky thead se fácil)     │
│ … paginação footer                                          │
└─────────────────────────────────────────────────────────────┘
         → Drawer detalhe (direita, overlay)
         → Modal FP (centrado, focus trap)
```

- **Sem cards decorativos** na hero/header. Chips de stats são controles (filtro rápido), não “dashboard tiles”.
- Fundo: `base` / `base-secondary` existentes; bordas `var(--oh-border)`.
- Largura: full content column do app (mesmo gutter de `/mcp` / automations).
- Mobile (&lt; md): filtros empilham; tabela vira **lista compacta** (ver §5.1) — não exigir scroll horizontal obrigatório.

### 4. Fluxo (happy path)

```
[Nav Findings] → /findings
       │
       ├─ sem engagement_id → Empty: selecione engagement
       ├─ sem capability    → Forbidden
       └─ com engagement_id
              │
              ├─ loading → skeleton
              ├─ error/403 → error / forbidden
              ├─ 0 findings → empty (sem findings)
              └─ N findings
                     │
                     ├─ filtra / “Só novos”
                     ├─ click row → abre drawer detalhe
                     └─ ações triage (se capability)
                            ├─ Confirmar / Duplicata / Aceitar risco
                            └─ Marcar FP → Modal (reason obrigatória) → POST triage → toast → invalidate
```

### 5. Componentes e comportamento

#### 5.1 Tabela (`findings-table`)

| Coluna | Conteúdo | Notas |
|---|---|---|
| Severidade | Badge | Ordenação default: severity desc |
| Título | Texto + truncate 1 linha | Click abre drawer |
| Ativo | Texto curto | Truncate + `title` nativo |
| Endpoint | Path/URL | Truncate mono leve (`font-mono text-xs`) |
| Ferramenta | `source_tool` | Chip/pill neutro (`extensionModuleCardPillClassName` ok) |
| Status | Badge status | |
| Atualizado | relative time | `updated_at` |
| Ações | Menu/botões | Só se `pentest.findings.triage` |

- Ordenação default: severidade desc + `updated_at` desc (server ou client estável na página).
- Row clicável (Enter/Space abre drawer); ações na row não disparam o click da row (`stopPropagation`).
- Desktop: HeroUI `Table` (ou tabela HTML estilizada no padrão do app se Table for pesada — preferir HeroUI).
- Mobile: cada finding vira bloco com severidade + título + status; endpoint/ferramenta em segunda linha; ações via menu “⋯”.

#### 5.2 Filtros (`findings-filters`)

| Controle | Tipo | Comportamento |
|---|---|---|
| Severidade | Multi-select | critical / high / medium / low / info |
| Status | Multi-select | new, triaging, confirmed, false_positive, duplicate, risk_accepted |
| Ferramenta | Select / multi | Valores distintos da API ou lista livre |
| Ativo | Input texto | Debounce ~300 ms → query |
| Busca título | Input | Client-side na página atual **ou** `q` se API existir |
| Limpar filtros | Link/botão texto | Como Automations filtered empty |
| Só novos | Chip toggle no banner | Aplica status=`new` |

Filtros ativos devem refletir na URL query (opcional MVP, recomendado): `severity`, `status`, `source_tool`, `asset`, `page` — além de `engagement_id`.

#### 5.3 Stats / diff banner (`findings-diff-banner`)

MVP Fase 1:

- Chips com contagens de `GET …/stats` (por status e/ou severidade — o que a API 184 entregar).
- Chip **“Só novos”** como atalho de filtro.
- Se API **não** expuser diff scan-a-scan: banner secundário discreto “Diff entre scans indisponível” (`FINDINGS$DIFF_UNAVAILABLE`) — **não** inventar endpoint.
- Não usar cards com sombra/glow; chips flat com borda `--oh-border`, ativo com `interactive-active` / primary sutil.

#### 5.4 Drawer de detalhe (`finding-detail-drawer`)

- Lado direito; largura ~420–480 px desktop; full-screen sheet no mobile.
- Fecha: Escape, overlay click, botão fechar.
- Conteúdo (ordem):
  1. Severidade + status badges
  2. Título (heading `h2`)
  3. Meta: ativo, endpoint, ferramenta, timestamps
  4. Descrição / detalhes (markdown plain se vier texto)
  5. Evidence: **colapsável** por padrão; se `event_id` + `conversation_id` → botão “Ver no event stream”; senão JSON/texto colapsado (sem secrets em plaintext expandido por default — ver AppSec)
  6. Footer sticky: ações de triagem (mesmas da row), gateadas por `pentest.findings.triage`

Loading do `GET /findings/{id}`: skeleton no corpo do drawer. Erro: mensagem + Retry.

#### 5.5 Modal FP (`finding-fp-modal`)

- Abre só em “Marcar FP”.
- Campo **motivo** (`fp_reason`) obrigatório — textarea, min 1 char trimmed (recomendado min ~10 chars no copy de validação, enforce ≥1 no AC).
- Botões: Cancelar (secundário) / Confirmar FP (danger ou primary conforme BrandButton existente).
- Submit disabled enquanto vazio ou enquanto mutation pending.
- Focus trap HeroUI Modal; ao abrir, foco no textarea; ao fechar, devolve foco ao trigger.
- Erro de API: inline no modal + toast.

#### 5.6 Outras ações de triagem

| Ação | UI | Reason |
|---|---|---|
| Confirmar | Botão / menu item | — |
| Duplicata | Menu item | Optional: prompt curto ou sem modal no MVP |
| Aceitar risco | Menu item | Reason **recomendada** — modal leve opcional no MVP; se sem modal, confirmar com ConfirmDialog e reason opcional em textarea |

Após sucesso: toast (`FINDINGS$TOAST_TRIAGE_SUCCESS`), fechar modal se aberto, invalidar list/stats/detail queries, atualizar badge na row.

Sem `pentest.findings.triage`: **ocultar** ações (preferível a disabled-only, para não vazar capacidade de ação); drawer só leitura.

### 6. Estados

| Estado | Quando | UI | `data-testid` |
|---|---|---|---|
| **Loading** | Fetch lista/stats com engagement | Skeleton header chips + 5–8 rows skeleton | `findings-loading` |
| **Empty (sem engagement)** | Sem `engagement_id` | Painel bordered (`extensionModuleEmptyStateClassName`): título + hint “Selecione um engagement” — sem tabela | `findings-empty-no-engagement` |
| **Empty (sem findings)** | Lista 0 com engagement | Empty bordered + hint (scans ainda não produziram findings) | `findings-empty` |
| **Empty (filtros)** | Lista existe mas filtros zeram | Dashed empty + “Limpar filtros” | `findings-filtered-empty` |
| **Error** | 5xx / network | Banner erro + Retry | `findings-error` |
| **Forbidden** | Sem capability view **ou** 403 API | Mensagem neutra (sem IDs de findings, sem contagens) | `findings-forbidden` |
| **Success / populated** | Dados OK | Banner + filtros + tabela | `findings-page`, `findings-table` |

**Hierarquia de estados (avaliar nesta ordem):** forbidden → no engagement → loading → error → empty → filtered-empty → table.

Capability sem view: sidebar **não** mostra o link; deep-link ainda pode chegar → forbidden.

### 7. Severidade e status — visual / contraste

Badges usam HeroUI `Chip`/`Badge` com tokens locais — **evitar roxo genérico de “AI”**.

| Severidade | Tratamento visual (AA) |
|---|---|
| critical | Fundo danger escuro / texto claro; ícone opcional; contraste ≥ 4.5:1 |
| high | Danger ou laranja do sistema se existir; senão danger atenuado + borda |
| medium | Warning / primary dourado `--oh-color-primary` com texto escuro se fundo claro o bastante; senão outline + texto foreground |
| low | Neutro (`tertiary` / border) |
| info | Muted / subtle |

Status:

| Status | Tom |
|---|---|
| new | Primary / destaque sutil |
| triaging | Neutro ativo |
| confirmed | Success |
| false_positive | Muted + strikethrough opcional no título? **Não** — só badge |
| duplicate | Muted |
| risk_accepted | Warning outline |

Não confiar só em cor: texto do badge legível; severidade também anunciada em `aria-label` na célula.

### 8. Acessibilidade (WCAG 2.1 AA)

- [ ] Contraste AA em badges, links, botões e texto tertiary sobre `base`
- [ ] Foco visível (`:focus-visible`) em row actions, filtros, chips, paginação
- [ ] Tab order: header → stats → filtros → tabela → paginação; drawer/modal capturam foco
- [ ] Operável por teclado: abrir row (Enter), fechar drawer (Escape), menu ações
- [ ] `aria-label` em ícones de ação e botão fechar drawer
- [ ] Modal FP: label associado ao textarea (`htmlFor` / `aria-labelledby`); erro `aria-describedby` + `aria-invalid`
- [ ] Loading: `aria-busy="true"` na região da tabela; empty/error com `role="status"` ou heading
- [ ] Tabela: `<th scope="col">`; row actions em `aria-label` com título do finding truncado
- [ ] Não depender só de cor para severidade
- [ ] Evidence colapsável: botão com `aria-expanded`
- [ ] Responsivo: filtros e ações usáveis em viewport estreita

### 9. Tokens / HeroUI (sem tema paralelo)

- Cores/superfície: `--oh-border`, `--oh-foreground`, `--oh-text-secondary`, `--oh-text-tertiary`, `--oh-muted`, `--oh-surface-raised`, `--oh-color-danger`, `--oh-color-success`, `--oh-color-primary`
- Empty: `extensionModuleEmptyStateClassName` / filtered dashed como Automations
- Componentes: HeroUI Table (ou equivalente), Dropdown/Autocomplete filtros, Modal, Chip/Badge, Button (`BrandButton` onde for CTA do app), Spinner
- Toasts: `displayErrorToast` / success handlers existentes
- Sem cards com multi-shadow / glow / pills decorativos no header

### 10. `data-testid` sugeridos (estáveis)

| ID | Onde |
|---|---|
| `sidebar-findings-link` | Nav |
| `findings-page` | Root da página |
| `findings-loading` | Skeleton |
| `findings-empty-no-engagement` | Empty sem engagement |
| `findings-empty` | Empty sem findings |
| `findings-filtered-empty` | Empty pós-filtro |
| `findings-error` | Erro |
| `findings-forbidden` | Sem capability / 403 |
| `findings-stats-banner` | Banner stats |
| `findings-filter-new-only` | Chip “Só novos” |
| `findings-diff-unavailable` | Nota diff |
| `findings-filters` | Toolbar |
| `findings-filter-severity` | Controle |
| `findings-filter-status` | Controle |
| `findings-filter-tool` | Controle |
| `findings-filter-asset` | Input ativo |
| `findings-search-title` | Busca |
| `findings-clear-filters` | Limpar |
| `findings-table` | Tabela / lista |
| `findings-row-{id}` | Row (id estável da API) |
| `findings-row-actions` | Menu ações |
| `findings-action-confirm` | |
| `findings-action-mark-fp` | |
| `findings-action-duplicate` | |
| `findings-action-accept-risk` | |
| `finding-detail-drawer` | Drawer |
| `finding-detail-loading` | |
| `finding-evidence-toggle` | |
| `finding-evidence-open-stream` | Link event stream |
| `finding-fp-modal` | Modal |
| `finding-fp-reason` | Textarea |
| `finding-fp-submit` | Confirmar |
| `finding-fp-cancel` | Cancelar |

### 11. i18n — prefixo `FINDINGS$`

Frontend adiciona em `src/i18n/translation.json` (todas as locales) e roda `npm run make-i18n`. Copy EN de referência abaixo; PT e demais seguem o padrão do repo.

#### Nav / página

| Key | EN (ref) |
|---|---|
| `FINDINGS$NAV_LABEL` | Findings |
| `FINDINGS$TITLE` | Findings |
| `FINDINGS$SUBTITLE` | Triage security findings for this engagement |
| `FINDINGS$ENGAGEMENT_LABEL` | Engagement |
| `FINDINGS$ENGAGEMENT_MISSING_HINT` | Open Findings from a pentest workspace, or provide an engagement id |

#### Estados

| Key | EN (ref) |
|---|---|
| `FINDINGS$EMPTY_NO_ENGAGEMENT` | Select an engagement |
| `FINDINGS$EMPTY_NO_ENGAGEMENT_HINT` | Findings are scoped to an engagement. Choose one to view results. |
| `FINDINGS$EMPTY` | No findings yet |
| `FINDINGS$EMPTY_HINT` | Findings will appear here after scans report results for this engagement. |
| `FINDINGS$NO_FILTER_MATCHES` | No findings match the current filters |
| `FINDINGS$CLEAR_FILTERS` | Clear filters |
| `FINDINGS$ERROR` | Couldn’t load findings |
| `FINDINGS$ERROR_RETRY` | Retry |
| `FINDINGS$FORBIDDEN` | You don’t have access to findings |
| `FINDINGS$FORBIDDEN_HINT` | Ask an admin for the findings view permission. |
| `FINDINGS$LOADING` | Loading findings |

#### Stats / diff

| Key | EN (ref) |
|---|---|
| `FINDINGS$STATS_LABEL` | Summary |
| `FINDINGS$FILTER_NEW_ONLY` | New only |
| `FINDINGS$DIFF_UNAVAILABLE` | Scan-to-scan diff is not available yet |
| `FINDINGS$COUNT_NEW` | {{count}} new |
| `FINDINGS$COUNT_CONFIRMED` | {{count}} confirmed |
| `FINDINGS$COUNT_FP` | {{count}} false positives |
| `FINDINGS$COUNT_TOTAL` | {{count}} total |

#### Filtros / tabela

| Key | EN (ref) |
|---|---|
| `FINDINGS$FILTER_SEVERITY` | Severity |
| `FINDINGS$FILTER_STATUS` | Status |
| `FINDINGS$FILTER_TOOL` | Tool |
| `FINDINGS$FILTER_ASSET` | Asset |
| `FINDINGS$SEARCH_TITLE_PLACEHOLDER` | Search by title |
| `FINDINGS$COL_SEVERITY` | Severity |
| `FINDINGS$COL_TITLE` | Title |
| `FINDINGS$COL_ASSET` | Asset |
| `FINDINGS$COL_ENDPOINT` | Endpoint |
| `FINDINGS$COL_TOOL` | Tool |
| `FINDINGS$COL_STATUS` | Status |
| `FINDINGS$COL_UPDATED` | Updated |
| `FINDINGS$COL_ACTIONS` | Actions |
| `FINDINGS$PAGINATION_PREV` | Previous |
| `FINDINGS$PAGINATION_NEXT` | Next |
| `FINDINGS$PAGINATION_STATUS` | Page {{page}} |

#### Severidade / status (valores)

| Key | EN (ref) |
|---|---|
| `FINDINGS$SEVERITY_CRITICAL` | Critical |
| `FINDINGS$SEVERITY_HIGH` | High |
| `FINDINGS$SEVERITY_MEDIUM` | Medium |
| `FINDINGS$SEVERITY_LOW` | Low |
| `FINDINGS$SEVERITY_INFO` | Info |
| `FINDINGS$STATUS_NEW` | New |
| `FINDINGS$STATUS_TRIAGING` | Triaging |
| `FINDINGS$STATUS_CONFIRMED` | Confirmed |
| `FINDINGS$STATUS_FALSE_POSITIVE` | False positive |
| `FINDINGS$STATUS_DUPLICATE` | Duplicate |
| `FINDINGS$STATUS_RISK_ACCEPTED` | Risk accepted |

#### Ações / drawer / modal

| Key | EN (ref) |
|---|---|
| `FINDINGS$ACTIONS_MENU` | Finding actions |
| `FINDINGS$ACTION_CONFIRM` | Confirm |
| `FINDINGS$ACTION_MARK_FP` | Mark as false positive |
| `FINDINGS$ACTION_DUPLICATE` | Mark as duplicate |
| `FINDINGS$ACTION_ACCEPT_RISK` | Accept risk |
| `FINDINGS$ACTION_OPEN_DETAIL` | View details |
| `FINDINGS$DRAWER_TITLE` | Finding detail |
| `FINDINGS$DRAWER_CLOSE` | Close details |
| `FINDINGS$EVIDENCE_TITLE` | Evidence |
| `FINDINGS$EVIDENCE_EXPAND` | Show evidence |
| `FINDINGS$EVIDENCE_COLLAPSE` | Hide evidence |
| `FINDINGS$EVIDENCE_OPEN_STREAM` | Open in event stream |
| `FINDINGS$EVIDENCE_NO_LINK` | No linked conversation event |
| `FINDINGS$FP_MODAL_TITLE` | Mark as false positive |
| `FINDINGS$FP_REASON_LABEL` | Reason |
| `FINDINGS$FP_REASON_PLACEHOLDER` | Explain why this finding is a false positive |
| `FINDINGS$FP_REASON_REQUIRED` | A reason is required |
| `FINDINGS$FP_SUBMIT` | Mark false positive |
| `FINDINGS$FP_CANCEL` | Cancel |
| `FINDINGS$RISK_REASON_LABEL` | Reason (recommended) |
| `FINDINGS$TOAST_TRIAGE_SUCCESS` | Finding updated |
| `FINDINGS$TOAST_TRIAGE_ERROR` | Couldn’t update finding |

### 12. Copy / tom

- Técnico, curto, sem alarmismo de marketing.
- Empty states orientam a próxima ação (engagement / aguardar scan / limpar filtro).
- Forbidden não revela se o engagement existe.

### 13. Fora de escopo (MVP)

- Diff scan-a-scan completo (banner “indisponível” basta).
- Seletor rico de engagement (MVP: query param + empty; seletor pode ser follow-up com Engagement Manager).
- Bulk triage.
- Edição inline de título/severidade.
- Tema claro dedicado — seguir tema atual do Canvas.

### 14. Critérios para gate Design (pós-Frontend)

Veredicto PASS somente se:

1. Layout e estados batem com este doc (incluindo forbidden / no-engagement).
2. Checklist a11y §8 cumprida.
3. FP exige reason; focus trap ok.
4. Triage oculta sem capability.
5. Tokens `--oh-*` / HeroUI; sem tema paralelo.
6. i18n `FINDINGS$*` completas + `data-testid` estáveis acima.
7. Sem `react-router` em `src/components/`.

---

## Decisões UX chave (resumo para TL)

1. **Findings é nav top-level** capability-gated, após Customize — não enterrado em Settings.
2. **Sem `engagement_id` = empty orientador**, não lista global.
3. **MVP de “diff” = stats chips + “Só novos”**; diff scan-a-scan só se API existir.
4. **FP sempre modal com reason obrigatória**; demais ações mais leves.
5. **Evidence colapsada por default**; deep-link ao event stream só com ids reais.
6. **Mobile = lista compacta**, não tabela horizontal forçada.
7. **Forbidden não vaza dados** (nem contagens).

**Pronto-para-Frontend:** sim  
**Bloqueios Design:** nenhum (dependências 184/182 são de implementação/API, não de definição UX).
