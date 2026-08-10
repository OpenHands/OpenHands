# Design Notes — PROJETOSIN-195: UI modos de autonomia

**Status:** pronto para Frontend (definição UX)  
**Card:** PROJETOSIN-195 · `8ba3955a-4268-4d85-88ea-b7d481039ab2`  
**ADR:** [0001](../../adrs/0001-plataforma-pentest-ia-extensao-openhands.md) · Blueprint §5.4  
**Spec canônica:** [195-autonomy-modes-ui.md](./195-autonomy-modes-ui.md)  
**Padrão visual:** `PentestWorkspaceFields` (183) + tokens `--oh-*` / HeroUI; confirmação via `ConfirmationModal` existente  
**Gate Design PASS:** **não** neste entregável — só após implementação FE + revisão a11y.

---

## 1. Princípios

1. **Uma superfície de escolha, dois contextos** — o mesmo padrão visual (radiogroup de 3 modos) na criação e mid-engagement; não inventar um segundo controle “Settings-only”.
2. **Semi-autônomo é o default** — alinhado ao blueprint §5.4 e AC-195-1. Hoje o código 183 inicializa `manual`; 195 corrige para `semi_autonomous`.
3. **Autônomo é restrição explícita** — opção **sempre visível**, `disabled` + tooltip quando falta `pentest.autonomy.autonomous` (não ocultar; AC-195-3 / spec § Segurança).
4. **Allowlist inviolável** — helper text permanente: o modo não amplia escopo; só o grau de confirmação de tools.
5. **Confirmar só ao entrar em Autônomo** — mudança *para* `autonomous` exige modal; Manual ↔ Semi é imediata (com save/feedback).
6. **Tokens existentes** — HeroUI + `--oh-*`; chips/radios inline; sem cards decorativos, sem glow “AI purple”.

---

## 2. Colocação

### 2.1 Criação de workspace pentest (obrigatório)

**Onde:** `PentestWorkspaceFields` — já montado em:

- Home: `workspace-selection-form.tsx`
- Sidebar: `local-new-conversation-menu.tsx`

**Ordem do bloco pentest:**

```
[Tipo: Código | Pentest]
  → Engagement (select)
  → Runtime profile (se UI já existir no fluxo; senão omitir no MVP 195)
  → Modo de autonomia (radiogroup)   ← 195 evolui o bloco atual
  → Helper allowlist + RBAC
  → Launch / Create
```

**Decisão:** manter o seletor **junto do engagement**, não em Settings. Motivo: a escolha faz parte do provisionamento da sessão; o pentester decide antes de Launch.

**Default:** `semi_autonomous` ao selecionar tipo Pentest ou ao resetar o form.

**Visibilidade:** só após `engagementId` selecionado (igual 183) — autonomy sem engagement não tem onde persistir no EngMgr.

### 2.2 Mid-engagement (alteração com sessão ativa)

**Onde (preferido):** chrome da conversa pentest — **header / barra superior** ao lado do `WorkspaceTypeBadge`, como chip compacto “Autonomia: Semi-autônomo ▾”.

```
┌─ Conversation header (pentest) ─────────────────────────────┐
│ [Pentest badge]  Engagement name trunc.  [Autonomia ▾]      │
│                              ↑                              │
│                    AutonomyModeControl (compact)            │
└─────────────────────────────────────────────────────────────┘
│ pending_restart banner (se propagation ≠ applied)           │
│ chat / tabs …                                               │
```

**Alternativa aceitável (se header estiver saturado):** popover “⋯ / sessão” no mesmo header — **não** enterrar só em `/settings`. Settings LLM Profiles não é o lugar da autonomia (evita confundir com modelo LLM).

**Arquivado / read-only:** se o engagement estiver arquivado (quando o estado existir na API), o controle fica `disabled` + helper `PENTEST$AUTONOMY_READONLY`; sem PATCH.

### 2.3 Fora de colocação

- Não no rail do Emulador / Desktop / Files.
- Não como argumento de tool no chat.
- Não em formulário de MCP marketplace.

---

## 3. Componente — contrato FE (sem implementar aqui)

Reutilizar / evoluir o bloco de autonomia de `pentest-workspace-fields.tsx`; extrair se necessário:

```
src/components/features/pentest/
  autonomy-mode-selector.tsx     # radiogroup compartilhado (create + mid)
  autonomy-confirm-modal.tsx     # só transição → autonomous
  autonomy-pending-banner.tsx    # propagation pending_restart
```

Props mínimas do seletor:

| Prop | Tipo | Notas |
|---|---|---|
| `value` | `AutonomyMode` | |
| `onChange` | `(mode) => void` | caller decide confirm / PATCH |
| `disabled` | `boolean` | saving / archived / loading |
| `canUseAutonomous` | `boolean` | de `useHasPentestCapability('pentest.autonomy.autonomous')` |
| `variant` | `"form" \| "compact"` | form = labels+helper; compact = chip no header |
| `id` | string | ligar a `aria-labelledby` / legend |

**Não** passar `autonomy_mode` a clients MCP / tool payloads (AppSec / AC-195-4).

---

## 4. Wireframe textual — radiogroup

### 4.1 Variant `form` (criação)

```
Modo de autonomia
┌──────────────┐ ┌──────────────────┐ ┌────────────────┐
│ ○ Manual     │ │ ● Semi-autônomo  │ │ ○ Autônomo     │
│   / Copiloto │ │   (padrão)       │ │   [se RBAC]    │
└──────────────┘ └──────────────────┘ └────────────────┘
  ↑ selected ring --oh-primary

Helper: A allowlist de escopo do engagement continua valendo em
qualquer modo. O modo Autônomo exige permissão administrativa.

[se autonomous disabled]
  Autônomo — cinza, aria-disabled; tooltip / title:
  “Requer permissão pentest.autonomy.autonomous”
```

Preferência a11y: **um** `role="radiogroup"` com três `role="radio"` (ou HeroUI `RadioGroup`), **não** botões `aria-pressed` soltos (melhoria sobre 183). Manter `data-testid="pentest-autonomy-{mode}"`.

### 4.2 Variant `compact` (header)

```
[ Semi-autônomo ▾ ]  → abre popover com o mesmo radiogroup + helper curto
                     → ConfirmModal se destino = autonomous
                     → saving: chip com spinner pequeno / aria-busy
```

---

## 5. Fluxos

### 5.1 Criação (happy path)

```
Usuário escolhe Pentest
  → default autonomy = semi_autonomous
  → escolhe engagement (scope OK)
  → (opcional) muda Manual / Semi — imediato no estado local
  → se escolhe Autônomo:
        ├─ sem RBAC → opção disabled; não seleciona
        └─ com RBAC → ConfirmModal → Confirm → value = autonomous
  → Launch → POST engagement / createConversation com autonomy_mode
  → metadata local sincronizada
```

### 5.2 Mid-engagement

```
Usuário abre chip Autonomia
  → radiogroup com valor atual (GET engagement / metadata)
  → seleciona Manual ou Semi:
        → PATCH { autonomy_mode }
        → saving → success toast discreto OU update silencioso do chip
        → se propagation = pending_restart → banner
  → seleciona Autônomo:
        → ConfirmModal (foco no Cancel por padrão seguro? → foco no título; Cancel = secondary; Confirm = primary/danger-sutil)
        → Confirm → PATCH
        → Cancel / Escape → fecha; valor permanece o anterior
  → erro PATCH → banner/toast erro; valor UI reverte ao last-known-good
```

### 5.3 Confirmação → Autônomo (obrigatória)

| Elemento | Copy PT (referência) |
|---|---|
| Título | Ativar modo Autônomo? |
| Corpo | No modo Autônomo o agente pode executar ferramentas sem pedir confirmação a cada ação intrusiva. A allowlist de escopo do engagement **permanece inviolável**. Continuar? |
| Confirmar | Ativar Autônomo |
| Cancelar | Cancelar (reusar `BUTTON$CANCEL` se existir) |

**Não** pedir confirmação ao sair de Autônomo → Manual/Semi.

Na **criação**, confirmar no momento da seleção (antes do Launch) — evita Launch acidental em Autônomo sem leitura do risco.

---

## 6. Copy PT (referência) — Frontend gera i18n

Prefixo sugerido pela spec: `PENTEST$AUTONOMY_*`.  
**Reuso:** labels curtas podem mapear às existentes `WORKSPACE_TYPE$AUTONOMY_*` **ou** espelhar nos novos keys para um único prefixo — FE escolhe uma fonte; Design recomenda **unificar em `PENTEST$AUTONOMY_*`** e deprecar labels duplicadas no follow-up (sem quebrar 183 no mesmo PR se risco alto: aliases).

| Key sugerida | PT | EN (ref) |
|---|---|---|
| `PENTEST$AUTONOMY_LABEL` | Modo de autonomia | Autonomy mode |
| `PENTEST$AUTONOMY_MANUAL` | Manual / Copiloto | Manual / Copilot |
| `PENTEST$AUTONOMY_MANUAL_HINT` | Toda ação intrusiva exige sua aprovação | Every intrusive action needs your approval |
| `PENTEST$AUTONOMY_SEMI` | Semi-autônomo | Semi-autonomous |
| `PENTEST$AUTONOMY_SEMI_HINT` | Recon e scans passivos livres; ações ativas pedem confirmação | Passive recon/scans free; active actions need confirmation |
| `PENTEST$AUTONOMY_SEMI_BADGE` | Padrão | Default |
| `PENTEST$AUTONOMY_FULL` | Autônomo | Autonomous |
| `PENTEST$AUTONOMY_FULL_HINT` | Sem gate de confirmação (exceto política de risco máximo) | No confirmation gate (except max-risk policy) |
| `PENTEST$AUTONOMY_HELPER` | A allowlist de escopo do engagement permanece inviolável em qualquer modo. | The engagement scope allowlist remains inviolable in every mode. |
| `PENTEST$AUTONOMY_RBAC_HINT` | O modo Autônomo exige permissão administrativa. | Autonomous mode requires an admin permission. |
| `PENTEST$AUTONOMY_DISABLED_TOOLTIP` | Você não tem permissão para o modo Autônomo. | You don’t have permission to use Autonomous mode. |
| `PENTEST$AUTONOMY_CONFIRM_TITLE` | Ativar modo Autônomo? | Enable Autonomous mode? |
| `PENTEST$AUTONOMY_CONFIRM_BODY` | No modo Autônomo o agente pode executar ferramentas sem pedir confirmação a cada ação intrusiva. A allowlist de escopo permanece inviolável. Continuar? | In Autonomous mode the agent may run tools without per-action confirmation. The scope allowlist remains inviolable. Continue? |
| `PENTEST$AUTONOMY_CONFIRM_ACTION` | Ativar Autônomo | Enable Autonomous |
| `PENTEST$AUTONOMY_SAVING` | Salvando modo de autonomia… | Saving autonomy mode… |
| `PENTEST$AUTONOMY_SAVE_SUCCESS` | Modo de autonomia atualizado | Autonomy mode updated |
| `PENTEST$AUTONOMY_SAVE_ERROR` | Não foi possível atualizar o modo de autonomia | Couldn’t update autonomy mode |
| `PENTEST$AUTONOMY_PENDING_RESTART` | Modo salvo. Reinicie o runtime do engagement para aplicar no MCP. | Mode saved. Restart the engagement runtime to apply it to MCP. |
| `PENTEST$AUTONOMY_LOADING` | Carregando modo de autonomia… | Loading autonomy mode… |
| `PENTEST$AUTONOMY_READONLY` | Engagement arquivado — modo somente leitura | Archived engagement — read-only mode |
| `PENTEST$AUTONOMY_CHIP_ARIA` | Modo de autonomia: {{mode}} | Autonomy mode: {{mode}} |

Hints por opção: opcionais no MVP compact; **obrigatório** o helper global de allowlist (§ Segurança UX).

---

## 7. Estados

| Estado | Quando | UI | `data-testid` |
|---|---|---|---|
| **Default / idle** | Valor conhecido | Radiogroup selecionável | `pentest-autonomy-selector` |
| **Loading** | GET engagement / metadata inicial (mid) | Skeleton curto no chip ou radios `disabled` + `aria-busy` | `pentest-autonomy-loading` |
| **Saving** | PATCH em voo | Controles disabled; chip `aria-busy`; opcional spinner | `pentest-autonomy-saving` |
| **RBAC autonomous disabled** | Sem `pentest.autonomy.autonomous` | Opção Autônomo visível, `aria-disabled`, tooltip | `pentest-autonomy-autonomous` + `…-disabled` |
| **Confirm open** | Transição → autonomous | `ConfirmationModal` focus trap | `pentest-autonomy-confirm-modal` |
| **pending_restart** | `propagation === "pending_restart"` | Banner informativo (não erro) sob o header | `pentest-autonomy-pending-restart` |
| **Error** | PATCH/GET falhou | Toast + inline `role="alert"`; reverte seleção | `pentest-autonomy-error` |
| **Read-only** | Engagement arquivado | Todo o grupo disabled + helper | `pentest-autonomy-readonly` |
| **propagation applied / n/a`** | Sucesso sem restart | Sem banner; chip atualizado | — |

**Hierarquia (mid):** readonly → loading → error → pending_restart (pode coexistir com idle) → idle/saving.

Banner `pending_restart`: tom informativo (`--oh-text-secondary` / surface sutil), ícone + texto; **não** danger. Dismissível só se o produto quiser; MVP: permanece até novo PATCH com `applied` ou remount com GET limpo.

---

## 8. Acessibilidade (WCAG 2.1 AA)

- [ ] **Contraste AA** — texto selected/unselected/disabled sobre `base` / surface; disabled não depende só de opacidade baixa sem texto auxiliar
- [ ] **Radiogroup** — `role="radiogroup"` + `aria-labelledby` (legend); cada opção `role="radio"` + `aria-checked`; setas ←/→ ou ↑/↓ movem foco entre opções (padrão radio)
- [ ] **Foco visível** — `focus-visible:ring` com token `--oh-focus` / primary (fechar D-183-1 no escopo tocado)
- [ ] **Tab order** — criação: engagement → radiogroup → helper → Launch; mid: chip → popover radios → Confirm modal
- [ ] **Teclado** — Space/Enter seleciona; Escape fecha popover/modal; Confirm modal trap
- [ ] **Disabled Autônomo** — permanece no tab order **ou** usa `aria-disabled` em radio focável que anuncia o tooltip (`aria-describedby`); não remover do DOM
- [ ] **Modal** — título `h2`; corpo associado; botões nomeados; foco inicial no título ou Cancel; restore focus no chip/trigger
- [ ] **Saving / loading** — `aria-busy="true"` na região; live region `polite` para sucesso; `assertive` só em erro
- [ ] **Banner pending** — `role="status"`; não roubar foco
- [ ] **Não só cor** — modo atual também em texto no chip compacto
- [ ] **Responsivo** — form: `flex-wrap` dos radios; compact: chip truncável com `title` nativo do label completo
- [ ] **i18n** — zero literals de UI; `t(I18nKey.PENTEST$AUTONOMY_…)`

### Riscos a11y (atenção FE)

1. **Chips `aria-pressed` (183)** não são radiogroup verdadeiro — migrar para radio evita estado “vários pressed” e melhora leitores de tela.
2. **Tooltip só em hover** no Autônomo disabled — garantir `title` + texto em `aria-describedby` (ou hint sempre visível no form).
3. **Confirm modal** — não usar `autoFocus` no botão destrutivo/confirm; preferir foco seguro (Cancel ou container).
4. **Popover no header** — fechar ao selecionar Manual/Semi; manter aberto só se Confirm estiver empilhado.

---

## 9. Tokens / HeroUI

| Uso | Token / padrão |
|---|---|
| Borda / superfície | `--oh-border`, `--oh-surface`, `--oh-surface-raised` |
| Selecionado | borda/`ring` `--oh-primary` + fundo `primary/15` (já em AutonomyChip) |
| Texto | `--oh-foreground`, `--oh-text-secondary`, `--oh-muted` |
| Erro | `--oh-status-error` / `--oh-color-danger` |
| Banner pending | fundo muted / border; sem danger |
| Modal | `ConfirmationModal` / `BaseModal*` existentes |
| Focus | `--oh-focus` ou ring primary |

**Proibido:** cards com shadow multi-layer; pills de marketing; purple/indigo glow; segundo tema.

---

## 10. Segurança UX (explícito)

1. **RBAC:** modo Autônomo restrito a `pentest.autonomy.autonomous` — UI disabled + tooltip; nunca “esconder e permitir via atalho de teclado”.
2. **Allowlist:** helper permanente — autonomia **não** altera nem ignora a allowlist de escopo (§4.3 / blueprint §5.4).
3. **Source of truth runtime:** env `PENTEST_AUTONOMY_MODE` via EngMgr; UI nunca injeta autonomy em args MCP.
4. **Confirmação humana** ao elevar para Autônomo — reduz ativação acidental.

---

## 11. `data-testid` estáveis

| ID | Onde |
|---|---|
| `pentest-autonomy-selector` | Root radiogroup |
| `pentest-autonomy-manual` | Opção |
| `pentest-autonomy-semi_autonomous` | Opção |
| `pentest-autonomy-autonomous` | Opção |
| `pentest-autonomy-chip` | Trigger compact header |
| `pentest-autonomy-confirm-modal` | Modal |
| `pentest-autonomy-confirm-action` | Botão confirmar |
| `pentest-autonomy-pending-restart` | Banner |
| `pentest-autonomy-error` | Erro inline |
| `pentest-autonomy-saving` | Região saving |
| `pentest-autonomy-loading` | Loading mid |

Manter compat com ids 183 (`pentest-autonomy-{mode}`) onde possível.

---

## 12. Ajustes menores vs spec 195 (UX)

Documentados aqui; spec atualizada em paralelo se necessário:

| # | Spec / código hoje | Decisão Design |
|---|---|---|
| U-1 | Código 183 default `manual` | **Default UI = `semi_autonomous`** (AC-195-1) |
| U-2 | `CapabilityGate` **oculta** Autônomo | **Mostrar disabled + tooltip** (spec § Segurança item 5) |
| U-3 | Chips `aria-pressed` | Preferir **radiogroup** HeroUI/nativo |
| U-4 | Label “Manual” só | Copy **“Manual / Copiloto”** alinhada ao blueprint |
| U-5 | Mid-session “Settings / header” | **Header da conversa pentest** (não Settings LLM) |
| U-6 | Confirm só mid-session implícito | Confirm **também na criação** ao selecionar Autônomo |

Nenhuma mudança de contrato API; só UX.

---

## 13. Fora de escopo (UI)

- Playbooks multi-agent / Fase 4  
- Editar allowlist de escopo  
- Approve-tool completo (Fase 4) — só copy coerente se toast `confirmation_required` já existir  
- Device físico (194)  
- Troca de autonomia via mensagem do agente  

---

## 14. Critérios de pronto para Frontend

| # | Critério | OK? |
|---|---|---|
| 1 | Colocação create + mid documentada | sim |
| 2 | Copy PT + keys sugeridas | sim |
| 3 | Estados loading/saving/RBAC/pending/erro | sim |
| 4 | Confirm → autonomous | sim |
| 5 | a11y radiogroup + riscos | sim |
| 6 | Segurança UX (RBAC + allowlist) explícita | sim |
| 7 | Gate Design PASS emitido | **não** (pós-FE) |

**Pronto-para-FE: sim** — Tech Lead pode despachar Frontend neste worktree.

---

## 15. Handoff Tech Lead

- Path: `docs/specs/fase-3/195-design-notes.md`
- Evoluir `PentestWorkspaceFields` + controle compacto no header pentest; PATCH EngMgr + banner `pending_restart`.
- Corrigir default para `semi_autonomous`; Autônomo always-visible disabled.
- Não auto-assinar gate Design; após PR FE → `docs/gates/PROJETOSIN-195/design.md`.
