# Spec Técnica — PROJETOSIN-195: UI modos de autonomia (Manual / Semi / Autônomo)

**ADR:** docs/adrs/0001-plataforma-pentest-ia-extensao-openhands.md (accepted) — autonomia · Blueprint §5.4  
**Card Plane:** PROJETOSIN-195 — `8ba3955a-4268-4d85-88ea-b7d481039ab2`  
**Agentes:** design (antes) → frontend (+ backend mínimo EngMgr/propagate se faltar)  
**Prioridade:** P1  
**Base git:** `4eb43759b`  
**Branch:** `feat/fase3-autonomy-ui-195`  
**Worktree:** `.tmp/worktrees/195`  
**PR target:** fork `klebersjunior/OpenHands` only

---

## Objetivo

Seletor de **modo de autonomia** na UI (engagement / sessão pentest), alinhado ao blueprint:

| Modo | Valor canônico | Comportamento (já no server) |
|---|---|---|
| Manual / Copiloto | `manual` | Toda tool intrusiva (e MVP: gate amplo) exige confirmação |
| Semi-autônomo (**padrão**) | `semi_autonomous` | Recon/passivo livre; ativas com confirmation gate |
| Autônomo | `autonomous` | Sem gate (exceto política max-risk futura); allowlist inviolável |

A UI **persiste** a escolha no Engagement Manager e garante que o runtime MCP recebe `PENTEST_AUTONOMY_MODE` — **nunca** via argumento de tool MCP.

---

## Segurança (não negociável — fix 187)

1. Source of truth runtime: env `PENTEST_AUTONOMY_MODE` lido por `services/mcp-servers/shared/confirmation.py::get_autonomy_mode()`.  
2. Schemas MCP **não** expõem `autonomy_mode`.  
3. UI/API EngMgr podem enviar `autonomy_mode` no **CRUD do engagement** (humano autenticado + session key) — isso **não** é o agente MCP.  
4. Mudança de modo → atualizar env dos containers MCP/runtime do engagement (recreate ou restart service com novo env). Se restart completo for pesado no MVP: documentar “aplica na próxima provision/restart” **e** preferir PATCH que dispara reload do env no compose project.  
5. Modo `autonomous` gated por capability RBAC (ex. `pentest.autonomy.autonomous` / perfil admin) — usuários sem capability veem opção disabled + tooltip.

---

## Ordem de trabalho

1. **Design** — `docs/specs/fase-3/195-design-notes.md` no worktree 195: colocação do seletor, copy, estados, a11y, RBAC disabled.  
2. **Frontend** — implementar conforme design + esta spec.  
3. **Backend** (se necessário no mesmo PR): endpoint PATCH engagement já existe parcialmente — completar propagate env.  
4. Gates: Design (UI) → QA → AppSec.

---

## Contratos

### Tipos (já existem)

`src/types/workspace-types.ts`:

```ts
export type AutonomyMode = "manual" | "semi_autonomous" | "autonomous";
```

Reutilizar; não criar union paralela.

### Engagement Manager

| Op | Uso |
|---|---|
| `POST /engagements` | `autonomy_mode` default `semi_autonomous` (já) |
| `PATCH /engagements/{id}` | body `{ autonomy_mode }` — garantir schema Update + service |
| `GET /engagements/{id}` | retorna modo atual |

Após PATCH bem-sucedido, EngMgr deve:

1. Persistir DB  
2. Best-effort: atualizar env `PENTEST_AUTONOMY_MODE` no project compose do engagement (runtime + mcp sidecars se aplicável)  
3. Resposta inclui `autonomy_mode` efetivo + `propagation: "applied" | "pending_restart" | "n/a"`

Client frontend: `src/api/pentest/engagement-service.ts` (estender se já houver de 185/183).

### Conversation metadata

Já há `autonomy_mode` em `PentestConversationMetadata` / `use-create-conversation`. Manter sincronizado:

- Criação workspace pentest: envia modo escolhido  
- Troca mid-session: PATCH engagement + atualizar metadata local; **não** patchar settings LLM com autonomy fake

### Confirmation UI (mínimo)

Gate `confirmation_required` já existe server-side. MVP 195:

- Seletor de modo + persistência + propagate  
- Superfície de “Approve tool” pode permanecer stub se 187 ainda stub — **não** expandir Fase 4; se já houver toast/error no chat para `confirmation_required`, apenas garantir copy i18n coerente com o modo Manual/Semi

---

## UI — requisitos para Design

**Design notes:** [195-design-notes.md](./195-design-notes.md) (canônico para FE).

**Colocação (fechada pelo Design):**

1. Fluxo criação workspace pentest (`PentestWorkspaceFields`, junto engagement / runtime profile) — seletor obrigatório com **default `semi_autonomous`** (corrigir init `manual` herdado de 183)  
2. Header da conversa pentest (chip compacto ao lado do badge) — **não** Settings LLM; alterar modo com confirm dialog se mudando **para** Autônomo (também na criação ao selecionar Autônomo)

**Estados:**

- Loading / saving  
- Error propagate (`pending_restart` → banner informativo)  
- Autonomous **sempre visível**, `disabled` + tooltip sem capability RBAC (não ocultar via CapabilityGate vazio)  
- Read-only se engagement arquivado (se estado existir)

**a11y:** radiogroup HeroUI/nativo (preferir a chips `aria-pressed`); labels via i18n; foco seguro no dialog de confirmação (não no botão Confirm).

**Tokens:** HeroUI + `--oh-*`; sem cards decorativos desnecessários — controle de formulário inline.

**Segurança UX:** helper permanente — allowlist de escopo inviolável em qualquer modo; Autônomo gated por RBAC.

i18n prefix sugerido: `PENTEST$AUTONOMY_*` (pode aliasar labels `WORKSPACE_TYPE$AUTONOMY_*` existentes).

### Ajustes UX vs tip pré-design (U-1…U-6)

Ver §12 de [195-design-notes.md](./195-design-notes.md) — default Semi, Autônomo disabled+tooltip, radiogroup, copy “Manual / Copiloto”, colocação no header, confirm também na criação.
---

## AC testáveis

| ID | Critério |
|---|---|
| AC-195-1 | Default UI = Semi-autônomo; valor enviado `semi_autonomous` |
| AC-195-2 | Trocar para Manual → PATCH EngMgr → GET reflete `manual` |
| AC-195-3 | Sem capability autonomous → opção Autônomo não selecionável |
| AC-195-4 | Vitest: client **não** adiciona `autonomy_mode` a payloads MCP tool |
| AC-195-5 | Com mock EngMgr, banner `pending_restart` quando propagation ≠ applied |
| AC-195-6 | i18n keys completas (`make-i18n` / check completeness no escopo tocado) |

---

## Gates

| Gate | Foco |
|---|---|
| Design | PASS em notes + review UI |
| QA | AC-195-* |
| AppSec | Confirmar ausência de autonomy em schema MCP; PATCH EngMgr authz; autonomous RBAC |

---

## Fora de escopo

- Playbooks multi-agent (Fase 4)  
- Alterar allowlist de escopo  
- Device físico (194)

---

## Entrega

1. Design notes no worktree → comentário Plane  
2. Frontend PR `Plane: PROJETOSIN-195`  
3. Não mergear sem Design + QA + AppSec PASS (revisor ≠ autor)
