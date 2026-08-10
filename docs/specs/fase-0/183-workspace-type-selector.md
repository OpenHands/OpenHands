# Spec Técnica — PROJETOSIN-183: Workspace Type Selector

**ADR:** docs/adrs/0001-plataforma-pentest-ia-extensao-openhands.md (accepted)
**Card Plane:** PROJETOSIN-183 — `0603f5d5-5d90-4ab6-a60a-27eea59c2600`
**Agente responsável:** frontend
**Depends on:** PROJETOSIN-182 (capability gate)
**Prioridade:** P1 — libera fluxo de criação de workspace pentest

---

## Contexto

Na criação de workspace, o usuário escolhe o **tipo**: `code` (padrão atual) ou `pentest`. A escolha determina:
- Imagem de runtime (padrão vs. ofensiva)
- MCP tools disponíveis
- Vínculo com Engagement Manager (PROJETOSIN-185)
- Exigência de autorização de escopo antes de provisionar

O seletor é gateado por `pentest.workspace.create` (PROJETOSIN-182) — usuários sem essa capability não veem a opção pentest.

---

## Tipo de workspace

```ts
// src/types/workspace-types.ts
export type WorkspaceType = "code" | "pentest";

export interface WorkspaceCreationParams {
  type: WorkspaceType;
  name: string;
  workingDir?: string;
  engagementId?: string;    // obrigatório se type === "pentest"
  autonomyMode?: "manual" | "semi_autonomous" | "autonomous"; // só pentest
}
```

---

## Componentes UI

### WorkspaceTypeSelector
```tsx
// src/components/features/pentest/workspace-type-selector.tsx
interface WorkspaceTypeSelectorProps {
  value: WorkspaceType;
  onChange: (type: WorkspaceType) => void;
}
```

Renderiza dois cards selecionáveis:
- **Código** — ícone de código, sempre visível
- **Pentest** — ícone de escudo/alvo, visível somente com capability `pentest.workspace.create` (via `CapabilityGate`)

Layout: dois cards side-by-side com descrição curta. Selecionado = borda primária.

i18n keys a adicionar em `src/i18n/translation.json`:
```json
"WORKSPACE_TYPE$CODE_TITLE": "Código",
"WORKSPACE_TYPE$CODE_DESC": "Desenvolvimento e análise de código",
"WORKSPACE_TYPE$PENTEST_TITLE": "Pentest",
"WORKSPACE_TYPE$PENTEST_DESC": "Teste de penetração assistido por IA",
"WORKSPACE_TYPE$PENTEST_UNAVAILABLE": "Requer permissão de pentester"
```

### Integração no fluxo de criação

Modificar `NewConversationButton` / `WorkspaceSelectionForm` para incluir o seletor **antes** do campo de diretório/engajamento.

Quando `type === "pentest"`:
1. Mostrar campo de seleção de engagement (lista do Engagement Manager — PROJETOSIN-185)
2. Se engagement selecionado, mostrar seletor de modo de autonomia
3. Bloquear criação se engagement não tem autorização de escopo registrada (campo `scope_authorized_at` no EngMgr)

---

## Fluxo de criação de workspace pentest

```
[Tipo: Pentest selecionado]
  ↓
[Selecionar Engagement] (GET /api/pentest/engagements — PROJETOSIN-185)
  ↓
[Selecionar Autonomia] (manual / semi_autonomous / autonomous*)
  ↓ *autonomous requer capability pentest.autonomy.autonomous
[Validar scope_authorized_at != null] → erro se null
  ↓
[Criar conversa com metadata pentest]
  ↓
[Provisionar runtime ofensivo via Engagement Manager]
```

---

## Metadata na conversa

Ao criar conversa do tipo pentest, persistir em `conversation-metadata-store.ts`:
```ts
interface PentestConversationMetadata {
  workspace_type: "pentest";
  engagement_id: string;
  autonomy_mode: "manual" | "semi_autonomous" | "autonomous";
  runtime_profile: "web" | "network" | "mobile" | "sast";
}
```

---

## Arquivos a criar/modificar

| Arquivo | Ação |
|---|---|
| `src/types/workspace-types.ts` | Criar — tipos WorkspaceType |
| `src/components/features/pentest/workspace-type-selector.tsx` | Criar — componente selector |
| `src/components/features/home/workspace-selection-form.tsx` | Modificar — adicionar type selector |
| `src/components/features/conversation-panel/new-conversation-button.tsx` | Modificar — passar type ao criar |
| `src/i18n/translation.json` | Adicionar keys WORKSPACE_TYPE$ |
| `src/api/conversation-service/` | Adicionar metadata pentest ao payload |
| `__tests__/components/pentest/workspace-type-selector.test.tsx` | Criar — testes TDD |

---

## Critérios de aceite (QA)

1. **AC-183-1:** Usuário sem `pentest.workspace.create` → seletor não mostra opção Pentest
2. **AC-183-2:** Usuário com capability → opção Pentest visível e selecionável
3. **AC-183-3:** Seleção Pentest sem engagement selecionado → botão Criar desabilitado
4. **AC-183-4:** Seleção Pentest com engagement sem `scope_authorized_at` → erro de validação
5. **AC-183-5:** Criação bem-sucedida de workspace pentest persiste `workspace_type: "pentest"` na metadata
6. **AC-183-6:** Workspace pentest criado mostra ícone/badge visual diferente de workspace de código

---

## Dependências

- **Requer:** PROJETOSIN-182 (CapabilityGate hook)
- **Requer (para engagement list):** PROJETOSIN-185 endpoint `GET /api/pentest/engagements` (pode mockar na Fase 0)
- **Não bloqueia:** 184 (Findings Service independente de UI de criação)

**Estimativa:** 2–3 dias frontend
