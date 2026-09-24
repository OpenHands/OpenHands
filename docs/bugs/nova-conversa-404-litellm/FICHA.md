# BUG — Nenhuma conversa nova funciona (`Internal error` / `litellm.NotFoundError`)

**Status: RESOLVIDO.** Duas causas independentes, ambas de infraestrutura (fora deste repositório).

> Registro sanitizado: hostnames, domínios e nomes de provedor foram substituídos por
> placeholders (`<host-central>`, `<host-remoto>`, `<upstream-llm>`). Nenhum valor de
> credencial aparece aqui.

## BUG
- **Sintoma**: toda conversa nova falhava. Com perfis LLM customizados aparecia
  `litellm.NotFoundError: OpenAIException - Not Found`; no caminho padrão, `Internal error`.
- **Esperado**: a conversa inicia e o agente responde.
- **Reprodução**: abrir o Canvas, digitar qualquer mensagem, enviar. Falhava 100% das vezes.
- **Escopo**: começou às ~07:40 de um dia para o outro (log do agent-server: 0 falhas na
  véspera, 7 no dia). Gravidade máxima — produto inutilizável.

## TRACE

### Tracer A — código (estático)
Percorreu o fluxo de seleção de modelo até a criação da conversa e descartou, linha a linha,
todo o diff não commitado do frontend (`model-selector.tsx`, `use-chat-input-profile-state.ts`,
`manage-backends-modal.tsx`, `agent-profiles-manager.tsx`, `skills-service.ts`, `vault-*`).
Nenhum corrompe o payload: o frontend envia apenas uma referência de perfil.

### Tracer B — comportamento (dinâmico)
Testes: 9 falhas em 56, todas explicadas por wrappers desatualizados
(`<ActiveBackendProvider>` ausente) e por mudança intencional em
`use-chat-input-profile-state.ts`. Nenhuma correlacionada ao bug. `tsc --noEmit` limpo.

### Tracer C — E2E no app + inspeção do host
Reproduziu ao vivo e então investigou o host e os logs do agent-server, onde as duas causas
apareceram.

### Convergência
Os três convergem: **o frontend não tem culpa**. As causas estavam na infraestrutura.

## DIAGNÓSTICO

### Causa raiz 1 — proxy de rate-limit duplicava o path
- **Onde**: `proxy.py` do rate-limit proxy local, combinado com seu `config.json`.
- **O quê**: `upstream_base_url` terminava em `/router/v1`, e o agent-server chama o proxy em
  `http://127.0.0.1:<porta>/router/v1/...`. Como a rota é catch-all (`/{path:path}`), o `path`
  capturado já continha `router/v1/...`, e `url = f"{upstream_base}/{path}"` produzia
  `<upstream-llm>/router/v1/router/v1/chat/completions` → **404**.
- **Sintoma**: `litellm.NotFoundError: OpenAIException - Not Found` em todo perfil que passa
  pelo proxy.
- **Correção**: remover o sufixo `/router/v1` do `upstream_base_url`. O proxy relê o config a
  cada requisição — não precisa reiniciar.
- **Validação**: log do proxy passou de `404 Not Found` para `200 OK` em
  `POST /router/v1/chat/completions`.

### Causa raiz 2 — Claude Code recusa `bypassPermissions` sob root
- **Onde**: binário do Claude Code embutido em `@anthropic-ai/claude-agent-sdk`, usado pelo
  provider ACP `claude-code`.
- **O quê**: o binário contém
  ```js
  if (t === "bypassPermissions" || r) {
    if (process.getuid() === 0 && process.env.IS_SANDBOX !== "1" && !CLAUDE_CODE_BUBBLEWRAP)
      console.error("--dangerously-skip-permissions cannot be used with root/sudo privileges for security reasons"),
      process.exit(1)
  }
  ```
  O serviço roda como root e o provider usa session mode `bypassPermissions` → o subprocesso
  morria em `exit(1)` logo após `set_session_mode` → o ACP devolvia `Internal error`
  (`acp_agent.py:2266`).
- **Por que começou naquele dia**: o cache do `npx` foi recriado às 07:40:17 — o minuto exato
  da primeira falha. O `claude-agent-acp` está pinado, mas o `@anthropic-ai/claude-agent-sdk`
  que ele puxa **não é**, e a versão nova trouxe/endureceu essa checagem.
- **Por que afetava TODA conversa**: o seletor do composer escolhe apenas o **perfil LLM**;
  quem executa a conversa é o **perfil de agente**, que por padrão é o ACP. Logo, qualquer
  perfil LLM escolhido continuava passando pelo ACP quebrado.
- **Correção**: `Environment=IS_SANDBOX=1` via override do systemd — a válvula de escape
  oficial do próprio Claude Code para ambientes isolados. Restaura o comportamento anterior
  sem introduzir risco novo (root + `bypassPermissions` já era o estado que funcionava).
- **Validação**: variável presente no serviço e nos processos filhos; log passou a mostrar
  `Sending ACP prompt` → `ACP prompt returned in 1.4s` sem `Failed to start ACP server`; no
  app, o agente respondeu e a conversa foi renomeada automaticamente pelo LLM.

### Por que passou despercebido
Nenhum teste cobre o caminho real de execução do agente (ACP sob root), e o `Internal error`
do ACP é genérico — o motivo real (`exit(1)` do subprocesso) não sobe para a UI. Os testes de
`model-selector` já estavam quebrados no working tree, mascarando cobertura naquela área.

### Risco residual
O `@anthropic-ai/claude-agent-sdk` **não é pinado** pelo `claude-agent-acp`. Qualquer recriação
futura do cache do `npx` pode trazer outra versão com mudança de comportamento. Se voltar a
quebrar, comparar a data de modificação do cache `_npx/` com o horário da primeira falha no log.

### Bug secundário confirmado (pertence ao agent-server)
`POST /api/agent-profiles/{id}/activate` grava apenas `active_agent_profile_id` e **nunca** o
bloco legado `agent_settings` — o próprio código documenta isso (`agent_profiles_router.py`, e a
resposta traz `agent_settings_applied: false`). Como `conversation_service.py` só resolve o
perfil quando a requisição traz `agent_profile_id`, o bloco legado continua sendo o fallback
real. Resultado: o badge "Padrão" na UI pode divergir do agente que de fato executa a conversa.
Corrigir exige mudança no `software-agent-sdk`, não neste repositório.

### Itens menores não corrigidos
1. Testes quebrados no working tree: `model-selector*.test.tsx` (falta `<ActiveBackendProvider>`
   no wrapper) e `chat-input-profile-picker.test.tsx` (desatualizado após mudança intencional em
   `use-chat-input-profile-state.ts`).
2. Saturação transitória observada (~2 min): o agent-server faz chamadas HTTP para si mesmo
   (`127.0.0.1:<porta>/api/settings/secrets/...`) e elas deram `ReadTimeout` durante o startup
   pesado de uma conversa (chromium + tmux + tools). Recuperou sozinho. Padrão de auto-chamada
   frágil, mas pré-existente.

## FERRAMENTAS RESULTANTES
- `tools/fix-openhands-host.sh` — aplica as duas correções acima num host, idempotente.
- `tools/deploy-remotes.sh` — deploy por push via o agent-server (HTTPS), para hosts sem SSH.
