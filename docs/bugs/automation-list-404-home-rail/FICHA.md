# BUG — Toast 404 na home ao listar automações (rail "Automações recomendadas")

## BUG
- **Sintoma**: ao abrir a tela inicial de conversas (`openhands.zadotec.com.br`), aparece um toast/erro "Request failed with status code 404" e o DevTools mostra duas chamadas `GET https://aqdata.hands.zadotec.com.br/api/automation/v1?limit=50&offset=0` retornando 404 `{"detail":"Not Found"}`.
- **Esperado**: a home não deveria tentar listar automações (nem gerar erro visível) quando o backend ativo não expõe o serviço de automação.
- **Reprodução**:
  1. Ter como backend "local" ativo um host que serve apenas o Agent Server puro (ex.: `aqdata.hands.zadotec.com.br`, confirmado via `/openapi.json` como expondo só `/api/conversations/*`, `/api/init`, `/health` — nenhuma rota `/api/automation/*`).
  2. Abrir a home (`/conversations`), onde é renderizado o rail `<RecommendedAutomationsLauncher variant="rail" />` (seção "Automações recomendadas").
  3. O componente chama `useAutomations({ enabled: isRail && activeBackend.backend.kind === "local" })` — sem checar se o backend tem o serviço de automação disponível — disparando `GET /api/automation/v1?limit=50&offset=0` contra o host ativo, que 404a.
- **Escopo**: qualquer usuário cujo backend "local" ativo não tenha o sidecar/ingress de automação montado atrás do agent-server (ou seja qualquer deployment self-hosted que sirva o agent-server puro, sem o roteamento `/api/automation/*`). Gravidade: baixa/média — não quebra a home, mas gera erro visível ao usuário e uma requisição desnecessária a cada carregamento.

## TRACE

### Tracer A — código
Caminho de execução:
- `src/components/features/home/home-chat-launcher.tsx:276` renderiza incondicionalmente `<RecommendedAutomationsLauncher variant="rail" />`.
- `src/components/features/automations/recommended-automations-launcher.tsx:99-103`:
  ```tsx
  const isRail = variant === "rail";
  const { data: automationsData, isLoading: isAutomationsLoading } =
    useAutomations({ enabled: isRail && activeBackend.backend.kind === "local" });
  ```
  Não há import nem uso de `useAutomationHealth` neste arquivo (confirmado por grep) — ao contrário de **todos os outros** consumidores de `useAutomations` no repo: `use-home-automations.ts:32-40`, `automations-list.tsx:129-140`, `conversation-overview-automations-panel.tsx:24-31`, `use-sidebar-onboarding-checklist.ts:57-63`, que exigem `healthData?.status === "ok"` antes de habilitar a query.
- `src/hooks/query/use-automations.ts:19-32` dispara `AutomationService.getAutomations(limit, offset)` assim que `enabled` vira `true`.
- `src/api/automation-service/automation-service.api.ts:274-294` (`listAutomations`) monta `AUTOMATION_BASE_PATH ("/api/automation") + getAutomationEndpoint("list")`, que resolve para `/v1` (`node_modules/@openhands/extensions/automations/interface.json:138`). Para backend "local", usa `localAutomationAxios`, cujo interceptor (linhas 83-111) resolve `config.baseURL = backend.host` via `getEffectiveLocalBackend()` a cada chamada.
- O comentário nas linhas 64-67 do mesmo arquivo é a premissa arquitetural exposta: "Local automation calls go to the automation sidecar that `scripts/dev-with-automation.mjs` mounts behind the local agent-server" — nada no frontend garante que esse sidecar exista atrás do host "local" ativo em produção.

### Tracer B — comportamento
- Não foi possível rodar a suíte de testes no ambiente (Node 20 instalado vs. `engines.node >= 24` do projeto; `vitest`/`jsdom` falha com `TypeError: webidl.util.markAsUncloneable is not a function` mesmo em testes triviais e não relacionados) — reportado como limitação de ambiente, não como resultado do bug.
- Confirmado via `curl` direto contra a rede real:
  - `GET https://aqdata.hands.zadotec.com.br/openapi.json` → 200, sem nenhuma rota `/api/automation/*`.
  - `GET https://aqdata.hands.zadotec.com.br/api/automation/health` → 404.
  - `GET https://aqdata.hands.zadotec.com.br/api/automation/v1?limit=50&offset=0` → 404 (bate exatamente com o bug relatado).
- Inspeção do bundle de produção real (`build/assets/*.js`, hashes batendo com os servidos por `openhands.zadotec.com.br`) confirma que o gate de saúde **existe e está deployado** em `use-home-automations.ts` (minificado, equivalente a `enabled: !demo && isBackendHealthy`) e em `use-sidebar-onboarding-checklist.ts`, mas o mesmo padrão **não aparece** em `recommended-automations-launcher.tsx`, que só checa `isRail && backend.kind === "local"` — confirmando exatamente o achado do Tracer A no código-fonte atual.
- Mocks MSW (`src/mocks/automation-handlers.ts`) simulam `/api/automation/health` e `/api/automation/v1` com os mesmos paths do bug real, descartando divergência de contrato como causa.
- Hipótese alternativa levantada (cache de health-check dessincronizado do host após troca de backend, afetando `use-home-automations`) é plausível para *aquele* hook, mas não é o caminho acionado pelo bug relatado: o texto "Automações recomendadas" na screenshot do usuário identifica exatamente o rail (`RecommendedAutomationsLauncher`), não a seção gateada por `use-home-automations`.

### Convergência
Os dois tracers concordam, de forma independente e com evidência direta em código e em bundle de produção real, que **`RecommendedAutomationsLauncher` é o único consumidor de `useAutomations` sem gate de saúde do backend**, e que esse é exatamente o componente renderizado na tela do bug relatado (seção "Automações recomendadas" da home). Não houve divergência sobre a causa raiz — a hipótese alternativa do Tracer B (cache de health stale) diz respeito a um hook diferente e não invalida a convergência sobre o caminho realmente acionado.

## DIAGNÓSTICO
- **Causa raiz**: `src/components/features/automations/recommended-automations-launcher.tsx:99-103` habilita `useAutomations` checando apenas `isRail && activeBackend.backend.kind === "local"`, sem exigir `useAutomationHealth()` como os demais consumidores do mesmo serviço fazem.
- **Por que causa o sintoma**: em qualquer backend "local" cujo host não tenha o sidecar/ingress de automação montado (como `aqdata.hands.zadotec.com.br`, que serve só o agent-server puro), a chamada a `/api/automation/v1` dispara incondicionalmente a cada carregamento da home e recebe 404, que o React Query/axios propaga como erro visível (toast).
- **Por que passou despercebido**: não existe teste cobrindo `RecommendedAutomationsLauncher` contra um backend sem o serviço de automação disponível (i.e., sem mock de `/api/automation/health` retornando erro) — o padrão de gate correto foi aplicado em três outros lugares do código mas não neste, e nada (lint, tipo, teste) força consistência entre os consumidores de `useAutomations`.
- **Confiança**: alta — dois tracers independentes (estático e comportamental/bundle real) chegaram ao mesmo `arquivo:linha`, e a URL/parâmetros da requisição batem exatamente (`limit=50&offset=0`, path `/v1`).
- **Impacto colateral**: nenhum dado é corrompido; é uma chamada supérflua com erro cosmético. Mas roda em todo carregamento da home nesse tipo de deployment, gerando ruído de log/telemetria e uma má primeira impressão (toast de erro na tela inicial).
- **Correção proposta (direção, não implementação)**: gatear `useAutomations` em `recommended-automations-launcher.tsx` com `useAutomationHealth()` (`healthData?.status === "ok"`), no mesmo padrão usado por `use-home-automations.ts`, `automations-list.tsx`, `conversation-overview-automations-panel.tsx` e `use-sidebar-onboarding-checklist.ts`.
- **Regressão a cobrir**: teste (unit/RTL) para `RecommendedAutomationsLauncher`/`useAutomations` garantindo que, quando o health-check do serviço de automação falha (mock 404/erro), a query de listagem de automações **não é disparada** — hoje não existe esse teste para este componente especificamente, embora exista o padrão de mock em `src/mocks/automation-handlers.ts` para simular a falha de health.
- **Alternativas descartadas**:
  - "Endpoint mudou de path" — descartado: `interface.json` e mocks MSW confirmam que `/api/automation/v1` é o path correto esperado pelo próprio frontend.
  - "Falha silenciosa do health-check" — descartado: `checkHealth()` já captura qualquer exceção e retorna `{status:"error"}` corretamente; o problema é que este componente específico nunca consulta esse resultado.
  - "Regressão recente" — descartado: `git log` nos arquivos relevantes não mostra mudança recente introduzindo o bug; parece ser uma inconsistência original entre os consumidores de `useAutomations`, não uma regressão pontual.
