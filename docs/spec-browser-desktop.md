# Especificação de Produto — Browser Desktop (Sandbox GUI)

**Feature:** Desktop Linux interativo, embutido no navegador, para o ambiente de sandbox do OpenHands customizado.
**Autor:** Kleber (Heimdall Security)
**Data:** 2026-08-09
**Status:** Accepted v1
**Stack assumida:** KasmVNC + XFCE dentro da imagem Docker `agent-canvas` (mesmo container do agent-server).

---

## 1. Resumo executivo

Hoje o produto entrega ao cliente um ambiente de desenvolvimento encapsulado numa imagem Docker (sandbox do OpenHands customizado), acessível principalmente por terminal e edição de arquivos. Esta feature adiciona um **desktop Linux gráfico completo, renderizado dentro do próprio navegador**, dentro do mesmo container de sandbox — o usuário abre a aba **Desktop** e vê um ambiente XFCE com terminal, gerenciador de arquivos e aplicativos, sem instalar nada localmente e sem cliente VNC externo.

A transmissão é feita por **KasmVNC**, que serve o framebuffer do desktop diretamente por WebSocket, embutido num `iframe` no frontend e roteado pela camada de proxy autenticada que já existe (`ingress.mjs` / `static-server.mjs`). O ciclo de vida do desktop é **sob demanda**: sobe ao abrir a aba e é destruído com o container da sessão.

---

## 2. Contexto e motivação

**Problema.** Boa parte do trabalho de segurança/desenvolvimento que os clientes executam no sandbox se beneficiaria de uma interface gráfica: abrir ferramentas com GUI, inspecionar arquivos visualmente, rodar um navegador dentro do ambiente isolado, usar aplicativos que não têm equivalente em linha de comando. Hoje isso exige que o cliente monte o próprio ambiente ou use túneis manuais.

**Oportunidade.** A infraestrutura central já existe: imagem Docker all-in-one, orquestração de sessão do OpenHands e uma camada de proxy autenticada. Falta apenas a camada de display + streaming dentro da imagem e um handler de proxy dedicado. O resultado é uma experiência equivalente à de produtos como Cursor/Kasm Workspaces, porém dentro do produto e sob controle total da plataforma.

**Não-objetivo desta fase.** Não é meta entregar um desktop persistente multiusuário, aceleração por GPU/vídeo de baixa latência, nem áudio. Esses ficam como evolução futura (ver §11).

---

## 3. Personas e público-alvo

**Cliente final (engenheiro/analista).** Usa o sandbox para desenvolvimento ou análise de segurança; quer uma GUI descartável e isolada, sem instalação local.

**Operador da plataforma (Heimdall).** Precisa controlar o que o ambiente oferece, impor limites de recurso, garantir isolamento e destruir a sessão ao final.

---

## 4. Objetivos e métricas de sucesso

O objetivo é que qualquer sessão Docker de sandbox possa expor um desktop gráfico no navegador com esforço zero de configuração por parte do cliente, mantendo isolamento e limites sob controle da plataforma.

Critérios de sucesso mensuráveis:

- Tempo entre "Abrir Desktop" e desktop interativo visível: **≤ 10 s** em condição normal.
- Overhead de recurso do desktop ocioso: **≤ 512 MB RAM** e **≤ 5% de 1 vCPU** por sessão (quando iniciado).
- Interatividade utilizável (teclado/mouse) em latência de rede típica (~50–100 ms RTT).
- Zero exposição de porta VNC fora do proxy autenticado (verificável por varredura).

---

## 5. Escopo

### 5.1 Dentro do escopo (v1)

- Desktop XFCE leve na imagem `agent-canvas` (`docker/Dockerfile`).
- KasmVNC em loopback (`ports.desktopVnc`, default **6901**).
- Proxy autenticado `createDesktopProxyHandler` em `scripts/desktop-proxy.mjs` (HTTP + WebSocket).
- Aba **Desktop** no painel direito da conversa (junto com Files, Browser, Terminal, Segurança).
- Ciclo de vida **sob demanda** (`POST /api/desktop/start`).
- Allowlist de apps na imagem: `xfce4-terminal`, Thunar, Chromium (wrapper com `--no-sandbox` para Docker).
- Clipboard e upload nativos do KasmVNC.

### 5.2 Fora do escopo (v1)

Áudio; Selkies/WebRTC/GPU; desktop persistente entre sessões; multiusuário no mesmo desktop; personalização de apps pelo cliente; Cloud sandbox remoto (`exposed_urls`).

---

## 6. Requisitos funcionais

**RF-1 — Provisionamento sob demanda.** Ao acionar “Abrir Desktop”, a plataforma inicia display/XFCE/KasmVNC no container se ainda não estiver rodando.

**RF-2 — Acesso pelo navegador.** Painel/iframe no frontend, sem cliente VNC externo.

**RF-3 — Interação completa.** Teclado, mouse, redimensionamento e clipboard bidirecional (KasmVNC).

**RF-4 — Transferência de arquivos.** Upload nativo do KasmVNC.

**RF-5 — Autenticação.** HTTP/WS exigem `X-Session-API-Key` válida (ou cookie HttpOnly emitido por `/start`); porta VNC nunca publicada.

**RF-6 — Ciclo de vida.** Desktop existe enquanto o container existir; estado descartado com o container.

**RF-7 — Controle de ambiente.** Apps definidos na imagem.

**RF-8 — Limites de recurso.** Respeitam `mem`/`cpus`/`pids` do `docker run`/`compose` do operador.

### 6.1 Requisitos não-funcionais

Isolamento por container; sem bind `0.0.0.0` no KasmVNC; fora do Docker a aba mostra empty state `DESKTOP$UNAVAILABLE`.

---

## 7. Experiência do usuário (fluxo)

1. Usuário abre a aba **Desktop** no painel direito.
2. Se o desktop ainda não estiver pronto, vê CTA “Abrir Desktop”.
3. Ao clicar, o frontend chama `POST /api/desktop/start` (com session key).
4. O proxy valida a sessão, sobe KasmVNC se necessário, seta cookie e devolve a URL do iframe.
5. O iframe carrega `/api/desktop/` (same-origin); teclado/mouse funcionam via WebSocket proxied.
6. Fora da imagem Docker: mensagem de indisponibilidade, sem spinner infinito.

---

## 8. Arquitetura

`Xvfb/KasmVNC X` → `XFCE` → `KasmVNC :6901 (127.0.0.1)` → `createDesktopProxyHandler` (`/api/desktop`) → `iframe` na aba Desktop.

Integração:

| Peça | Path |
|------|------|
| Imagem | `docker/Dockerfile`, `docker/desktop/*` |
| Proxy | `scripts/desktop-proxy.mjs` |
| Wiring | `scripts/ingress.mjs`, `scripts/static-server.mjs` |
| UI | `src/routes/desktop-tab.tsx`, `src/components/features/desktop/desktop-panel.tsx` |
| Cliente | `src/api/integrations/desktop-service.ts` |
| Porta | `config/defaults.json` → `ports.desktopVnc` |

---

## 9. Segurança e isolamento

- KasmVNC só em `127.0.0.1`.
- Proxy exige session key validada em `/server_info` do agent-server.
- Cookie `agent-canvas-desktop-auth` (HttpOnly, `Path=/api/desktop`) autentica o iframe/WS.
- Upgrade WebSocket autenticado **antes** do router genérico do ingress.

---

## 10. Critérios de aceite

- Cliente autenticado abre Desktop interativo em ≤ 10 s na imagem Docker.
- Teclado/mouse/clipboard/upload funcionam.
- Varredura: porta 6901 não acessível de fora do container.
- Encerrar o container destrói o desktop.
- `npm run dev` sem imagem: aba visível com empty state, sem crash.
- Fluxo terminal/edição sem regressão.

---

## 11. Riscos e evolução futura

Riscos: overhead de RAM/CPU; latência em redes ruins; superfície GUI. Evoluções: áudio, Selkies, desktop persistente, personalização controlada, Cloud `exposed_urls`.

---

## 12. Decisões fechadas (v1)

| Pergunta | Decisão |
|----------|---------|
| Sobe com toda sessão ou sob demanda? | **Sob demanda** (`POST /api/desktop/start`) |
| Onde vive XFCE + KasmVNC? | **Imagem `agent-canvas`** deste repo |
| Porta interna | **6901** (`ports.desktopVnc`) |
| Allowlist inicial | Terminal XFCE, Thunar, Chromium (default browser via wrapper Docker) |
| Resolução | 1280×720 inicial, resize permitido pelo KasmVNC |
| Cota multi-sessão | Fora do v1 (um desktop por container) |
