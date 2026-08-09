# Especificação de Produto — Browser Desktop (Sandbox GUI)

**Feature:** Desktop Linux interativo, embutido no navegador, para o ambiente de sandbox do OpenHands customizado.
**Autor:** Kleber (Heimdall Security)
**Data:** 2026-08-09
**Status:** Draft v1
**Stack assumida:** KasmVNC + XFCE dentro da imagem Docker de sandbox por cliente.

---

## 1. Resumo executivo

Hoje o produto entrega ao cliente um ambiente de desenvolvimento encapsulado numa imagem Docker (sandbox do OpenHands customizado), acessível principalmente por terminal e edição de arquivos. Esta feature adiciona um **desktop Linux gráfico completo, renderizado dentro do próprio navegador**, dentro do mesmo container de sandbox — o usuário abre uma aba/painel e vê um ambiente XFCE com terminal, gerenciador de arquivos e aplicativos, sem instalar nada localmente e sem cliente VNC externo.

A transmissão é feita por **KasmVNC**, que serve o framebuffer do desktop diretamente por WebSocket, embutido num `iframe` no frontend e roteado pela camada de proxy autenticada que já existe (`ingress.mjs` / `static-server.mjs`). O ciclo de vida do desktop acompanha o ciclo de vida da sessão de sandbox: efêmero, isolado por cliente, com limites de recurso controlados pela plataforma.

---

## 2. Contexto e motivação

**Problema.** Boa parte do trabalho de segurança/desenvolvimento que os clientes executam no sandbox se beneficiaria de uma interface gráfica: abrir ferramentas com GUI, inspecionar arquivos visualmente, rodar um navegador dentro do ambiente isolado, usar aplicativos que não têm equivalente em linha de comando. Hoje isso exige que o cliente monte seu próprio ambiente ou use túneis manuais.

**Oportunidade.** A infraestrutura central já existe: geração de imagem Docker por cliente, orquestração de sessão do OpenHands e uma camada de proxy autenticada. Falta apenas a camada de display + streaming dentro da imagem e um handler de proxy dedicado. O resultado é uma experiência equivalente à de produtos como Cursor/Kasm Workspaces, porém dentro do produto e sob controle total da plataforma.

**Não-objetivo desta fase.** Não é meta entregar um desktop persistente multiusuário, aceleração por GPU/vídeo de baixa latência, nem áudio. Esses ficam como evolução futura (ver §11).

---

## 3. Personas e público-alvo

**Cliente final (engenheiro/analista).** Usa o sandbox para desenvolvimento ou análise de segurança; quer uma GUI descartável e isolada, sem instalação local.

**Operador da plataforma (Heimdall).** Precisa controlar o que o ambiente oferece, impor limites de recurso, garantir isolamento e destruir a sessão ao final.

---

## 4. Objetivos e métricas de sucesso

O objetivo é que qualquer sessão de sandbox possa expor um desktop gráfico no navegador com esforço zero de configuração por parte do cliente, mantendo isolamento e limites sob controle da plataforma.

Critérios de sucesso mensuráveis:

- Tempo entre "abrir desktop" e desktop interativo visível: **≤ 10 s** em condição normal.
- Overhead de recurso do desktop ocioso: **≤ 512 MB RAM** e **≤ 5% de 1 vCPU** por sessão.
- Interatividade utilizável (teclado/mouse) em latência de rede típica (~50–100 ms RTT): sem travamentos perceptíveis para uso de terminal e navegação de janelas.
- Zero exposição de porta VNC fora do proxy autenticado (verificável por varredura).

---

## 5. Escopo

### 5.1 Dentro do escopo (v1)

A entrega inclui: um desktop XFCE leve dentro da imagem de sandbox; transmissão via KasmVNC servida por WebSocket; um handler de proxy dedicado (`createDesktopProxyHandler`) no padrão dos handlers atuais, com autenticação reaproveitada da plataforma; embutir o desktop num `iframe` no frontend; ciclo de vida atrelado à sessão de sandbox (sobe com a sessão, é destruído com ela); limites de CPU/memória/processos por container; clipboard bidirecional e upload de arquivo (recursos nativos do KasmVNC); e uma allowlist de aplicativos/menu definida pela plataforma na construção da imagem.

### 5.2 Fora do escopo (v1)

Ficam de fora nesta fase: áudio; aceleração por GPU e streaming de vídeo por WebRTC (Selkies); desktop persistente entre sessões; multiusuário simultâneo no mesmo desktop; e personalização do ambiente pelo cliente além do que a imagem oferece.

---

## 6. Requisitos funcionais

**RF-1 — Provisionamento do desktop.** Ao iniciar (ou sob demanda dentro de) uma sessão de sandbox, a plataforma inicia o servidor de display virtual, o window manager/XFCE e o KasmVNC dentro do container.

**RF-2 — Acesso pelo navegador.** O cliente acessa o desktop por um painel/iframe no frontend, sem instalar cliente VNC nem plugin.

**RF-3 — Interação completa.** Teclado, mouse, redimensionamento de resolução e clipboard bidirecional funcionam dentro do desktop.

**RF-4 — Transferência de arquivos.** Upload de arquivo do host do usuário para dentro do desktop via recurso nativo do KasmVNC.

**RF-5 — Autenticação.** O acesso ao WebSocket do desktop é autenticado pela mesma camada de auth da plataforma; nenhuma sessão é acessível sem credencial válida.

**RF-6 — Ciclo de vida.** O desktop existe enquanto a sessão de sandbox existir; ao encerrar a sessão, o desktop e todo o seu estado são destruídos.

**RF-7 — Controle de ambiente.** O conjunto de aplicativos disponíveis (menu, atalhos) é definido pela plataforma na imagem; o cliente não pode instalar fora dos limites concedidos.

**RF-8 — Limites de recurso.** Cada desktop respeita limites de CPU, memória e número de processos definidos por sessão.

## 6.1 Requisitos não-funcionais

Isolamento: cada sessão roda em seu próprio container, em rede fechada, sem exposição direta da porta do KasmVNC — todo tráfego passa pelo proxy autenticado. Segurança: o desktop nunca é alcançável a partir da internet pública diretamente. Efemeridade: nenhum dado do desktop persiste após o fim da sessão, salvo o que a plataforma explicitamente montar. Observabilidade: métricas de uso de recurso e status de sessão do desktop devem ser coletáveis pela plataforma. Portabilidade: a camada de display é adicionada à imagem base existente sem quebrar o fluxo atual de terminal/edição.

---

## 7. Experiência do usuário (fluxo)

O cliente abre uma sessão de sandbox como hoje. Um controle "Abrir Desktop" fica disponível no frontend. Ao acioná-lo, o painel carrega um `iframe` apontando para a rota de proxy da sessão; em poucos segundos o desktop XFCE aparece com wallpaper, barra de tarefas e um terminal, semelhante ao mostrado em referências como Cursor/Kasm. O cliente interage normalmente — abre o terminal, o gerenciador de arquivos, um navegador interno, etc. Ao encerrar a sessão, o desktop desaparece e o estado é descartado.

---

## 8. Arquitetura (visão de alto nível)

O fluxo de dados, da renderização ao navegador do cliente:

`Xvfb` (display virtual) → `XFCE` (desktop/window manager) → `KasmVNC` (serve o framebuffer nativamente por WebSocket) → `createDesktopProxyHandler` no `ingress.mjs` (proxy de WebSocket autenticado) → `iframe` no frontend.

Pontos de integração com o que já existe:

- **Imagem de sandbox:** a stack de display (Xvfb, XFCE, KasmVNC e a allowlist de apps) é adicionada à imagem Docker gerada por cliente.
- **Proxy:** um novo handler `createDesktopProxyHandler`, análogo aos `createAppwriteProxyHandler` / `createDependencyTrackProxyHandler` já presentes, faz o proxy do WebSocket do KasmVNC do container correto, autenticando pela camada da plataforma.
- **Orquestração:** o start/stop do desktop acompanha o start/stop da sessão de sandbox do OpenHands.

Por que KasmVNC (e não noVNC clássico ou Selkies): o KasmVNC fala WebSocket nativamente (dispensa `websockify` separado), traz clipboard, upload e melhor compressão prontos, reduzindo o número de peças a manter. noVNC+x11vnc+websockify seria mais componentes manuais; Selkies (WebRTC/GPU) só se justifica quando latência de vídeo/áudio virar requisito — ver §11.

---

## 9. Segurança e isolamento

Cada sessão é um container isolado, em rede fechada, com a porta do KasmVNC jamais exposta diretamente — o único caminho de acesso é o WebSocket proxied e autenticado. O acesso exige credencial válida da plataforma, herdada da camada de auth existente. Limites de CPU, memória e PIDs por container evitam que uma sessão degrade as demais. O ambiente é efêmero: destruído ao final da sessão, sem persistência de estado por padrão. A superfície de aplicativos é definida pela plataforma na imagem, não pelo cliente.

---

## 10. Critérios de aceite

A feature é considerada pronta quando: um cliente autenticado consegue abrir, a partir de uma sessão de sandbox, um desktop XFCE interativo no navegador em ≤ 10 s, sem instalação local; teclado, mouse, redimensionamento, clipboard bidirecional e upload de arquivo funcionam; uma varredura confirma que a porta do KasmVNC não é acessível fora do proxy autenticado; encerrar a sessão destrói o desktop e seu estado; os limites de CPU/memória/PIDs por sessão são aplicados e verificáveis; e o fluxo existente de terminal/edição do sandbox continua funcionando sem regressão.

---

## 11. Riscos e evolução futura

Principais riscos: overhead de recurso por sessão pode limitar densidade de containers por host (mitigação: XFCE leve, limites estritos, medir overhead ocioso); latência de interação em redes ruins pode degradar a experiência (mitigação: começar com casos de terminal/janelas, avaliar Selkies se vídeo virar requisito); e a superfície de segurança cresce ao expor uma GUI (mitigação: rede fechada, proxy autenticado, container efêmero, allowlist de apps).

Evoluções futuras candidatas: áudio; aceleração por GPU e streaming por WebRTC via Selkies para baixa latência; desktop persistente opcional por cliente; e personalização controlada do ambiente pelo próprio cliente.

---

## 12. Perguntas em aberto

Fica a decidir: se o desktop sobe automaticamente com toda sessão ou apenas sob demanda (impacta densidade de recursos); qual o conjunto exato de aplicativos da allowlist inicial; qual a política de resolução/escalonamento do display; e se haverá cota de sessões de desktop simultâneas por cliente.
