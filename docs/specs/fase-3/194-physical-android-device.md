# Spec Técnica — PROJETOSIN-194: Device Android físico + reconexão ADB

**ADR:** docs/adrs/0001-plataforma-pentest-ia-extensao-openhands.md (accepted) — Implementation notes §3 · Blueprint §6.4  
**Card Plane:** PROJETOSIN-194 — `f1cf8afe-6a68-4f0f-9746-9727429a6f2a`  
**Agentes:** backend (lead) → frontend (UI estado device na aba Emulador)  
**Prioridade:** P1 — depende contrato IPC PROJETOSIN-193  
**Base git:** `4eb43759b` (+ rebase/merge da branch 193 quando estável)  
**Branch:** `feat/fase3-physical-device-194`  
**Worktree:** `.tmp/worktrees/194`  
**PR target:** fork `klebersjunior/OpenHands` only  
**Blocked until:** handlers `adb:*` + `window.pentestNative` de 193 mergeados **ou** disponíveis na branch base (contrato em [193-electron-docker-adb-ipc.md](./193-electron-docker-adb-ipc.md))

---

## Objetivo

Permitir que, no **modo Electron local** com hooks habilitados, o pentester use um **device Android físico** (USB ou `adb connect` LAN) como alvo dinâmico, com:

1. Descoberta / listagem de devices  
2. Seleção do serial alvo do engagement  
3. **Reconexão** com backoff sem reiniciar o engagement  
4. Eventos connect/disconnect para a UI (aba Emulador reutilizada)

`mcp-mobile` continua falando com um **endpoint ADB genérico**; a origem (emulador container vs host físico) é config + modo de execução.

---

## Dependência 193

Consumir **apenas** o contrato público:

- `pentestNative.adb.devices | connect | disconnect | waitForDevice`
- `pentestNative.getStatus()`

Não adicionar verbos IPC perigosos neste card. Install/shell/Frida permanecem no **mcp-mobile** / runtime via ADB TCP.

### Ponte runtime ↔ host ADB (decisão MVP)

| Opção | MVP? |
|---|---|
| A) Runtime no compose usa `ADB_HOST` apontando para gateway que encaminha ao adb do host | Preferida se já houver padrão de rede engagement |
| B) Electron sobe `adb reverse` / socat local e injeta `ADB_HOST=host.docker.internal` (Desktop) | Aceitável no Electron |
| C) Agente chama só tools que o host executa via IPC | **Rejeitada** — quebra adaptador único mcp-mobile |

**Decisão:** Opção **B** no Electron (documentar `host.docker.internal` / IP do host gateway). Backend documenta env `PENTEST_ADB_TARGET=physical|emulator` no engagement metadata; provisioner **não** precisa recriar compose — só atualizar env do runtime ou um sidecar `adb-proxy` se já existir na Fase 2. Se Fase 2 só tem emulator interno, 194 adiciona doc + flag de override `ADB_HOST`/`ADB_PORT` para o host quando `physical`.

---

## Reconexão (contrato comportamental)

```
loop:
  devices = adb.devices()
  if selectedSerial in devices && state == "device":
    emit connected; reset backoff
  else:
    emit disconnected(reason)
    if lastWasTcpConnect: adb.connect(host, port)
    adb.waitForDevice(serial, timeoutMs)
    backoff = min(backoff * 2, maxBackoff)  # ex. 1s → 2 → 4 → … → 30s
```

| Parâmetro | Default |
|---|---|
| poll interval (healthy) | 5s |
| backoff inicial | 1s |
| backoff máx | 30s |
| wait-for-device timeout | 30s |

Eventos (EventTarget ou callback store):

```ts
type DeviceConnectionEvent =
  | { type: "connected"; serial: string }
  | { type: "disconnected"; serial: string; reason: "usb" | "tcp" | "unknown" }
  | { type: "reconnecting"; serial: string; attempt: number }
  | { type: "error"; code: string; message: string };
```

Engagement **não** reinicia; UI mostra banner de reconexão.

---

## Camadas

### Backend / lib

```
src/api/pentest/
  physical-device-service.ts   # wrap pentestNative + reconnect loop (browser-safe)
  physical-device-types.ts
```

- Se `typeof window.pentestNative === "undefined"` ou `getStatus().enabled === false` → serviço retorna `unavailable` (modo servidor / flag off).
- Sem `fetch` para Agent Server; exceção de integração local já listada se necessário (preferir só IPC).
- Query keys: `PHYSICAL_DEVICE_QUERY_KEYS` em `query-keys.ts`.

### Frontend

Estender aba Emulador existente (`EmulatorPanel` / toolbar):

| Estado | UI |
|---|---|
| Hooks off / servidor | Empty: device físico indisponível (i18n); emulador container inalterado |
| Hooks on, sem device | CTA “Conectar device” + lista serial + campo host:port TCP |
| Conectado | Badge serial + indicador; stream: reutilizar proxy scrcpy/emulator **se** disponível para serial físico; senão mensagem “espelho via scrcpy host (MVP: status only)” |
| Reconectando | Banner não-bloqueante + attempt |

**MVP espelho:** se scrcpy-web para device físico exigir packaging extra, AC mínimo é **status + seleção + reconnect**; espelho visual é stretch alinhado ao padrão 192 — documentar gap se scrcpy host não estiver pronto.

i18n: keys `EMULATOR$PHYSICAL_*` — `npm run make-i18n`.

Capability: reutilizar gate pentest mobile existente se houver; senão feature flag hooks.

---

## AC testáveis

| ID | Critério |
|---|---|
| AC-194-1 | Sem hooks: UI não oferece connect físico; serviço `unavailable` |
| AC-194-2 | Com mock `pentestNative`, listar devices e selecionar serial persiste no metadata da conversa/engagement |
| AC-194-3 | Simular disconnect → eventos `disconnected` → `reconnecting` → `connected` sem reload da conversa |
| AC-194-4 | Backoff respeita teto 30s (teste unitário do scheduler) |
| AC-194-5 | Não chama IPC `adb shell` / install |

---

## Ordem de trabalho

1. Confirmar branch 193 mergeada ou cherry-pick do módulo `pentest-ipc` + preload types.  
2. Backend: service + reconnect + testes.  
3. Frontend: toolbar/empty states + i18n.  
4. Gates: QA → AppSec (authz/hooks abuse) → Design só se UI nova substancial (revisão leve OK).

---

## Fora de escopo

- Farm remoto  
- IPA  
- Novos verbos IPC além do contrato 193  
- Fase 4

---

## Entrega

PR fork `Plane: PROJETOSIN-194`; comentar Plane com evidência AC e dependência 193 (SHA/PR).
