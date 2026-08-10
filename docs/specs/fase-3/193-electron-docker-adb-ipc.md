# Spec Técnica — PROJETOSIN-193: Ponte Electron ↔ Docker/ADB (IPC)

**ADR:** docs/adrs/0001-plataforma-pentest-ia-extensao-openhands.md (accepted) — Implementation notes §1  
**Card Plane:** PROJETOSIN-193 — `ccc26939-e70a-44f8-8124-0c830b376aea`  
**Agentes:** devops (lead) → backend (helpers / allowlist / testes de contrato)  
**Prioridade:** P0 — fundação Fase 3  
**Base git:** `4eb43759b`  
**Branch:** `feat/fase3-electron-ipc-193`  
**Worktree:** `.tmp/worktrees/193`  
**PR target:** fork `klebersjunior/OpenHands` only

---

## Objetivo

Expor ganchos nativos seguros no Electron para orquestrar **Docker Compose do host** e **ADB do host**, com feature-flag e superfície mínima. Estes hooks são a fundação de PROJETOSIN-194 (device físico) e do modo local do blueprint §4.2.

**Não** entregar UI de device físico (194) nem seletor de autonomia (195) neste card.

---

## Premissas

1. Hooks ativos **somente** quando **todas** forem verdadeiras:
   - Processo é Electron packaged/dev desktop (`app.isPackaged` ou flag de desktop)
   - Env `PENTEST_ELECTRON_HOOKS_ENABLED=true`
   - Não é modo servidor / `--public` server-only (sem daemon Docker local do pentester)
2. Fora dessas condições: handlers retornam erro tipado `{ code: "hooks_disabled", status: 403 }` — sem executar nada.
3. Renderer **nunca** recebe `child_process` / `fs` — só API via `contextBridge`.
4. Sem shell arbitrário (`sh -c`, `cmd /c` com string livre). Só argv allowlisted.

---

## Superfície IPC (contrato estável para 194)

### Channels — `ipcMain.handle`

| Channel | Args (JSON) | Resultado | Notas |
|---|---|---|---|
| `pentest:hooks:status` | — | `{ enabled: boolean, dockerAvailable: boolean, adbAvailable: boolean, version: "1" }` | Probe sem side-effect pesado |
| `docker:compose:up` | `{ projectDir: string, file?: string, projectName?: string }` | `{ ok: true, stdout, stderr }` \| error | `docker compose … up -d` |
| `docker:compose:down` | `{ projectDir, file?, projectName? }` | idem | `down` (sem `-v` por default) |
| `docker:compose:ps` | `{ projectDir, file?, projectName? }` | `{ services: Array<{ name, state, health? }> }` | parse estável |
| `adb:devices` | — | `{ devices: Array<{ serial, state, transport? }> }` | `adb devices -l` |
| `adb:connect` | `{ host: string, port?: number }` | `{ serial, state }` | só host:port; port default 5555 |
| `adb:disconnect` | `{ serialOrHost: string }` | `{ ok: true }` | |
| `adb:wait-for-device` | `{ serial?: string, timeoutMs?: number }` | `{ serial, state }` | timeout default 30s, max 120s |

Prefixos ADR (`docker:compose:*`, `adb:*`) + `pentest:hooks:status` para discovery.

### Preload / API renderer

Arquivo novo (além do boot preload): ex. `electron/pentest-preload.cjs` **ou** extensão controlada do preload da janela principal — **não** misturar com `desktopBoot` do loading splash.

```js
// window.pentestNative (só se hooks compile-time/runtime allow)
{
  getStatus(): Promise<HooksStatus>,
  dockerCompose: { up, down, ps },
  adb: { devices, connect, disconnect, waitForDevice },
}
```

Tipagem TS em `src/types/pentest-native.d.ts` (ambient) para o frontend/194 consumir sem magic.

### Feature flag

| Var | Default | Onde |
|---|---|---|
| `PENTEST_ELECTRON_HOOKS_ENABLED` | `false` | env do processo main Electron; documentar em `.env.sample` + AGENTS.md nota curta |

Dev desktop: documentar como ligar (`PENTEST_ELECTRON_HOOKS_ENABLED=true npm run desktop`). Packaged: opt-in explícito (não ligar “por acidente” em builds CI).

---

## Allowlist de segurança (AppSec — crítico)

### Docker

- Binário: resolver `docker` no `PATH` injetado (já existe bridging Node/uv no desktop) — **não** aceitar path de binário do renderer.
- Subcomando: apenas `compose` + verbos `up|down|ps`.
- Flags fixas: `up` → sempre `-d`; proibir `--build` com Dockerfile remoto arbitrário no MVP se aumentar risco — permitir `--build` **só** se `projectDir` estiver sob allowlist de roots.
- `projectDir` / `file`:
  - Path absoluto normalizado
  - Deve estar sob roots permitidos: ex. `~/.openhands/engagements/`, workspace state dir, ou lista em `config/defaults.json` → `pentest.electronComposeRoots`
  - Rejeitar `..`, symlinks que escapem root (resolve + `relative` check)
- `projectName`: regex `^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$`
- Timeout por invoke (ex. 5 min up, 2 min down/ps); kill process tree no timeout
- **Não** passar env arbitrário do renderer para o child

### ADB

- Binário: `adb` do PATH (documentar Android platform-tools no host)
- Verbos: `devices`, `connect`, `disconnect`, `wait-for-device` **apenas**
- `adb:connect` host: IPv4/IPv6/hostname validado; **bloquear** link-local? Permitir LAN; bloquear credenciais em string
- Serial: charset allowlist `[A-Za-z0-9._:-]+`, max len 128
- **Proibido neste card:** `adb shell`, `adb push/pull`, `adb install`, `forward` genérico — isso é 194+/mcp-mobile via endpoint ADB, não IPC livre
- Output truncado (ex. 256 KiB) para não DoS o IPC

### Hardening Electron

- Manter `contextIsolation: true`, `nodeIntegration: false`, sandbox renderer
- Validar `event.senderFrame` / webContents da janela própria (rejeitar invoke de webview estranha)
- Log estruturado sem dump de secrets; não logar paths fora de allowlist em clear se sensíveis

---

## Layout de código sugerido

```
electron/
  main.mjs                 # registerPentestIpc() se flag
  pentest-ipc/
    index.mjs              # register handlers
    guard.mjs              # hooks_enabled + sender check
    docker-compose.mjs     # spawn allowlisted
    adb.mjs                # spawn allowlisted
    path-policy.mjs        # roots / normalize
  pentest-preload.cjs      # contextBridge
config/defaults.json       # pentest.electronComposeRoots (opcional)
__tests__/electron/        # unit tests dos parsers/guards (node env)
```

Testes **sem** exigir Docker/ADB reais: mock `spawn`; testes de path escape, flag off → 403, regex serial.

---

## Backend (após skeleton DevOps)

Escopo backend neste card (mesmo worktree, **depois** do skeleton IPC):

1. Módulo compartilhado de validação (se útil fora do Electron) **ou** apenas documentação do contrato em `docs/specs/fase-3/` (já aqui).
2. Se EngMgr/runtime precisar saber se hooks existem: endpoint **opcional** `GET /api/pentest/electron-hooks/status` via proxy local **somente** quando o launcher Electron injeta um side-car — **MVP pode omitir HTTP** e deixar só `window.pentestNative` (194 lê do renderer). Preferir **omitir HTTP** no MVP para não ampliar superfície.
3. Atualizar `services/mcp-servers/README.md` com nota: device físico depende de ADB endpoint resolvido pelo engagement; IPC 193 não substitui mcp-mobile.

---

## AC testáveis

| ID | Critério |
|---|---|
| AC-193-1 | Com flag off / non-Electron: qualquer `docker:*` / `adb:*` → `hooks_disabled` / 403; zero spawn |
| AC-193-2 | `pentest:hooks:status` reflete flag + availability probes |
| AC-193-3 | `projectDir` fora dos roots → erro `path_not_allowed`; path com `..` rejeitado |
| AC-193-4 | `adb:devices` parseia lista estável; serial inválido em disconnect rejeitado |
| AC-193-5 | Preload expõe só API documentada; sem `require`/`process` no bridge |
| AC-193-6 | Testes unitários dos guards passam em CI sem Docker/ADB |

---

## Gates

| Gate | Foco |
|---|---|
| **AppSec (crítico)** | Superfície IPC/shell: allowlist, path policy, flag, isolation; sem RCE via renderer |
| QA | AC-193-* + regressão Electron boot (`boot-log:*` intacto) |
| Design | N/A (sem UI) |

Tech Lead **não** auto-assina. AppSec revisor ≠ autor do PR.

---

## Fora de escopo

- Reconexão / UI device (194)
- `adb shell` / install via IPC
- Compose genérico fora dos roots
- Corellium / farm

---

## Entrega

1. Implementação + testes no worktree 193  
2. PR no fork com `Plane: PROJETOSIN-193`  
3. Comentário Plane: branch + PR URL  
4. Pedir gate AppSec **antes** de merge (crítico)
