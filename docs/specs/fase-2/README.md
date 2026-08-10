# Specs — Fase 2 Mobile (PROJETOSIN-181)

**ADR:** [0001](../../adrs/0001-plataforma-pentest-ia-extensao-openhands.md) (accepted)  
**Blueprint:** §6.4 Mobile — [blueprint](../../product/blueprint-plataforma-pentest-ia.md)  
**Base git:** fork `klebersjunior/OpenHands` tip `dbfa75033` (Fase 0+1 Done)

## Fora de escopo (Fase 2)

- Device físico + ponte Electron ADB (Fase 3)
- Farm Corellium / Genymotion
- IPA / iOS

## Cards

| Card | Spec | Branch | Worktree | Agente(s) |
|------|------|--------|----------|-----------|
| PROJETOSIN-190 | [190-mcp-mobile-mobsf.md](./190-mcp-mobile-mobsf.md) | `feat/fase2-mcp-mobile-190` | `.tmp/worktrees/190` | backend |
| PROJETOSIN-191 | [191-android-emulator-engmgr.md](./191-android-emulator-engmgr.md) | `feat/fase2-emulator-engmgr-191` | `.tmp/worktrees/191` | devops (+ backend) |
| PROJETOSIN-192 | [192-emulator-ui-apk-upload.md](./192-emulator-ui-apk-upload.md) · [192-design-notes.md](./192-design-notes.md) | `feat/fase2-emulator-ui-192` | `.tmp/worktrees/192` | design → frontend |

## Paralelismo

- **Paralelo agora:** 190 · 191 · Design-192 (checkouts isolados).
- **Depois:** Frontend-192 **somente após** `192-design-notes.md` no worktree 192.
- PRs **somente** no fork `klebersjunior/OpenHands`. Gates: Design (192) → QA → AppSec; **sem auto-assinatura**.

## Dependências cruzadas

```
186 runtime-mobile (Done) ──► 190 mcp-mobile + MobSF client
                              191 emulator (+ MobSF) no template EngMgr
191 (ADB/noVNC no compose) ──► 192 aba Emulador + proxy (padrão Desktop)
190 MobSF API + findings   ──► 192 upload APK (dispara scan estático)
```

192 pode mockar endpoints se 190/191 ainda não mergeados; contrato está nas specs.
