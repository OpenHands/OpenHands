# Design Notes — PROJETOSIN-192: Aba Emulador + upload APK

**Status:** pronto para Frontend (definição UX)  
**Card:** PROJETOSIN-192 · `be7bc815-6c37-4e66-a8d6-7994efab19c7`  
**ADR:** [0001](../../adrs/0001-plataforma-pentest-ia-extensao-openhands.md)  
**Spec canônica:** [192-emulator-ui-apk-upload.md](./192-emulator-ui-apk-upload.md)  
**Padrão visual/comportamental:** Desktop VNC — `docs/spec-browser-desktop.md` + `src/components/features/desktop/desktop-panel.tsx`  
**Gate Design PASS:** **não** neste entregável — só após implementação FE + revisão a11y.

---

## 1. Princípios

1. **Espelhar Desktop** — mesma hierarquia mental: painel cheio → CTA sob demanda → iframe dominante; empty state sem spinner infinito.
2. **Uma composição** — na aba Emulador, o stream GUI é o plano dominante; upload é secundário (faixa/rail abaixo ou colapsável), não um dashboard com cards.
3. **Tokens existentes** — HeroUI + `--oh-*` / `var(--oh-muted)`, `var(--foreground)`; sem tema “AI purple”, sem cards decorativos, sem glow.
4. **Capability first** — sem `pentest.mobile.dynamic` a aba **não aparece** (como Desktop/Findings gated). Com capability e sem sidecar → aba visível + empty `EMULATOR$UNAVAILABLE`.
5. **IPA fora** — rejeição client-side imediata (zero POST); copy clara, sem culpar o usuário.

---

## 2. Layout — aba Emulador

### 2.1 Onde vive

- Tab no painel direito da conversa (mesmo `conversation-tabs` que Desktop / Files / Browser / Terminal).
- Ícone sugerido: `Smartphone` (lucide), paralelo a `Monitor` do Desktop.
- Label i18n: `COMMON$EMULATOR` (tooltip + aria + label da tab).
- Wiring espelhando `src/routes/desktop-tab.tsx` → `EmulatorPanel`.

### 2.2 Hierarquia (desktop viewport ≥ ~640px no drawer)

```
┌─ EmulatorPanel (h-full flex-col min-h-0) ─────────────────┐
│ [toolbar fina — só quando live / recoverable error]       │  ← opcional, 36–40px
│ ┌─ EmulatorStage (flex-1 min-h-0) ───────────────────────┐ │
│ │  unavailable | starting | live iframe | error           │ │  ← DOMINANTE
│ └─────────────────────────────────────────────────────────┘ │
│ ┌─ MobileArtifactsRail (shrink-0, max ~160–200px) ───────┐ │
│ │  dropzone + lista compacta de artifacts                 │ │  ← SECUNDÁRIO
│ └─────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────┘
```

**Decisão de produto (Design):** upload fica **na mesma aba**, em rail inferior — não em modal separado nem em Settings. Motivo: o pentester sobe APK e observa o emulador no mesmo contexto; evita navegação extra.

### 2.3 Mobile / drawer estreito

- Empilhar: Stage → Rail (sem side-by-side).
- Rail pode ser um `<details>` / disclosure “APK / artifacts” **fechado por padrão** quando o stage está `live`, para maximizar o iframe; aberto por padrão em `unavailable` / idle (ainda dá valor sem GUI).
- Sem cards; separador = borda `border-[var(--oh-border)]` ou `border-neutral` já usada no app.

### 2.4 Toolbar (live)

Espelhar minimalismo Desktop (hoje Desktop não tem toolbar em live — iframe full). Para Emulador, toolbar **só se necessário**:

| Controlo | Quando | Notas |
|---|---|---|
| Refresh iframe | live | recarrega `src` same-origin `/api/emulator/` |
| Reabrir / Start | error recuperável (não unavailable) | mesmo CTA do idle |
| Open external | **MVP: omitir** | evita surfacing de URL interna; same-origin only |

Não mostrar “stop emulator” no MVP (ciclo de vida = container do engagement).

### 2.5 Componentes (contrato FE — sem implementar aqui)

```
src/components/features/emulator/
  emulator-panel.tsx          # orquestra estados + rail
  emulator-empty-state.tsx    # idle / unavailable / error / starting
  emulator-toolbar.tsx        # refresh (live)
  emulator-apk-upload.tsx     # dropzone + progress + rejeição IPA
  emulator-artifacts-list.tsx # lista compacta (nome, status scan)
```

Test ids (spec): `emulator-panel`, `emulator-unavailable`, `emulator-iframe`, `emulator-start-button`.  
Adicionar: `emulator-apk-dropzone`, `emulator-upload-progress`, `emulator-artifacts-list`, `emulator-scan-status`.

---

## 3. Estados da superfície Emulador (Stage)

Máquina de estados alinhada a `DesktopViewState`:

| Estado | UI | Spinner? | CTA |
|---|---|---|---|
| `loading` (probe inicial) | ícone Smartphone + spinner pequeno centrado | sim (curto) | nenhum |
| `idle` + disponível | mensagem `EMULATOR$OPEN` + botão start | não | `emulator-start-button` com foco inicial |
| `idle` + `unavailable` | `EMULATOR$UNAVAILABLE` + ícone muted | **não** (nunca infinito) | nenhum botão start |
| `starting` | `EMULATOR$STARTING` + spinner | sim | nenhum (desabilitar double-submit) |
| `live` | iframe full do stage (`data-testid="emulator-iframe"`) | não | toolbar refresh |
| `error` + recuperável | `EMULATOR$FAILED` + botão retry (= start) | não | foco no retry |
| `error` + unavailable | mesma copy unavailable | não | sem CTA start |

**Regra de ouro (AC-192-3):** unavailable ≠ loading. Probe falho / 404 proxy → `unavailable`, sem CTA enganoso.

### 3.1 Live — iframe

- `src`: same-origin `/api/emulator/` (nunca URL interna do noVNC no browser).
- `title={t(EMULATOR$IFRAME_TITLE)}` — obrigatório a11y.
- `sandbox` igual Desktop: `allow-scripts allow-same-origin allow-forms allow-popups allow-downloads`.
- `allow="clipboard-read; clipboard-write"` se noVNC exigir; caso contrário omitir.
- Fundo `bg-black` no iframe (como Desktop) para letterbox do stream.
- Foco: após `live`, não roubar foco do chat; usuário Tab até o iframe se quiser interagir.

---

## 4. Fluxo — upload APK (+ rejeição IPA)

### 4.1 Posição

Rail **Mobile artifacts** sob o stage (sempre que a aba Emulador está montada e capability ok). Upload **não** depende do emulador estar `live` — scan estático MobSF pode rodar com stage unavailable (copy auxiliar: `EMULATOR$UPLOAD_HINT_OFFLINE`).

### 4.2 Dropzone

- Controlo único: região focável (`role="button"` ou `<label>` associado a `<input type="file" accept=".apk,application/vnd.android.package-archive" hidden>`).
- Interação:
  - Clique / Enter / Space → abre file picker.
  - Drag-and-drop de arquivo.
- Visual: borda tracejada sutil (`border-dashed`), texto `EMULATOR$UPLOAD_DROPZONE` + helper `EMULATOR$UPLOAD_ACCEPT`; **sem** card com shadow.
- Estados visuais: default · drag-over (`ring` / border accent `--oh-*`) · uploading · error.

### 4.3 Validação client-side (antes de POST)

| Condição | Ação | Copy |
|---|---|---|
| Extensão `.ipa` (case-insensitive) | rejeitar; **zero POST** | `EMULATOR$UPLOAD_REJECT_IPA` |
| Extensão ≠ `.apk` (e ≠ `.aab` se MVP+ off) | rejeitar | `EMULATOR$UPLOAD_REJECT_TYPE` |
| MIME se disponível e claramente IPA | rejeitar | mesma IPA |
| Tamanho > limite (ex. 200 MB config) | rejeitar | `EMULATOR$UPLOAD_REJECT_SIZE` |
| `.apk` ok | POST multipart | progress |

**AAB:** fora do MVP visual — não oferecer no accept; se usuário dropar `.aab`, usar `EMULATOR$UPLOAD_REJECT_TYPE` (ou chave futura `…_AAB` se produto liberar MVP+).

### 4.4 Após upload OK

1. Progresso indeterminado ou `%` se a API reportar → `EMULATOR$UPLOAD_PROGRESS`.
2. Sucesso → item na lista: nome do arquivo (basename only — **nunca** path absoluto do host) + status.
3. Disparo scan MobSF (backend) → status `queued` / `scanning` / `ready` / `failed` na linha do artifact.
4. Toggle opcional **“Instalar no emulador”** (`EMULATOR$INSTALL_TOGGLE`):
   - Só habilitado se stage `live` **ou** emulador anunciado como ready no status.
   - Ao ligar + confirmar: diálogo/confirmação semi-autônoma (`EMULATOR$INSTALL_CONFIRM` / Cancel) → `POST …/install`.
   - Sem emulador: toggle disabled + `EMULATOR$INSTALL_DISABLED_HINT`.

### 4.5 Lista de artifacts

- Lista plana (ul), uma linha por artifact: nome truncado + chip/texto de status (`EMULATOR$SCAN_QUEUED` | `SCANNING` | `READY` | `FAILED`).
- Empty list: uma linha muted `EMULATOR$ARTIFACTS_EMPTY` — sem empty-state ilustrado.
- Link para findings (se 190 expuser): texto `EMULATOR$VIEW_FINDINGS` só quando `ready` e houver deep-link; senão omitir.

### 4.6 Fluxo resumido (happy path)

```
Usuário abre aba Emulador
  → probe GET /api/emulator/status
  → idle: CTA “Abrir Emulador”  OU  unavailable
  → Start → starting → live (iframe)
Usuário (em paralelo) dropa app.apk no rail
  → validação → POST upload → progress → artifact listado
  → scan queued (UI) → scanning → ready
  → [opcional] confirmar install → adb install via API
```

IPA: drop `app.ipa` → toast/inline error `EMULATOR$UPLOAD_REJECT_IPA` → nenhum network call de upload.

---

## 5. Acessibilidade (WCAG 2.1 AA) — checklist FE

- [ ] **Contraste AA** — texto em `var(--oh-muted)` / foreground sobre fundo do painel; erros não só por cor (ícone + texto).
- [ ] **Foco visível** — CTA start / retry / dropzone / refresh com ring de foco do design system (não `outline-none` sem substituto).
- [ ] **Tab order** — toolbar (se houver) → iframe → dropzone → lista → toggle install; coerente top→bottom.
- [ ] **Teclado dropzone** — focável; Enter/Space abrem picker; drag-and-drop é extra, não único path.
- [ ] **iframe `title`** — `EMULATOR$IFRAME_TITLE` sempre presente quando montado.
- [ ] **Labels / ARIA** — botão start com texto visível (não ícone-only); dropzone com `aria-label` ou label visível; progress com `role="status"` / `aria-live="polite"`; erros `aria-live="assertive"` ou associados ao controlo.
- [ ] **Estados** — loading/erro/empty/unavailable distintos; unavailable sem spinner infinito.
- [ ] **Confirmação install** — diálogo focável, Escape cancela, foco retorna ao toggle.
- [ ] **Responsivo** — empilhamento no drawer estreito; touch target ≥ 44×44px no CTA e dropzone.
- [ ] **Sem literal strings** — todas as strings via `t(I18nKey.EMULATOR$…)` / `COMMON$EMULATOR`.
- [ ] **Redução de movimento** — sem animações decorativas obrigatórias; progress pode ser estático.

---

## 6. Tokens / HeroUI

| Uso | Token / padrão |
|---|---|
| Texto secundário / empty | `text-[var(--oh-muted)]` (como Desktop) |
| Ícone empty | lucide `Smartphone`, `aria-hidden`, `text-[var(--oh-muted)]` |
| CTA primário | mesmo padrão Desktop: botão compacto `h-9`, contraste alto (fundo claro / texto escuro **ou** Button HeroUI primary já usado em settings) — **não** inventar variante neon |
| Borda rail / dropzone | `border` + `border-dashed` com cor de borda do tema (`--oh-border` se existir; senão neutral do Tailwind do app) |
| Progress | HeroUI `Progress` **ou** texto + `LoadingSpinner` small — preferir spinner+copy se Progress não estiver no padrão do painel |
| Status scan | texto semântico, não badge colorido saturado; opcional `Chip` HeroUI size sm se já usado em Findings |

**Proibido nesta aba:** cards com shadow multi-layer; grid de “feature tiles”; purple/indigo glow; pills de marketing.

---

## 7. Chaves i18n sugeridas

Prefixo `EMULATOR$…` + uma chave comum para a tab. FE roda `make-i18n` / `check-translation-completeness`. Valores EN/PT de referência (outras locales: copiar EN no MVP se processo permitir, como `DESKTOP$UNAVAILABLE`).

### Tab / chrome

| Key | en (sugestão) | pt (sugestão) |
|---|---|---|
| `COMMON$EMULATOR` | Emulator | Emulador |
| `EMULATOR$IFRAME_TITLE` | Android emulator display | Tela do emulador Android |
| `EMULATOR$OPEN` | Open Emulator | Abrir Emulador |
| `EMULATOR$STARTING` | Starting emulator… | Iniciando emulador… |
| `EMULATOR$UNAVAILABLE` | Emulator is unavailable in this environment. Mobile dynamic analysis needs the engagement Android emulator (Docker/compose). | Emulador indisponível neste ambiente. A análise dinâmica mobile exige o emulador Android do engagement (Docker/compose). |
| `EMULATOR$FAILED` | Failed to start the emulator. Try again or check engagement logs. | Falha ao iniciar o emulador. Tente novamente ou verifique os logs do engagement. |
| `EMULATOR$REFRESH` | Refresh display | Atualizar tela |

### Upload / artifacts

| Key | en | pt |
|---|---|---|
| `EMULATOR$UPLOAD_SECTION` | Mobile artifacts | Artefatos mobile |
| `EMULATOR$UPLOAD_DROPZONE` | Drop an APK here or browse | Solte um APK aqui ou escolha um arquivo |
| `EMULATOR$UPLOAD_ACCEPT` | Android APK only. iOS IPA is not supported. | Somente APK Android. IPA (iOS) não é suportado. |
| `EMULATOR$UPLOAD_HINT_OFFLINE` | You can upload an APK for static analysis even if the emulator is offline. | Você pode enviar um APK para análise estática mesmo com o emulador offline. |
| `EMULATOR$UPLOAD_PROGRESS` | Uploading APK… | Enviando APK… |
| `EMULATOR$UPLOAD_SUCCESS` | APK uploaded | APK enviado |
| `EMULATOR$UPLOAD_FAILED` | Upload failed. Try again. | Falha no upload. Tente novamente. |
| `EMULATOR$UPLOAD_REJECT_IPA` | iOS IPA files are not supported in this phase. Use an Android APK. | Arquivos IPA (iOS) não são suportados nesta fase. Use um APK Android. |
| `EMULATOR$UPLOAD_REJECT_TYPE` | Unsupported file type. Upload an Android APK. | Tipo de arquivo não suportado. Envie um APK Android. |
| `EMULATOR$UPLOAD_REJECT_SIZE` | File exceeds the maximum APK size. | O arquivo excede o tamanho máximo de APK. |
| `EMULATOR$ARTIFACTS_EMPTY` | No APK uploaded yet. | Nenhum APK enviado ainda. |
| `EMULATOR$SCAN_QUEUED` | Scan queued | Scan na fila |
| `EMULATOR$SCAN_SCANNING` | Scanning… | Analisando… |
| `EMULATOR$SCAN_READY` | Static scan ready | Scan estático pronto |
| `EMULATOR$SCAN_FAILED` | Scan failed | Falha no scan |
| `EMULATOR$VIEW_FINDINGS` | View findings | Ver findings |
| `EMULATOR$INSTALL_TOGGLE` | Install on emulator | Instalar no emulador |
| `EMULATOR$INSTALL_CONFIRM` | Install this APK on the running emulator? | Instalar este APK no emulador em execução? |
| `EMULATOR$INSTALL_CONFIRM_ACTION` | Install | Instalar |
| `EMULATOR$INSTALL_CANCEL` | Cancel | Cancelar |
| `EMULATOR$INSTALL_DISABLED_HINT` | Start the emulator before installing. | Inicie o emulador antes de instalar. |
| `EMULATOR$INSTALL_PROGRESS` | Installing on emulator… | Instalando no emulador… |
| `EMULATOR$INSTALL_SUCCESS` | Installed on emulator | Instalado no emulador |
| `EMULATOR$INSTALL_FAILED` | Install failed | Falha na instalação |

Reuse `BUTTON$CANCEL` / confirmações comuns se já existirem — preferir keys partilhadas a duplicar “Cancel”.

---

## 8. Copy / tom

- Direto, operacional (pentester), sem marketing.
- Unavailable: explicar **o quê falta** (emulador do engagement / Docker), não “feature coming soon”.
- IPA: uma frase, sem abrir roadmap iOS.
- Não expor hosts internos, paths absolutos, chaves ou IDs longos de infra na UI (artifact_id pode ficar só em data attributes / debug).

---

## 9. Fora de escopo (UI)

- Device físico / scrcpy / Electron ADB  
- IPA / iOS  
- Farm externo  
- Gravação de vídeo como evidência  
- Open-in-new-window com URL do noVNC  
- Cards de “preset apps” ou marketplace de APKs  

---

## 10. Critérios de pronto para Frontend

| # | Critério | OK? |
|---|---|---|
| 1 | Layout: iframe dominante + rail upload secundário documentado | sim |
| 2 | Estados stage + upload/scan nomeados | sim |
| 3 | Fluxo APK + rejeição IPA | sim |
| 4 | Checklist a11y AA | sim |
| 5 | Keys `EMULATOR$…` / `COMMON$EMULATOR` | sim |
| 6 | Tokens HeroUI / `--oh-*`; sem cards extras | sim |
| 7 | Gate Design PASS emitido | **não** (pós-FE) |

**Pronto-para-FE: sim** — Tech Lead / PM podem despachar Frontend neste worktree após confirmar no card.

---

## 11. Handoff Tech Lead

- Implementar conforme esta nota + [192-emulator-ui-apk-upload.md](./192-emulator-ui-apk-upload.md).
- Mockar `/api/emulator/*` e upload se 190/191 ainda não mergeados.
- Não auto-assinar gate Design; após PR FE, Design revisa e grava `docs/gates/PROJETOSIN-192/design.md`.
