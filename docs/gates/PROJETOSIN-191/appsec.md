---
card: PROJETOSIN-191
pr: 9
veredicto: PASS
agente: appsec
data: 2026-08-10
tip: 9a52f10a0de82f8111123b416910f391faa24ce8
ci: inspeção estática template+provisioner+defaults (sem npm audit — escopo EngMgr Python/compose)
pr_review: comment (approve bloqueado se mesmo login do autor)
repo: klebersjunior/OpenHands
branch: feat/fase2-emulator-engmgr-191
worktree: .tmp/worktrees/191
---

# AppSecurity — PROJETOSIN-191 (Emulador Android + MobSF no compose EngMgr)

**Veredicto:** PASS

Gate de segurança no worktree `.tmp/worktrees/191`, PR [#9](https://github.com/klebersjunior/OpenHands/pull/9). Spec § Segurança: `docs/specs/fase-2/191-android-emulator-engmgr.md`. **Não** reutiliza o PASS de QA como evidência — superfície própria abaixo. Design N/A (sem UI; proxy autenticado = PROJETOSIN-192).

## Checklist

| Item | Status |
|------|--------|
| Sem segredos versionados / hardcoded | **PASS** |
| `npm audit` high/critical | **N/A** — diff sem alteração npm; escopo Python/Jinja/Docker docs |
| Session key / MobSF key não vazada em bundle público | **PASS** — EngMgr server-side; dry-run só `test-mobsf-key` |
| Proxies autenticados; VNC não exposto na LAN do host | **PASS** (neste card) — sem `ports:`; VNC/ADB só rede internal; proxy UI = 192 |
| Cloud / callCloudProxy | **N/A** |
| Logs sem secrets | **PASS** |

## Achados (superfície própria)

### 1. Segredos — PASS

- `MOBSF_API_KEY` gerada em runtime via `secrets.token_urlsafe(32)` (`RuntimeProvisioner._mobsf_api_key`); dry-run usa placeholder `test-mobsf-key` (constante de teste, não credencial de produção).
- Nenhum segredo real no git (scan de padrões no escopo EngMgr/config/docker/docs: só placeholders / Jinja).
- `runtime-metadata.json` **não** inclui a API key (teste `test_ac191_dry_run_compose_up_args` + inspeção de `build_mobile_network_metadata`).
- Respostas HTTP de provision/sandbox-status expõem apenas `sandbox_compose_project` / status — **não** o YAML com a key.
- Provisioner não loga stdout/stderr do compose nem a key (`_default_runner` descarta pipes; sem `logger` com key).

### 2. Exposição de portas — PASS

- Template `compose-mobile-runtime.yml.j2`: **zero** chaves `ports:`; comentário explícito AC-191-3.
- Contagem estática: `ports:` = 0; sem mapeamento host `5555` / `6901` / `6080` / `8000`.
- Emulator e MobSF apenas em `{{ network_internal }}`; rede `internal: true`. Runtime também em egress (esperado).

### 3. Privileged / non-root — PASS

- `privileged: true` aparece **uma** vez, no serviço `{{ project }}-emulator`.
- Runtime e MobSF sem privileged.
- Documentação: `docker/runtimes/README.md` (privilege model: mobile non-root uid 1000; sidecars) + `services/engagement-manager/README.md` (“Privileged somente no serviço emulator”).

### 4. KVM fail-closed / slow flag — PASS

- Sem `/dev/kvm` e `allow_slow_emulator=False` → `RuntimeError` claro (fail-closed).
- `ALLOW_SLOW_EMULATOR=1` omite `/dev/kvm` e seta `EMULATOR_ACCEL=false`.
- Dry-run assume KVM para fixtures estáveis (não bypass de produção).

### 5. Supply chain / pins — PASS com residual Medium

- Pins em `config/defaults.json`: `images.androidEmulator=budtmo/docker-android:emulator_13.0`, `images.mobsf=opensecurity/mobile-security-framework-mobsf:latest`.
- Override via env `ANDROID_EMULATOR_IMAGE` / `MOBSF_IMAGE`.
- **Residual Medium (não bloqueante):** tag floating `:latest` no MobSF — aceitável para MVP documentado; follow-up: pin digest/`sha-` quando estabilizar imagem.

### 6. Persistência operacional — residual Medium (não bloqueante)

- Compose escrito em `compose_work_dir` contém `MOBSF_API_KEY` em claro (necessário para `docker compose`). Dir default `/tmp/engmgr-compose`. Follow-up: `chmod 0600` no YAML + dir 0700; não é critical/high neste escopo (serviço já monta docker.sock — trust boundary EngMgr).

## Classificação

| ID | Severidade | Bloqueia? | Nota |
|----|------------|-----------|------|
| APPSEC-191-01 | Medium | Não | MobSF `:latest` floating |
| APPSEC-191-02 | Medium | Não | Compose on-disk com API key; endurecer perms em follow-up |

**Critical / High:** nenhum.

## Ação requerida

Nenhuma para merge deste card. **Tech Lead** pode mesclar após gates aplicáveis (QA PASS + AppSec PASS). Card 192 deve manter VNC/ADB atrás de proxy autenticado (não reintroduzir `ports:` no template prod).

**Não mergeado por este gate.**
