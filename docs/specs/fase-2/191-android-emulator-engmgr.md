# Spec Técnica — PROJETOSIN-191: Emulador Android no compose EngMgr

**ADR:** docs/adrs/0001-plataforma-pentest-ia-extensao-openhands.md (accepted)  
**Card Plane:** PROJETOSIN-191 — `f2d470b9-3af1-4fb6-92b2-30fbbf55f37d`  
**Agentes:** devops (lead) + backend (template Jinja / provisioner)  
**Prioridade:** P1 high — sidecar emulator + MobSF no engagement mobile  
**Base git:** `dbfa75033`  
**Branch:** `feat/fase2-emulator-engmgr-191`  
**Worktree:** `.tmp/worktrees/191`  
**PR target:** fork `klebersjunior/OpenHands` only

---

## Objetivo

Completar o template `compose-mobile-runtime.yml.j2` do Engagement Manager para, ao provisionar `runtime_profile=mobile`, subir:

1. **runtime-mobile** (já stubado) — `ghcr.io/heimdall/runtime-mobile:latest`
2. **android-emulator** — ADB TCP + GUI web (noVNC), rede interna do engagement
3. **mobsf** — OWASP MobSF na mesma rede interna
4. Variáveis / labels para o runtime conectar (`ADB_HOST`, `MOBSF_URL`, `MOBSF_API_KEY`)
5. Documentar requisitos de host (KVM / privileged) e fallback degradado (ADR notes)

**Não** entregar UI (192) nem mcp-mobile (190) neste card — só infra compose + testes do provisioner.

---

## Estado atual

`services/engagement-manager/app/templates/compose-mobile-runtime.yml.j2` já tem stub:

- `{{ project }}-runtime` + `{{ project }}-emulator` (imagem placeholder `ghcr.io/heimdall/android-emulator:latest`)
- Redes `internal` + `egress`

Falta: MobSF, portas internas corretas, env, privileged/KVM, healthchecks, secret injection, testes.

---

## Template alvo (contrato)

Serviços na rede `{{ network_internal }}` (e runtime também em egress se já padrão web):

| Serviço | Image (pin documentado) | Portas **internas** | Notas |
|---|---|---|---|
| `{{ project }}-runtime` | `ghcr.io/heimdall/runtime-{{ profile }}:latest` | (agent) | Env: `ADB_HOST={{ project }}-emulator`, `ADB_PORT=5555`, `MOBSF_URL=http://{{ project }}-mobsf:8000`, `MOBSF_API_KEY`, `PENTEST_SCOPE_ALLOWLIST` |
| `{{ project }}-emulator` | `budtmo/docker-android:emulator_13.0` (ou pin em `config/defaults.json` → `images.androidEmulator`) | `5555` ADB, `6080`/`6901` noVNC (conforme imagem) | `privileged: true`; **sem** publish host em modo servidor; só rede Docker |
| `{{ project }}-mobsf` | `opensecurity/mobile-security-framework-mobsf:latest` (pin) | `8000` | Volume `{{ volume_prefix }}-mobsf-data`; API key via env |

```yaml
# Esqueleto — nomes exatos a fechar no PR alinhados ao Jinja existente
services:
  {{ project }}-runtime:
    image: {{ runtime_image }}
    networks: [{{ network_internal }}, {{ network_egress }}]
    environment:
      ALLOW_RULES: {{ allow_rules | tojson }}
      DENY_RULES: {{ deny_rules | tojson }}
      ADB_HOST: "{{ project }}-emulator"
      ADB_PORT: "5555"
      MOBSF_URL: "http://{{ project }}-mobsf:8000"
      MOBSF_API_KEY: "{{ mobsf_api_key }}"
      PENTEST_MCP_MOBILE_CMD: "{{ mcp_mobile_cmd | default('') }}"
    depends_on:
      - {{ project }}-emulator
      - {{ project }}-mobsf

  {{ project }}-emulator:
    image: {{ android_emulator_image }}
    privileged: true
    networks: [{{ network_internal }}]
    environment:
      DEVICE: "{{ emulator_device | default('Samsung Galaxy S10') }}"
      # WEB_VNC / EMULATOR_HEADLESS conforme doc da imagem escolhida
    # NÃO ports: "5555:5555" / "6901:6901" no template de produção
    # (UI 192 acessa via proxy autenticado no ingress)

  {{ project }}-mobsf:
    image: {{ mobsf_image }}
    networks: [{{ network_internal }}]
    environment:
      MOBSF_API_KEY: "{{ mobsf_api_key }}"
    volumes:
      - {{ volume_prefix }}-mobsf-data:/home/mobsf/.MobSF
```

### Provisioner

- `_render` deve gerar `mobsf_api_key` (settings / secrets store do EngMgr — **não** commit). Se dry-run: placeholder `test-mobsf-key`.
- Pins de imagem: preferir `config/defaults.json` (ex. `images.androidEmulator`, `images.mobsf`) lidos pelo EngMgr config — alinhado à regra “defaults.json fonte da verdade”.
- Health: opcional `healthcheck` no emulator (adb wait) — best-effort; documentar boot lento (60–180s).

### KVM / fallback (documentar, não bloquear merge)

| Host | Comportamento |
|---|---|
| Linux com `/dev/kvm` | `devices: [/dev/kvm]` + privileged |
| Sem KVM | Documentar `EMULATOR_ACCEL=false` / fallback ARM lento **só** com flag explícita `allow_slow_emulator=true` no engagement (campo opcional MVP+; MVP pode fail-fast com mensagem clara no provision log) |
| Windows/mac Docker Desktop | Documentar hypervisor; Fase 3 Electron |

---

## Proxy / exposição para UI (coordenação 192)

**Este card** prepara labels/metadata opcionais no compose ou retorno do provisioner:

```json
{
  "emulator": {
    "adb": "{{ project }}-emulator:5555",
    "vnc_internal": "http://{{ project }}-emulator:6080",
    "service_name": "{{ project }}-emulator"
  },
  "mobsf": {
    "url_internal": "http://{{ project }}-mobsf:8000"
  }
}
```

O proxy autenticado `/api/emulator` (análogo a `/api/desktop`) é **implementação 192** (scripts + ingress). 191 só garante que a GUI **não** fica publicada no host sem auth.

---

## Testes

| Teste | Assert |
|---|---|
| `test_runtime_provisioner` mobile | YAML renderizado contém serviços emulator + mobsf + env ADB/MOBSF |
| Snapshot/string | Sem `ports:` host-mapping de 5555/6901 no template prod |
| Dry-run | `docker compose … up -d` args corretos; não exige Docker real |
| Config pins | defaults.json keys documentadas |

---

## Critérios de aceite (QA)

1. **AC-191-1:** Provision `runtime_profile=mobile` gera compose com 3 serviços (runtime, emulator, mobsf).
2. **AC-191-2:** Runtime recebe `ADB_HOST`/`MOBSF_URL`/`MOBSF_API_KEY`.
3. **AC-191-3:** Emulator e MobSF **somente** na rede internal (sem publish host no template default).
4. **AC-191-4:** Pins de imagem em defaults ou config EngMgr (não tags soltas sem doc).
5. **AC-191-5:** README EngMgr / `docker/runtimes/README.md` atualizado (KVM, privileged, boot time).
6. **AC-191-6:** Testes unitários do provisioner verdes.
7. **AC-191-7:** Teardown `compose down -v` remove volume MobSF do projeto.

---

## Segurança (AppSec)

- Privileged só no serviço emulator (justificado); runtime permanece non-root.
- Sem API key MobSF no git; geração/injeção no provision.
- Rede internal: MobSF/VNC não alcançáveis da LAN do host sem passar pelo proxy autenticado (192).
- Não logar `MOBSF_API_KEY` nos outputs do provisioner.

---

## Dependências

- **Depende de:** PROJETOSIN-183 (EngMgr), PROJETOSIN-186 (imagens runtime).
- **Paralelo:** 190 (consome hostnames), Design-192.
- **Bloqueia UX real do emulador:** 192 precisa deste contrato de rede/serviço.

**Estimativa:** 3–4 dias
