---
card: PROJETOSIN-191
pr: 9
veredicto: PASS
agente: qa
data: 2026-08-10
tip: 4869b7758f5c442bd371adde5355e3b3adf49b71
ci: pytest services/engagement-manager/tests/test_runtime_provisioner.py (9 passed) + cenário QA independente (asyncio render/teardown)
repo: klebersjunior/OpenHands
branch: feat/fase2-emulator-engmgr-191
worktree: .tmp/worktrees/191
---

# QA — PROJETOSIN-191 (Emulador Android + MobSF no compose EngMgr)

**Veredicto:** PASS

Gate de AC/regressão no worktree `.tmp/worktrees/191`, HEAD alinhado ao PR [#9](https://github.com/klebersjunior/OpenHands/pull/9) (`4869b775`). Spec: `docs/specs/fase-2/191-android-emulator-engmgr.md`. Design N/A (sem UI — card 192). **Não** emite AppSec.

Verificação própria: inspeção do template Jinja + `defaults.json` + READMEs + `runtime_provisioner.py`, pytest do autor **e** cenário independente (provision dry-run → assert serviços/env/rede/pins; teardown argv com `-v`). Sem auto-assinatura do relato do implementador.

## Critérios de aceite

| AC | Status | Evidência |
|----|--------|-----------|
| AC-191-1 — compose mobile com 3 serviços (runtime, emulator, mobsf) | **PASS** | Template `compose-mobile-runtime.yml.j2` define `{{ project }}-runtime`, `-emulator`, `-mobsf`. Pytest `test_ac191_mobile_compose_three_services_and_env`. Cenário QA independente: 3 blocos presentes no YAML renderizado. |
| AC-191-2 — runtime recebe `ADB_HOST` / `MOBSF_URL` / `MOBSF_API_KEY` | **PASS** | Env no template: `ADB_HOST="{{ project }}-emulator"`, `ADB_PORT="5555"`, `MOBSF_URL=http://{{ project }}-mobsf:8000`, `MOBSF_API_KEY`. Dry-run injeta `test-mobsf-key` (`DRY_RUN_MOBSF_API_KEY`). Pytest + cenário independente. |
| AC-191-3 — emulator/MobSF só na rede internal; sem publish host | **PASS** | Template **sem** chave `ports:`; comentário explícito AC-191-3. Emulator/MobSF só `network_internal`; runtime também em egress. Pytest `test_ac191_no_host_port_publish` (regex 5555/6080/6901/8000 + ausência egress nos sidecars). Template source: `ports:` ausente. |
| AC-191-4 — pins em defaults/config EngMgr | **PASS** | `config/defaults.json` → `images.androidEmulator=budtmo/docker-android:emulator_13.0`, `images.mobsf=opensecurity/mobile-security-framework-mobsf:latest`. `app/config.py` lê defaults (+ env override). Pytest `test_ac191_defaults_json_image_pins` + assert pins no YAML renderizado. |
| AC-191-5 — README EngMgr + `docker/runtimes/README.md` (KVM, privileged, boot) | **PASS** | `services/engagement-manager/README.md` § Mobile: KVM table, privileged só no emulator, boot 60–180s, teardown `-v`. `docker/runtimes/README.md` § Mobile sidecars: fragmento sem `ports:`, KVM/boot/fallback, teardown `-v`. |
| AC-191-6 — testes unitários do provisioner verdes | **PASS** | `python -m pytest tests/test_runtime_provisioner.py -v` → **9 passed** (0.25s) no worktree 191. Inclui AC mobile, no-ports, dry-run up args, teardown `-v`, pins, KVM fail-fast / slow flag. |
| AC-191-7 — teardown `compose down -v` remove volume MobSF | **PASS** | `RuntimeProvisioner.teardown` args incluem `down`, `-v`. Volume nomeado `{{ volume_prefix }}-mobsf-data` no template. Pytest `test_ac191_teardown_down_v_removes_volumes`. Cenário QA: `docker compose -p eng-… down -v`. |

## Asserções falsificáveis (cenário QA próprio)

| Asserção | Se o controle sumisse… | Resultado |
|----------|------------------------|-----------|
| YAML mobile sem os 3 serviços | asserts `*-runtime/emulator/mobsf` falhariam | PASS |
| Sem env ADB/MOBSF | asserts `ADB_HOST` / `MOBSF_URL` / key falhariam | PASS |
| Host publish reintroduzido | `assert "ports:" not in text` falharia | PASS |
| Pins fora de defaults.json | `test_ac191_defaults_json_image_pins` falharia | PASS |
| Teardown sem `-v` | `assert "-v" in down` falharia | PASS |
| Sem KVM e sem flag | `test_ac191_kvm_missing_fail_fast` espera `RuntimeError` | PASS |

## Regressão

| Checagem | Resultado |
|----------|-----------|
| `pytest tests/test_runtime_provisioner.py` | **PASS** — 9/9 |
| Cenário QA independente (asyncio provision+teardown) | **PASS** |
| npm lint / vitest / E2E mock-LLM | **N/A** — escopo EngMgr Python + compose/docs; sem UI frontend |
| Design gate | **N/A** — sem UI (PROJETOSIN-192) |

## Residual (não bloqueante)

- Token GitHub autenticado no ambiente QA (`klebersjunior`) = autor do commit do PR. Review GitHub formal registrado com o veredicto QA; **APPROVE nativo** pode ser rejeitado por política “revisor ≠ autor” — Tech Lead deve confirmar o review no PR se o GitHub bloquear self-approve.
- AppSec ainda não emitido (privileged emulator, secret MobSF, rede internal) — próximo gate.

## Ação requerida

Nenhuma para AC. **Tech Lead:** AppSec pode seguir no PR #9. Merge só após AppSec PASS.
