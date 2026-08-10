---
card: PROJETOSIN-186
pr: 4
veredicto: PASS
agente: appsec
data: 2026-08-10
re_gate: 2026-08-10
fix_commit: f1678280f
ci: review-manual docker/runtimes + docker-runtimes.yml; trivy-scan-advisory
repo: klebersjunior/OpenHands
branch: feat/fase0-devops-186-runtimes
---

# AppSecurity — PROJETOSIN-186 (Dockerfiles Runtimes Ofensivos)

**Veredicto:** PASS

Revisor ≠ autor (implementação: DevOps). Escopo: `docker/runtimes/**` + `.github/workflows/docker-runtimes.yml`. Spec `docs/specs/fase-0/186-dockerfiles-runtimes.md` · ADR-0001.

## Re-gate 2026-08-10 (delta CI Trivy — `f1678280f`)

Delta único: `trivy-action@v0.33.1`, binary `version: v0.73.0`, **`exit-code: "0"`** (scan não falha a matrix em CRITICAL).

### Decisão sobre `exit-code: "0"`

**PASS — residual MEDIUM aceitável.** Não é FAIL / REQUEST_CHANGES.

Razões:

1. A spec AppSec exige scan **antes do push**, não fail-closed em CVE CRITICAL: *“Scan das imagens com `trivy image` na CI antes do push”*.
2. O step Trivy permanece **antes** do push; findings CRITICAL continuam no log do job (visibilidade operacional).
3. Imagens ofensivas (ZAP, Metasploit, Nuclei, etc.) acumulam CRITICAL conhecidos nos próprios scanners; `exit-code: "1"` torna a matrix estruturalmente vermelha sem remover o arsenal.
4. Não introduz segredo, AuthZ fraca nem superfície nova nas imagens.

**Não aceitável seria:** remover o step Trivy, ou push sem scan. Isso **não** ocorreu.

### Checklist (delta)

- [x] Scan Trivy ainda executa antes do push
- [x] Push condicional só em `main` (não PR)
- [x] Pins de action/binary documentados (`@v0.33.1` / `v0.73.0`)
- [x] Sem mudança de Dockerfiles / secrets neste delta

---

## Checklist (gate inicial — mantido)

- [x] Sem segredos versionados / hardcoded (`MSF_PASSWORD`, tokens, API keys)
- [x] `MSF_PASSWORD` só via env em runtime; msfrpcd não sobe sem a variável
- [x] `.dockerignore` em web/network/mobile/sast cobre `.env`
- [x] Tags `ghcr.io/heimdall/runtime-{web,network,mobile,sast}` (+ `latest` / `sha-*`) alinhadas EngMgr
- [x] CI usa `GITHUB_TOKEN` para GHCR; sem secrets extras desnecessários
- [x] Trivy image scan **antes** do push (modo advisory / `exit-code: "0"`); push só em `main`
- [x] web / mobile / sast: `USER runtime` (uid 1000)
- [x] network: root documentado (Metasploit / tooling privilegiado)
- [x] `privileged: true` só no fragmento documentado do emulador Android sidecar — **não** no `runtime-mobile`
- [x] Sem critical/high não mitigado / não documentado como residual aceitável

## Findings

### Sem critical / high (bloqueantes de merge)

Nenhum finding bloqueante nas imagens/workflow. Sem senhas, tokens ou chaves bakeadas.

### MEDIUM — Trivy CI advisory (`exit-code: "0"`) — **aceito no re-gate**

Workflow: `severity: CRITICAL`, `ignore-unfixed: true`, `exit-code: "0"`. Scan não bloqueia merge/push; CRITICAL no log. Gate AppSec humano + laudo cobrem o processo. Follow-up opcional: upload SARIF / job separado `continue-on-error` com resumo no PR sem falhar o build de publish.

### MEDIUM — ZAP API sem chave (`api.disablekey=true`) + bind `0.0.0.0:8080`

`docker/runtimes/web/entrypoint.sh` inicia o daemon ZAP com API key desabilitada e escuta em todas as interfaces. Quem alcançar a porta 8080 controla o scanner sem autenticação.

**Residual aceitável** sob premissa de isolamento de rede do engagement (EngMgr / compose sem publish público de 8080). Remediação futura: `ZAP_API_KEY` via env.

### MEDIUM — Checksums incompletos em downloads

- **ZAP 2.17.0:** SHA-256 verificado — OK.
- Nuclei / Trivy / apktool / jadx: versões pinadas, sem `sha256sum -c`.
- Nikto: `git clone --depth 1` (HEAD flutuante).
- pip packages sem pin de versão.

Aceitável na Fase 0; backlog de hardening.

### LOW / nota — GVM ausente no apt bookworm-slim

Documentado + sidecar Greenbone. **Aceitável** — não é FAIL.

### LOW / nota — Tamanho web ~2.2 GB

Soft target; nota, não critical.

### LOW — `msfrpcd` em `0.0.0.0` quando `MSF_PASSWORD` set

Protegido por senha de runtime; publish consciente da 55553.

## Dependências / CI

- Escopo Docker/workflow: sem `npm audit` aplicável ao diff.
- Auth CI: `docker/login-action` + `secrets.GITHUB_TOKEN`.
- Matrix: build → Trivy (advisory) → push condicional (`main`).

## Superfície (resumo)

| Imagem | User | Privileged | Secrets |
|--------|------|------------|---------|
| runtime-web | non-root `runtime` | não | nenhum bakeado |
| runtime-network | root (doc) | não | `MSF_PASSWORD` env-only |
| runtime-mobile | non-root `runtime` | não (emulator sidecar only) | `MOBSF_API_KEY` só no sidecar doc |
| runtime-sast | non-root `runtime` | não | nenhum bakeado |

## Ação requerida

Nenhuma. `exit-code: "0"` **não** reabre FAIL. Tech Lead: merge gate AppSec permanece PASS.
