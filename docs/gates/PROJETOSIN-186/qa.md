---
card: PROJETOSIN-186
pr: 4
veredicto: FAIL
agente: qa
data: 2026-08-10
tip: 4355ddaec
ci: local docker smoke (4 imagens) + GHA run 31401756697 (matrix FAIL)
repo: klebersjunior/OpenHands
branch: feat/fase0-devops-186-runtimes
---

# QA — PROJETOSIN-186 (Dockerfiles Runtimes Ofensivos)

**Veredicto:** FAIL

Revisor ≠ autor (implementação: DevOps). Escopo: `docker/runtimes/**` + `.github/workflows/docker-runtimes.yml`. Spec `docs/specs/fase-0/186-dockerfiles-runtimes.md` · ADR-0001 · PR [#4](https://github.com/klebersjunior/OpenHands/pull/4).

Design N/A (sem UI).

## Critérios de aceite

| AC | Status | Evidência |
|----|--------|-----------|
| AC-186-1 `docker build` das 4 imagens | **PASS** | Imagens locais recentes `ghcr.io/heimdall/runtime-{web,network,mobile,sast}:local` presentes e usáveis (IDs `b3656f4b777b`, `612fc4317d9c`, `974680acf0d7`, `aa4e3b848475`) |
| AC-186-2 web: zap + nuclei | **PASS** | `zap -version` → `2.17.0`; `nuclei -version` → `v3.3.9` |
| AC-186-3 network: nmap + msfconsole | **PASS** | `nmap --version` → `7.93`; `msfconsole -v` → `Framework Version: 6.5.2-dev-` |
| AC-186-4 mobile: adb + apktool | **PASS** | `adb version` → `1.0.41` / `29.0.6-debian`; `apktool --version` → `2.9.3` |
| AC-186-5 sast: semgrep + trivy | **PASS** | `semgrep --version` → `1.172.0`; `trivy -v` → `0.73.0` |
| AC-186-6 tamanhos | **PASS*** | mobile `959MB`, sast `981MB`, network `1.85GB` (≤4GB); **web `2.23GB`** — desvio soft &lt;2GB documentado (ZAP + JDK); aceito como nota, não FAIL (alinhavado Tech Lead) |
| AC-186-7 matrix CI verde / coerente | **FAIL** | Workflow matrix `[web, network, mobile, sast]` existe e é coerente; **porém** GHA run [31401756697](https://github.com/klebersjunior/OpenHands/actions/runs/31401756697) falhou nos **4** jobs: `Unable to resolve action aquasecurity/trivy-action@0.28.0`. Tag `0.28.0` **não existe** (releases atuais: `v0.36.0` … `v0.29.0`). Jobs morrem no setup — build/scan/push não executam em CI |
| AC-186-8 tags `latest` + `sha-*` (push main) | **PASS** | Workflow define `IMAGE_LATEST_TAG` / `IMAGE_SHA_TAG` e `push: true` só se `github.ref == 'refs/heads/main' && event != pull_request`. Publicação GHCR no fork/PR não esperada; contrato de tags OK por análise estática |

\* Desvio web ~2.23 GB: soft target 2 GB; causa = arsenal ZAP Linux + OpenJDK. README e AppSec já registram. Tools AC OK → **PASS com nota**.

## Regressão

| Checagem | Resultado |
|----------|-----------|
| Smoke tools nas 4 imagens locais | PASS (comandos AC acima) |
| Workflow `docker-runtimes.yml` estrutura (matrix, tags, push gate, Trivy pré-push) | estrutura OK; **pin Trivy Action inválido** |
| GHA “Build Offensive Runtime Images” no PR | **FAIL** (4/4) |
| npm lint/test/build | N/A — escopo só Docker/CI runtime |

## Asserções falsificáveis

| Asserção | Status | Evidência |
|----------|--------|-----------|
| Entrypoint override alcança binários AC | PASS | `--entrypoint zap|nuclei|nmap|msfconsole|adb|apktool|semgrep|trivy` retorna versão |
| `trivy-action@0.28.0` resolvível no Actions marketplace | **FAIL** | erro GHA + releases API sem tag `0.28.0` |

## Residual (não eleva o FAIL além de AC-186-7)

- web `2.23GB` soft limit — nota documentada.
- Push GHCR só em `main` / org `heimdall` — esperado no fork.
- GVM/OpenVAS sidecar (não no apt bookworm) — fora de AC-186-3; AppSec PASS.

## Ação requerida (bloqueante)

1. **DevOps:** corrigir pin em `.github/workflows/docker-runtimes.yml` — trocar `aquasecurity/trivy-action@0.28.0` por tag existente (ex. `v0.36.0` ou `0.35.0`).
2. Re-disparar workflow no PR e confirmar matrix **Build web/network/mobile/sast** verde (build + Trivy; push continua só em `main`).
3. Re-gate QA após CI verde.

Até lá: label **Blocked** no Plane; merge bloqueado por AC-186-7.
