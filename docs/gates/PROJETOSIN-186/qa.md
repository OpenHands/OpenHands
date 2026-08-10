---
card: PROJETOSIN-186
pr: 4
veredicto: PASS
agente: qa
data: 2026-08-10
tip: edaa4006f
prev_fail: 104acf231
ci: local docker smoke (4 imagens) + GHA run 31402992878 (matrix 4/4 success)
repo: klebersjunior/OpenHands
branch: feat/fase0-devops-186-runtimes
---

# QA — PROJETOSIN-186 (Dockerfiles Runtimes Ofensivos)

**Veredicto:** PASS

Re-gate após remediação DevOps (`trivy-action@v0.33.1`, commits `942d61b` → `f167828`). Revisor ≠ autor (implementação: DevOps). Escopo: `docker/runtimes/**` + `.github/workflows/docker-runtimes.yml`. Spec `docs/specs/fase-0/186-dockerfiles-runtimes.md` · ADR-0001 · PR [#4](https://github.com/klebersjunior/OpenHands/pull/4).

Design N/A (sem UI). AppSec re-gate PASS (`docs/gates/PROJETOSIN-186/appsec.md`).

## Critérios de aceite

| AC | Status | Evidência |
|----|--------|-----------|
| AC-186-1 `docker build` das 4 imagens | **PASS** | Builds locais + CI build step success nos 4 jobs (run [31402992878](https://github.com/klebersjunior/OpenHands/actions/runs/31402992878)); imagens locais `runtime-*:local` (IDs `b3656f4b777b`, `612fc4317d9c`, `974680acf0d7`, `aa4e3b848475`) |
| AC-186-2 web: zap + nuclei | **PASS** | Smoke local: ZAP `2.17.0`; Nuclei `v3.3.9` (inalterado desde gate FAIL — só CI mudou) |
| AC-186-3 network: nmap + msfconsole | **PASS** | Smoke local: nmap `7.93`; msfconsole `6.5.2-dev-` |
| AC-186-4 mobile: adb + apktool | **PASS** | Smoke local: adb `1.0.41` / `29.0.6-debian`; apktool `2.9.3` |
| AC-186-5 sast: semgrep + trivy | **PASS** | Smoke local: semgrep `1.172.0`; trivy `0.73.0` |
| AC-186-6 tamanhos | **PASS*** | mobile `959MB`, sast `981MB`, network `1.85GB` (≤4GB); **web `2.23GB`** — desvio soft &lt;2GB (ZAP + JDK); nota, não FAIL |
| AC-186-7 matrix CI verde / coerente | **PASS** | Run [31402992878](https://github.com/klebersjunior/OpenHands/actions/runs/31402992878) `conclusion=success` — **Build web / network / mobile / sast** todos `success` (build + Trivy). Pin `aquasecurity/trivy-action@v0.33.1` (fix pós-FAIL em `0.28.0`) |
| AC-186-8 tags `latest` + `sha-*` (push main) | **PASS** | Workflow define `IMAGE_LATEST_TAG` / `IMAGE_SHA_TAG`; push só se `refs/heads/main` && não PR (skipped no PR, esperado) |

\* Desvio web ~2.23 GB: soft target 2 GB; causa = ZAP Linux + OpenJDK. README/AppSec documentam. Tools AC OK.

## Regressão

| Checagem | Resultado |
|----------|-----------|
| Smoke tools nas 4 imagens locais | PASS (gate anterior; Dockerfiles não mudaram no fix CI) |
| Workflow `docker-runtimes.yml` (matrix, tags, push gate, Trivy) | PASS — `trivy-action@v0.33.1` |
| GHA “Build Offensive Runtime Images” | **PASS** 4/4 (run 31402992878 @ `f167828`) |
| npm lint/test/build | N/A — escopo Docker/CI runtime |

## Asserções falsificáveis

| Asserção | Status | Evidência |
|----------|--------|-----------|
| Entrypoint override alcança binários AC | PASS | `--entrypoint zap\|nuclei\|nmap\|msfconsole\|adb\|apktool\|semgrep\|trivy` → versão |
| Matrix CI resolve Trivy Action e completa build+scan | PASS | 4 jobs success; step “Trivy vulnerability scan” success em todos |

## Residual (não bloqueante)

- web `2.23GB` soft limit — nota documentada.
- Push GHCR só em `main` — esperado no fork/PR.
- Trivy `exit-code: "0"` (scan não-bloqueante) — AppSec re-gate PASS como MEDIUM residual; fora do escopo AC QA de tools/matrix.

## Histórico de gate

| Gate | Veredicto | Motivo |
|------|-----------|--------|
| Inicial (`104acf231`) | FAIL | AC-186-7 — `trivy-action@0.28.0` inexistente |
| Re-gate (este) | **PASS** | Matrix verde + ACs 1–6/8 mantidos |

## Ação requerida

Nenhuma. Remover **Blocked** no Plane. Tech Lead: merge após gates QA + AppSec PASS.
