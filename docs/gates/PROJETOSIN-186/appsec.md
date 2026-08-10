---
card: PROJETOSIN-186
pr: 4
veredicto: PASS
agente: appsec
data: 2026-08-10
ci: review-manual docker/runtimes + docker-runtimes.yml; trivy-CRITICAL-gate
repo: klebersjunior/OpenHands
branch: feat/fase0-devops-186-runtimes
---

# AppSecurity — PROJETOSIN-186 (Dockerfiles Runtimes Ofensivos)

**Veredicto:** PASS

Revisor ≠ autor (implementação: DevOps). Escopo: `docker/runtimes/**` + `.github/workflows/docker-runtimes.yml`. Spec `docs/specs/fase-0/186-dockerfiles-runtimes.md` · ADR-0001.

## Checklist

- [x] Sem segredos versionados / hardcoded (`MSF_PASSWORD`, tokens, API keys)
- [x] `MSF_PASSWORD` só via env em runtime; msfrpcd não sobe sem a variável
- [x] `.dockerignore` em web/network/mobile/sast cobre `.env`
- [x] Tags `ghcr.io/heimdall/runtime-{web,network,mobile,sast}` (+ `latest` / `sha-*`) alinhadas EngMgr
- [x] CI usa `GITHUB_TOKEN` para GHCR; sem secrets extras desnecessários
- [x] Trivy image scan **antes** do push; push só em `main` (não em PR)
- [x] web / mobile / sast: `USER runtime` (uid 1000)
- [x] network: root documentado (Metasploit / tooling privilegiado)
- [x] `privileged: true` só no fragmento documentado do emulador Android sidecar — **não** no `runtime-mobile`
- [x] Sem critical/high não mitigado / não documentado como residual aceitável

## Findings

### Sem critical / high

Nenhum finding bloqueante. Sem senhas, tokens ou chaves bakeadas nas imagens.

### MEDIUM — ZAP API sem chave (`api.disablekey=true`) + bind `0.0.0.0:8080`

`docker/runtimes/web/entrypoint.sh` inicia o daemon ZAP com API key desabilitada e escuta em todas as interfaces. Quem alcançar a porta 8080 controla o scanner sem autenticação.

**Residual aceitável** sob premissa de isolamento de rede do engagement (EngMgr / compose sem publish público de 8080). Remediação futura recomendada: `ZAP_API_KEY` (ou equivalente) via env e documentar no README que 8080 é só overlay interno.

### MEDIUM — Checksums incompletos em downloads

- **ZAP 2.17.0:** SHA-256 verificado — OK.
- Nuclei / Trivy / apktool / jadx: versões pinadas, fontes oficiais GitHub, **sem** `sha256sum -c`.
- Nikto: `git clone --depth 1` (HEAD flutuante) — risco de supply-chain residual.
- pip packages (`wapiti3`, `sqlmap`, `frida-tools`, `semgrep`): sem pin de versão.

Aceitável na Fase 0 com pins de versão + fontes oficiais; backlog: checksums onde o release publicar digests e pin de commit para Nikto.

### MEDIUM (intencional) — Trivy CI falha só em CRITICAL

Workflow: `severity: CRITICAL`, `ignore-unfixed: true`. Comentário no YAML explica HIGH esperados em imagens ofensivas. Gate AppSec ainda tem bloqueio em CRITICAL pré-push. Não eleva a FAIL.

### LOW / nota — GVM ausente no apt bookworm-slim

Dockerfile tenta `openvas-scanner` / `gvm-tools` e continua com nmap+msf se apt falhar. README recomenda sidecar Greenbone. AC-186-3 (nmap + msfconsole) permanece. **Risco documentado aceitável** — não é FAIL (critério do card).

### LOW / nota — Tamanho web ~2.2 GB

Soft target 2 GB; web ~2.2 GB (ZAP + JDK) documentado. Critério do gate: nota, não critical.

### LOW — `msfrpcd` em `0.0.0.0` quando `MSF_PASSWORD` set

Protegido por senha de runtime; superfície depende de publish consciente da porta 55553 pelo provisioner.

## Dependências / CI

- Escopo Docker/workflow: sem `npm audit` aplicável ao diff.
- Auth CI: `docker/login-action` + `secrets.GITHUB_TOKEN`; permissions `contents:read`, `packages:write`, `security-events:write`.
- Matrix build → Trivy → push condicional (`refs/heads/main` && não PR).

## Superfície (resumo)

| Imagem | User | Privileged | Secrets |
|--------|------|------------|---------|
| runtime-web | non-root `runtime` | não | nenhum bakeado |
| runtime-network | root (doc) | não | `MSF_PASSWORD` env-only |
| runtime-mobile | non-root `runtime` | não (emulator sidecar only) | `MOBSF_API_KEY` só no sidecar doc |
| runtime-sast | non-root `runtime` | não | nenhum bakeado |

## Ação requerida

Nenhuma para merge deste gate. Tech Lead: residual ZAP API key e checksums podem virar follow-ups não bloqueantes (EngMgr isolation / hardening Fase 1).
