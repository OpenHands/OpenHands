# Spec Técnica — PROJETOSIN-186: Dockerfiles Runtimes Ofensivos

**ADR:** docs/adrs/0001-plataforma-pentest-ia-extensao-openhands.md (accepted)
**Card Plane:** PROJETOSIN-186 — `bb8d408b-12d3-475d-b2ee-5732b7acd676`
**Agente responsável:** devops
**Prioridade:** P1 — imagens usadas pelos templates do EngMgr (PROJETOSIN-185)

---

## Objetivo

Criar Dockerfiles das 4 imagens de runtime ofensivo que o Engagement Manager provisiona por tipo de workspace pentest. Cada imagem é **slim por domínio** — sem monolito Kali completo — para reduzir superfície e tempo de boot.

---

## Estrutura de arquivos

```
docker/runtimes/
├── web/
│   ├── Dockerfile
│   └── entrypoint.sh
├── network/
│   ├── Dockerfile
│   └── entrypoint.sh
├── mobile/
│   ├── Dockerfile
│   └── entrypoint.sh
├── sast/
│   ├── Dockerfile
│   └── entrypoint.sh
└── README.md
```

---

## Runtime Web (`ghcr.io/heimdall/runtime-web`)

**Base:** `python:3.12-slim` (Debian bookworm)
**Arsenal:** ZAP (daemon mode), Nuclei, Wapiti, Nikto, sqlmap

```dockerfile
# docker/runtimes/web/Dockerfile
FROM python:3.12-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget git openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# OWASP ZAP
ENV ZAP_VERSION=2.15.0
RUN curl -L "https://github.com/zaproxy/zaproxy/releases/download/v${ZAP_VERSION}/ZAP_${ZAP_VERSION}_Linux.tar.gz" \
    | tar -xz -C /opt && ln -s /opt/ZAP_${ZAP_VERSION}/zap.sh /usr/local/bin/zap

# Nuclei
RUN curl -sSfL https://github.com/projectdiscovery/nuclei/releases/latest/download/nuclei_linux_amd64.zip \
    | unzip -d /usr/local/bin - nuclei && chmod +x /usr/local/bin/nuclei

# Wapiti
RUN pip install --no-cache-dir wapiti3

# Nikto
RUN apt-get update && apt-get install -y --no-install-recommends nikto \
    && rm -rf /var/lib/apt/lists/*

# sqlmap
RUN pip install --no-cache-dir sqlmap

# MCP server (Web)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

WORKDIR /workspace
EXPOSE 8080 8090
ENTRYPOINT ["/entrypoint.sh"]
```

**entrypoint.sh:** Inicia `zap.sh -daemon -host 0.0.0.0 -port 8080` em background; expõe `/healthz`.

---

## Runtime Network (`ghcr.io/heimdall/runtime-network`)

**Base:** `debian:bookworm-slim`
**Arsenal:** nmap, OpenVAS/GVM (scanner only), Metasploit Framework RPC

```dockerfile
# docker/runtimes/network/Dockerfile
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    nmap curl wget git python3 python3-pip \
    gvm openvas-scanner \
    && rm -rf /var/lib/apt/lists/*

# Metasploit Framework
RUN curl -fsSL https://apt.metasploit.com/metasploit-framework.gpg | gpg --dearmor > /usr/share/keyrings/metasploit.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/metasploit.gpg] https://apt.metasploit.com/ buster main" \
    > /etc/apt/sources.list.d/metasploit.list \
    && apt-get update && apt-get install -y metasploit-framework \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
EXPOSE 8091 55553
ENTRYPOINT ["/entrypoint.sh"]
```

**entrypoint.sh:** Inicializa BD do GVM, inicia `openvasmd`, expõe Metasploit via `msfrpcd -P ${MSF_PASSWORD} -S -a 0.0.0.0`.

**Nota:** Build pesado (~2GB). CI deve fazer cache por layer. Imagem separada do OpenVAS scanner para evitar tempo de boot alto em todos os engagements.

---

## Runtime Mobile (`ghcr.io/heimdall/runtime-mobile`)

**Base:** `debian:bookworm-slim`
**Arsenal:** adb, Frida, apktool, jadx, MobSF (via sidecar no compose)

```dockerfile
# docker/runtimes/mobile/Dockerfile
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    android-tools-adb curl python3 python3-pip openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Frida client
RUN pip3 install --no-cache-dir frida-tools

# apktool
ENV APKTOOL_VERSION=2.9.3
RUN curl -L "https://github.com/iBotPeaches/Apktool/releases/download/v${APKTOOL_VERSION}/apktool_${APKTOOL_VERSION}.jar" \
    -o /usr/local/lib/apktool.jar \
    && printf '#!/bin/sh\nexec java -jar /usr/local/lib/apktool.jar "$@"\n' > /usr/local/bin/apktool \
    && chmod +x /usr/local/bin/apktool

# jadx (DEX decompiler)
ENV JADX_VERSION=1.5.0
RUN curl -L "https://github.com/skylot/jadx/releases/download/v${JADX_VERSION}/jadx-${JADX_VERSION}.zip" \
    | unzip -d /opt/jadx - \
    && ln -s /opt/jadx/bin/jadx /usr/local/bin/jadx

WORKDIR /workspace
EXPOSE 8092
ENTRYPOINT ["/entrypoint.sh"]
```

**MobSF:** Corre como **container separado** no compose do engagement (não embutido nesta imagem):
```yaml
# fragment compose mobile
  mobsf:
    image: opensecurity/mobile-security-framework-mobsf:latest
    environment:
      - MOBSF_API_KEY=${MOBSF_API_KEY}
    ports:
      - "8093:8000"
    volumes:
      - mobsf-data:/home/mobsf/.MobSF
```

**Emulador Android:** Também separado como sidecar:
```yaml
  android-emulator:
    image: budtmo/docker-android:emulator_13.0
    privileged: true
    ports:
      - "5555:5555"   # ADB over TCP
      - "6901:6901"   # noVNC (GUI web)
    environment:
      - DEVICE=Samsung Galaxy S10
```

---

## Runtime SAST (`ghcr.io/heimdall/runtime-sast`)

**Base:** `python:3.12-slim`
**Arsenal:** Semgrep, Trivy

```dockerfile
# docker/runtimes/sast/Dockerfile
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git \
    && rm -rf /var/lib/apt/lists/*

# Semgrep
RUN pip install --no-cache-dir semgrep

# Trivy
ENV TRIVY_VERSION=0.57.0
RUN curl -sfL https://github.com/aquasecurity/trivy/releases/download/v${TRIVY_VERSION}/trivy_${TRIVY_VERSION}_Linux-64bit.tar.gz \
    | tar -xz -C /usr/local/bin trivy

WORKDIR /workspace
EXPOSE 8094
ENTRYPOINT ["/entrypoint.sh"]
```

---

## CI — GitHub Actions para build das imagens

```yaml
# .github/workflows/docker-runtimes.yml
name: Build Offensive Runtime Images

on:
  push:
    paths:
      - "docker/runtimes/**"
      - ".github/workflows/docker-runtimes.yml"
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        runtime: [web, network, mobile, sast]
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - name: Build and push
        uses: docker/build-push-action@v6
        with:
          context: docker/runtimes/${{ matrix.runtime }}
          push: ${{ github.ref == 'refs/heads/main' }}
          tags: |
            ghcr.io/heimdall/runtime-${{ matrix.runtime }}:latest
            ghcr.io/heimdall/runtime-${{ matrix.runtime }}:sha-${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

---

## .dockerignore por runtime

```
# docker/runtimes/web/.dockerignore
.git
__pycache__
*.pyc
.env
node_modules
```

---

## Critérios de aceite (QA)

1. **AC-186-1:** `docker build` de cada imagem conclui sem erro
2. **AC-186-2:** Runtime Web: `zap --version` disponível; `nuclei -version` disponível
3. **AC-186-3:** Runtime Network: `nmap --version` disponível; `msfconsole -v` disponível
4. **AC-186-4:** Runtime Mobile: `adb version` disponível; `apktool --version` disponível
5. **AC-186-5:** Runtime SAST: `semgrep --version` disponível; `trivy -v` disponível
6. **AC-186-6:** Todas as imagens abaixo de 2GB (network é exceção: aceita até 4GB por Metasploit)
7. **AC-186-7:** GitHub Actions build matrix verde para todos os 4 runtimes
8. **AC-186-8:** Tags `latest` e `sha-*` publicadas em GHCR

---

## Segurança (AppSec)

- Ferramentas só instaladas de releases oficiais com checksums verificados quando possível
- `RUN as non-root` onde ferramentas permitirem (Metasploit e GVM exigem root/privilege — documentar)
- Runtime Mobile container privileged **apenas** para emulador, não para o runtime mobile principal
- Scan das imagens com `trivy image` na CI antes do push

---

## Dependências

- **Não depende de:** outros cards da Fase 0
- **Desbloqueia:** PROJETOSIN-185 provisioner (precisa das tags de imagem)

**Estimativa:** 2–3 dias (4 Dockerfiles + CI workflow)
