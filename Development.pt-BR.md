# Guia de Desenvolvimento

Este guia é para pessoas que trabalham no OpenHands e editam o código-fonte.
Se você deseja contribuir com suas alterações, consulte o
[CONTRIBUTING.md](https://github.com/OpenHands/OpenHands/blob/main/CONTRIBUTING.md)
para saber como clonar e configurar o projeto inicialmente antes de prosseguir. Caso contrário,
você pode clonar o projeto OpenHands diretamente.

## Iniciar o Servidor para Desenvolvimento

### 1. Requisitos

- Linux, Mac OS ou [WSL no Windows](https://learn.microsoft.com/en-us/windows/wsl/install) [Ubuntu >= 22.04]
- [Docker](https://docs.docker.com/engine/install/) (Para quem usa MacOS, certifique-se de permitir que o socket padrão do Docker seja usado nas configurações avançadas!)
- [Python](https://www.python.org/downloads/) = 3.12
- [NodeJS](https://nodejs.org/en/download/package-manager) >= 22.x
- [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer) >= 1.8
- Dependências específicas do SO:
  - Ubuntu: build-essential => `sudo apt-get install build-essential python3.12-dev`
  - WSL: netcat => `sudo apt-get install netcat`

Certifique-se de ter todas essas dependências instaladas antes de prosseguir para `make build`.

#### Dev container

Há um [dev container](https://containers.dev/) disponível que fornece um
ambiente pré-configurado com todas as dependências necessárias instaladas, caso você
esteja usando um [editor ou ferramenta compatível](https://containers.dev/supporting). Por
exemplo, se você estiver usando o Visual Studio Code (VS Code) com a
extensão [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
instalada, você pode abrir o projeto em um dev container usando o comando
_Dev Container: Reopen in Container_ no Command Palette (Ctrl+Shift+P).

#### Desenvolver sem acesso sudo

Se você quiser desenvolver sem privilégios de administrador/sudo para atualizar/instalar `Python` e/ou `NodeJS`, pode usar
`conda` ou `mamba` para gerenciar os pacotes:

```bash
# Baixar e instalar o Mamba (uma versão mais rápida do conda)
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh

# Instalar Python 3.12, nodejs e poetry
mamba install python=3.12
mamba install conda-forge::nodejs
mamba install conda-forge::poetry
```

### 2. Build e Configuração do Ambiente

Comece construindo o projeto, o que inclui configurar o ambiente e instalar dependências. Esta etapa garante
que o OpenHands esteja pronto para ser executado no seu sistema:

```bash
make build
```

### 3. Configurando o Modelo de Linguagem

O OpenHands suporta uma variedade de Modelos de Linguagem (LMs) por meio da poderosa biblioteca [litellm](https://docs.litellm.ai).

Para configurar o LM de sua preferência, execute:

```bash
make setup-config
```

Esse comando solicitará que você insira a chave de API do LLM, o nome do modelo e outras variáveis, garantindo que o OpenHands seja
personalizado para suas necessidades. Observe que o nome do modelo será aplicado apenas quando você executar em modo headless. Se você usar a UI,
defina o modelo na própria interface.

Observação: se você já executou o OpenHands usando o comando docker, talvez já tenha definido algumas variáveis de ambiente no seu terminal. As configurações finais são aplicadas da maior para a menor prioridade:
Variáveis de ambiente > variáveis em config.toml > variáveis padrão

**Observação sobre Modelos Alternativos:**
Consulte [nossa documentação](https://docs.all-hands.dev/usage/llms) para modelos recomendados.

### 4. Executando a aplicação

#### Opção A: Executar a Aplicação Completa

Quando a configuração estiver concluída, este comando inicia os servidores backend e frontend, permitindo que você interaja com o OpenHands:

```bash
make run
```

#### Opção B: Inicialização Individual dos Servidores

- **Iniciar o Servidor Backend:** Se preferir, você pode iniciar o servidor backend de forma independente para focar em
  tarefas ou configurações relacionadas ao backend.

  ```bash
  make start-backend
  ```

- **Iniciar o Servidor Frontend:** Da mesma forma, você pode iniciar o servidor frontend isoladamente para trabalhar em componentes
  ou melhorias da interface.
  ```bash
  make start-frontend
  ```

### 5. Executando o OpenHands com o próprio OpenHands

Você pode usar o OpenHands para desenvolver e melhorar o próprio OpenHands! Esta é uma forma poderosa de aproveitar a assistência de IA para contribuir com o projeto.

#### Início Rápido

1. **Build e execução do OpenHands:**

   ```bash
   export INSTALL_DOCKER=0
   export RUNTIME=local
   make build && make run
   ```

2. **Acessar a interface:**

   - Desenvolvimento local: http://localhost:3001
   - Ambientes remotos/cloud: use a URL externa apropriada

3. **Configurar para acesso externo (se necessário):**
   ```bash
   # Para acesso externo (por exemplo, em ambientes cloud)
   make run FRONTEND_PORT=12000 FRONTEND_HOST=0.0.0.0 BACKEND_HOST=0.0.0.0
   ```

### 6. Depuração de LLM

Se você encontrar problemas com o Modelo de Linguagem (LM) ou estiver curioso, exporte `DEBUG=1` no ambiente e reinicie o backend.
O OpenHands registrará os prompts e respostas em `logs/llm/CURRENT_DATE`, permitindo que você identifique as causas.

### 7. Ajuda

Precisa de ajuda ou informações sobre alvos (targets) e comandos disponíveis? Use o comando de ajuda para obter toda a orientação necessária com o OpenHands.

```bash
make help
```

### 8. Testes

Para executar testes, consulte o seguinte:

#### Testes unitários

```bash
poetry run pytest ./tests/unit/test_*.py
```

### 9. Adicionar ou atualizar dependência

1. Adicione sua dependência em `pyproject.toml` ou use `poetry add xxx`.
2. Atualize o arquivo `poetry.lock` via `poetry lock --no-update`.

### 10. Usar imagem Docker existente

Para reduzir o tempo de build (por exemplo, se nenhuma alteração foi feita no componente client-runtime), você pode usar uma imagem Docker
existente configurando a variável de ambiente `SANDBOX_RUNTIME_CONTAINER_IMAGE` para a imagem Docker desejada.

Exemplo: `export SANDBOX_RUNTIME_CONTAINER_IMAGE=ghcr.io/openhands/runtime:1.0-nikolaik`

## Desenvolver dentro de um contêiner Docker

TL;DR

```bash
make docker-dev
```

Veja mais detalhes [aqui](./containers/dev/README.md).

Se você estiver interessado apenas em executar o `OpenHands` sem instalar todas as ferramentas necessárias no host:

```bash
make docker-run
```

Se você não tiver `make` no host, execute:

```bash
cd ./containers/dev
./dev.sh
```

Você precisa ter o [Docker](https://docs.docker.com/engine/install/) instalado no host.

## Principais Recursos de Documentação

Aqui está um guia dos arquivos de documentação importantes no repositório:

- [/README.md](./README.md): Visão geral do projeto, recursos e instruções básicas de configuração
- [/Development.md](./Development.md) (este arquivo): Guia abrangente para desenvolvedores que trabalham no OpenHands
- [/CONTRIBUTING.md](./CONTRIBUTING.md): Diretrizes para contribuir com o projeto, incluindo estilo de código e processo de PR
- [DOC_STYLE_GUIDE.md](https://github.com/All-Hands-AI/docs/blob/main/openhands/DOC_STYLE_GUIDE.md): Padrões para escrever e manter a documentação do projeto
- [/openhands/README.md](./openhands/README.md): Detalhes sobre a implementação backend em Python
- [/frontend/README.md](./frontend/README.md): Guia de configuração e desenvolvimento do aplicativo frontend em React
- [/containers/README.md](./containers/README.md): Informações sobre contêineres Docker e implantação
- [/tests/unit/README.md](./tests/unit/README.md): Guia para escrever e executar testes unitários
- [/evaluation/README.md](./evaluation/README.md): Documentação para o framework de avaliação e benchmarks
- [/skills/README.md](./skills/README.md): Informações sobre a arquitetura e implementação de skills
- [/openhands/server/README.md](./openhands/server/README.md): Detalhes de implementação do servidor e documentação da API
- [/openhands/runtime/README.md](./openhands/runtime/README.md): Documentação sobre o ambiente de runtime e modelo de execução
