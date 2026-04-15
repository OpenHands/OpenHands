# Development Guide

This guide is for people working on OpenHands and editing the source code.
If you wish to contribute your changes, check out the
[CONTRIBUTING.md](https://github.com/OpenHands/OpenHands/blob/main/CONTRIBUTING.md)
on how to clone and setup the project initially before moving on. Otherwise,
you can clone the OpenHands project directly.

## Choose Your Setup

Select your operating system to see the specific setup instructions:

- [macOS](#macos-setup)
- [Linux](#linux-setup)
- [Windows WSL](#windows-wsl-setup)
- [Dev Container](#dev-container)
- [Developing in Docker](#developing-in-docker)
- [No sudo access?](#develop-without-sudo-access)

---

## macOS Setup

### 1. Install Prerequisites

You'll need the following installed:

- **Python 3.12** — `brew install python@3.12` (see the [official Homebrew Python docs](https://docs.brew.sh/Homebrew-and-Python) for details). Make sure `python3.12` is available in your PATH (the `make build` step will verify this).
- **Node.js >= 22** — `brew install node`
- **Poetry >= 1.8** — `brew install poetry`
- **Docker Desktop** — `brew install --cask docker`
  - After installing, open Docker Desktop → **Settings → Advanced** → Enable **"Allow the default Docker socket to be used"**

### 2. Build and Setup the Environment

```bash
make build
```

### 3. Configure the Language Model

OpenHands supports a diverse array of Language Models (LMs) through the powerful [litellm](https://docs.litellm.ai) library.

For the V1 web app, start OpenHands and configure your model and API key in the Settings UI.

If you are running headless or CLI workflows, you can prepare local defaults with:

```bash
make setup-config
```

**Note on Alternative Models:**
See [our documentation](https://docs.openhands.dev/usage/llms) for recommended models.

### 4. Run the Application

```bash
# Run both backend and frontend
make run

# Or run separately:
make start-backend  # Backend only on port 3000
make start-frontend # Frontend only on port 3001
```

These targets serve the current OpenHands V1 API by default. In the codebase, `make start-backend` runs `openhands.server.listen:app`, and that app includes the `openhands/app_server` V1 routes unless `ENABLE_V1=0`.

---

## Linux Setup

This guide covers Ubuntu/Debian. For other distributions, adapt the package manager commands accordingly.

### 1. Install Prerequisites

```bash
# Update package list
sudo apt update

# Install system dependencies
sudo apt install -y build-essential curl netcat software-properties-common

# Install Python 3.12
# Ubuntu 24.04+ and Debian 13+ ship with Python 3.12 — skip the PPA step if
# python3.12 --version already works on your system.
# The deadsnakes PPA is Ubuntu-only and needed for Ubuntu 22.04 or older:
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.12 python3.12-dev python3.12-venv

# Install Node.js 22.x
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to your PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Install Docker
# Follow the official guide: https://docs.docker.com/engine/install/ubuntu/
# Quick version:
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
# Log out and back in for Docker group changes to take effect
```

### 2. Build and Setup the Environment

```bash
make build
```

### 3. Configure the Language Model

See the [macOS section above](#3-configure-the-language-model) for guidance: configure your model and API key in the Settings UI.

### 4. Run the Application

```bash
# Run both backend and frontend
make run

# Or run separately:
make start-backend  # Backend only on port 3000
make start-frontend # Frontend only on port 3001
```

---

## Windows WSL Setup

WSL2 with Ubuntu is recommended. The setup is similar to Linux, with a few WSL-specific considerations.

### 1. Install WSL2

**Option A: Windows 11 (Microsoft Store)**
The easiest way on Windows 11:
1. Open the **Microsoft Store** app
2. Search for **"Ubuntu 22.04 LTS"** or **"Ubuntu"**
3. Click **Install**
4. Launch Ubuntu from the Start menu

**Option B: PowerShell**
```powershell
# Run this in PowerShell as Administrator
wsl --install -d Ubuntu-22.04
```

After installation, restart your computer and open Ubuntu.

### 2. Install Prerequisites (in WSL Ubuntu)

Follow [Step 1 from the Linux setup](#1-install-prerequisites-1) to install system dependencies, Python 3.12, Node.js, and Poetry. Skip the Docker installation — Docker is provided through Docker Desktop below.

### 3. Configure Docker for WSL2

1. Install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)
2. Open Docker Desktop > Settings > General
3. Enable: "Use the WSL 2 based engine"
4. Go to Settings > Resources > WSL Integration
5. Enable integration with your Ubuntu distribution

**Important:** Keep your project files in the WSL filesystem (e.g., `~/workspace/openhands`), not in `/mnt/c`. Files accessed via `/mnt/c` will be significantly slower.

### 4. Build and Setup the Environment

```bash
make build
```

### 5. Configure the Language Model

See the [macOS section above](#3-configure-the-language-model) for the current V1 guidance: configure your model and API key in the Settings UI for the web app, and use `make setup-config` only for headless or CLI workflows.

### 6. Run the Application

```bash
# Run both backend and frontend
make run

# Or run separately:
make start-backend  # Backend only on port 3000
make start-frontend # Frontend only on port 3001
```

Access the frontend at `http://localhost:3001` from your Windows browser.

---

## Dev Container

There is a [dev container](https://containers.dev/) available which provides a
pre-configured environment with all the necessary dependencies installed if you
are using a [supported editor or tool](https://containers.dev/supporting). For
example, if you are using Visual Studio Code (VS Code) with the
[Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
extension installed, you can open the project in a dev container by using the
_Dev Container: Reopen in Container_ command from the Command Palette
(Ctrl+Shift+P).

---

## Developing in Docker

If you don't want to install dependencies on your host machine, you can develop inside a Docker container.

### Quick Start

```bash
make docker-dev
```

For more details, see the [dev container documentation](./containers/dev/README.md).

### Alternative: Docker Run

If you just want to run OpenHands without setting up a dev environment:

```bash
make docker-run
```

If you don't have `make` installed, run:

```bash
cd ./containers/dev
./dev.sh
```

---

## Develop without sudo access

If you want to develop without system admin/sudo access to upgrade/install `Python` and/or `NodeJS`, you can use
`conda` or `mamba` to manage the packages for you:

```bash
# Download and install Mamba (a faster version of conda)
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh

# Install Python 3.12, nodejs, and poetry
mamba install python=3.12
mamba install conda-forge::nodejs
mamba install conda-forge::poetry
```

---

## Running OpenHands with OpenHands

You can use OpenHands to develop and improve OpenHands itself!

### Quick Start

```bash
export INSTALL_DOCKER=0
export RUNTIME=local
make build && make run
```

Access the interface at:
- Local development: http://localhost:3001
- Remote/cloud environments: Use the appropriate external URL

For external access:
```bash
make run FRONTEND_PORT=12000 FRONTEND_HOST=0.0.0.0 BACKEND_HOST=0.0.0.0
```

---

## LLM Debugging

If you encounter issues with the Language Model, enable debug logging:

```bash
export DEBUG=1
# Restart the backend
make start-backend
```

Logs will be saved to `logs/llm/CURRENT_DATE/` for troubleshooting.

---

## Testing

### Unit Tests

```bash
poetry run pytest ./tests/unit/test_*.py
```

---

## Adding Dependencies

1. Add your dependency in `pyproject.toml` or use `poetry add xxx`
2. Update the lock file: `poetry lock --no-update`

---

## Using Existing Docker Images

To reduce build time, you can use an existing runtime image:

```bash
export SANDBOX_RUNTIME_CONTAINER_IMAGE=ghcr.io/openhands/runtime:1.2-nikolaik
```

---

## Help

```bash
make help
```

---

## Key Documentation Resources

- [/README.md](./README.md): Main project overview, features, and basic setup instructions
- [/Development.md](./Development.md) (this file): Comprehensive guide for developers working on OpenHands
- [/CONTRIBUTING.md](./CONTRIBUTING.md): Guidelines for contributing to the project, including code style and PR process
- [DOC_STYLE_GUIDE.md](https://github.com/OpenHands/docs/blob/main/openhands/DOC_STYLE_GUIDE.md): Standards for writing and maintaining project documentation
- [/openhands/app_server/README.md](./openhands/app_server/README.md): Current V1 application server implementation and REST API modules
- [/frontend/README.md](./frontend/README.md): Frontend React application setup and development guide
- [/containers/README.md](./containers/README.md): Information about Docker containers and deployment
- [/tests/unit/README.md](./tests/unit/README.md): Guide to writing and running unit tests
- [OpenHands/benchmarks](https://github.com/OpenHands/benchmarks): Documentation for the evaluation framework and benchmarks
- [/skills/README.md](./skills/README.md): Information about the skills architecture and implementation
- [/openhands/runtime/README.md](./openhands/runtime/README.md): Documentation for the runtime environment and execution model
