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
- [Developing in Docker](#developing-in-docker)

---

## macOS Setup

### 1. Install Prerequisites

If you're starting fresh on a new Mac, run these commands in your terminal:

```bash
# Install Homebrew (if you don't have it)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Xcode CLI tools (required for building some dependencies)
xcode-select --install

# Install Python 3.12, Node.js, and Poetry via Homebrew
brew install python@3.12 node poetry

# Ensure Python 3.12 is available
# Add to ~/.zshrc if needed: export PATH="/usr/local/opt/python@3.12/bin:$PATH"

# Install Docker Desktop
# Download from: https://www.docker.com/products/docker-desktop
# After installing, go to Docker Desktop > Settings > General
# Enable: "Allow the default Docker socket to be used"
```

### 2. Build and Setup the Environment

```bash
make build
```

### 3. Configure the Language Model

```bash
make setup-config
```

### 4. Run the Application

```bash
# Run both backend and frontend
make run

# Or run separately:
make start-backend  # Backend only on port 3000
make start-frontend # Frontend only on port 3001
```

---

## Linux Setup

This guide covers Ubuntu/Debian. For other distributions, adapt the package manager commands accordingly.

### 1. Install Prerequisites

```bash
# Update package list
sudo apt update

# Install system dependencies
sudo apt install -y build-essential python3.12-dev python3.12-venv netcat curl

# Install Node.js 22.x
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install Docker
# See: https://docs.docker.com/engine/install/
# For Ubuntu: https://docs.docker.com/engine/install/ubuntu/
sudo apt install -y docker.io docker-compose
sudo usermod -aG docker $USER
# Log out and back in for Docker group changes to take effect
```

### 2. Build and Setup the Environment

```bash
make build
```

### 3. Configure the Language Model

```bash
make setup-config
```

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

```bash
# Update package list
sudo apt update

# Install system dependencies
sudo apt install -y build-essential python3.12-dev python3.12-venv netcat curl

# Install Node.js 22.x
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to your PATH if needed
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

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

```bash
make setup-config
```

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

- [/README.md](./README.md): Main project overview and basic setup
- [/CONTRIBUTING.md](./CONTRIBUTING.md): Contributing guidelines and PR process
- [/frontend/README.md](./frontend/README.md): Frontend React development
- [/openhands/README.md](./openhands/README.md): Backend Python implementation
- [/containers/README.md](./containers/README.md): Docker container information
- [/tests/unit/README.md](./tests/unit/README.md): Unit testing guide
