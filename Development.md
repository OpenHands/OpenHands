# Development Guide

This guide is for people working on OpenHands and editing the source code.
If you wish to contribute your changes, check out the
[CONTRIBUTING.md](https://github.com/OpenHands/OpenHands/blob/main/CONTRIBUTING.md)
on how to clone and setup the project initially before moving on. Otherwise,
you can clone the OpenHands project directly.

## Start the Server for Development

### 1. Requirements

- Linux, Mac OS, or [WSL on Windows](https://learn.microsoft.com/en-us/windows/wsl/install) [Ubuntu >= 22.04]
- [Docker](https://docs.docker.com/engine/install/) (For those on MacOS, make sure to allow the default Docker socket to be used from advanced settings!)
- [Python](https://www.python.org/downloads/) = 3.12
- [NodeJS](https://nodejs.org/en/download/package-manager) >= 22.x
- [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer) >= 1.8
- OS-specific dependencies:
  - Ubuntu: build-essential => `sudo apt-get install build-essential python3.12-dev`
  - WSL: netcat => `sudo apt-get install netcat`

Make sure you have all these dependencies installed before moving on to `make build`.

#### Dev container

There is a [dev container](https://containers.dev/) available which provides a
pre-configured environment with all the necessary dependencies installed if you
are using a [supported editor or tool](https://containers.dev/supporting). For
example, if you are using Visual Studio Code (VS Code) with the
[Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
extension installed, you can open the project in a dev container by using the
_Dev Container: Reopen in Container_ command from the Command Palette
(Ctrl+Shift+P).

#### Develop without sudo access

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

### 2. Build and Setup The Environment

Begin by building the project which includes setting up the environment and installing dependencies. This step ensures
that OpenHands is ready to run on your system:

```bash
make build
```

### 3. Configuring the Language Model

OpenHands supports a diverse array of Language Models (LMs) through the powerful [litellm](https://docs.litellm.ai) library.

To configure the LM of your choice, run:

```bash
make setup-config
```

This command will prompt you to enter the LLM API key, model name, and other variables ensuring that OpenHands is
tailored to your specific needs. Note that the model name will apply only when you run headless. If you use the UI,
please set the model in the UI.

Note: If you have previously run OpenHands using the docker command, you may have already set some environment
variables in your terminal. The final configurations are set from highest to lowest priority:
Environment variables > config.toml variables > default variables

**Note on Alternative Models:**
See [our documentation](https://docs.openhands.dev/usage/llms) for recommended models.

### 4. Running the application

#### Option A: Run the Full Application

Once the setup is complete, this command starts both the backend and frontend servers, allowing you to interact with OpenHands:

```bash
make run
```

#### Option B: Individual Server Startup

- **Start the Backend Server:** If you prefer, you can start the backend server independently to focus on
  backend-related tasks or configurations.

  ```bash
  make start-backend
  ```

- **Start the Frontend Server:** Similarly, you can start the frontend server on its own to work on frontend-related
  components or interface enhancements.
  ```bash
  make start-frontend
  ```

### 5. Running OpenHands with OpenHands

You can use OpenHands to develop and improve OpenHands itself! This is a powerful way to leverage AI assistance for contributing to the project.

#### Quick Start

1. **Build and run OpenHands:**

   ```bash
   export INSTALL_DOCKER=0
   export RUNTIME=local
   make build && make run
   ```

2. **Access the interface:**

   - Local development: http://localhost:3001
   - Remote/cloud environments: Use the appropriate external URL

3. **Configure for external access (if needed):**
   ```bash
   # For external access (e.g., cloud environments)
   make run FRONTEND_PORT=12000 FRONTEND_HOST=0.0.0.0 BACKEND_HOST=0.0.0.0
   ```

### 6. LLM Debugging

If you encounter any issues with the Language Model (LM) or you're simply curious, export DEBUG=1 in the environment and restart the backend.
OpenHands will log the prompts and responses in the logs/llm/CURRENT_DATE directory, allowing you to identify the causes.

### 7. Help

Need help or info on available targets and commands? Use the help command for all the guidance you need with OpenHands.

```bash
make help
```

### 8. Testing

To run tests, refer to the following:

#### Unit tests

```bash
poetry run pytest ./tests/unit/test_*.py
```

### 9. Add or update dependency

1. Add your dependency in `pyproject.toml` or use `poetry add xxx`.
2. Update the poetry.lock file via `poetry lock --no-update`.

### 10. Use existing Docker image

To reduce build time (e.g., if no changes were made to the client-runtime component), you can use an existing Docker
container image by setting the SANDBOX_RUNTIME_CONTAINER_IMAGE environment variable to the desired Docker image.

Example: `export SANDBOX_RUNTIME_CONTAINER_IMAGE=ghcr.io/openhands/runtime:1.2-nikolaik`

## Develop inside Docker container

TL;DR

```bash
make docker-dev
```

See more details [here](./containers/dev/README.md).

If you are just interested in running `OpenHands` without installing all the required tools on your host.

```bash
make docker-run
```

If you do not have `make` on your host, run:

```bash
cd ./containers/dev
./dev.sh
```

You do need [Docker](https://docs.docker.com/engine/install/) installed on your host though.

## Key Documentation Resources

Here's a guide to the important documentation files in the repository:

- [/README.md](./README.md): Main project overview, features, and basic setup instructions
- [/Development.md](./Development.md) (this file): Comprehensive guide for developers working on OpenHands
- [/CONTRIBUTING.md](./CONTRIBUTING.md): Guidelines for contributing to the project, including code style and PR process
- [DOC_STYLE_GUIDE.md](https://github.com/OpenHands/docs/blob/main/openhands/DOC_STYLE_GUIDE.md): Standards for writing and maintaining project documentation
- [/openhands/README.md](./openhands/README.md): Details about the backend Python implementation
- [/frontend/README.md](./frontend/README.md): Frontend React application setup and development guide
- [/containers/README.md](./containers/README.md): Information about Docker containers and deployment
- [/tests/unit/README.md](./tests/unit/README.md): Guide to writing and running unit tests
- [/evaluation/README.md](./evaluation/README.md): Documentation for the evaluation framework and benchmarks
- [/skills/README.md](./skills/README.md): Information about the skills architecture and implementation
- [/openhands/server/README.md](./openhands/server/README.md): Server implementation details and API documentation
- [/openhands/runtime/README.md](./openhands/runtime/README.md): Documentation for the runtime environment and execution model
